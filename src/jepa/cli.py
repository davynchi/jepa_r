"""Command-line interface for single JEPA runs and experiment sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from jepa import __version__
from jepa.binance import prepare_binance_data
from jepa.config import BinanceDataConfig, load_config
from jepa.reporting import run_sweep
from jepa.training import train_experiment


def _key_value(value: str) -> tuple[str, str]:
    key, separator, raw = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("overrides must use KEY=VALUE")
    return key, raw


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in args.set_values:
        if key in values:
            raise ValueError(f"duplicate override: {key}")
        values[key] = value
    direct = {
        "architecture": getattr(args, "architecture", None),
        "sg": getattr(args, "sg", None),
        "ema": getattr(args, "ema", None),
        "objective": getattr(args, "objective", None),
        "system_kind": (
            args.system_kind
            if getattr(args, "system_kind", None) in {"linear", "nonlinear"}
            else None
        ),
        "output.root": getattr(args, "output_root", None),
        "output.overwrite": getattr(args, "overwrite", None),
    }
    for key, value in direct.items():
        if value is None:
            continue
        if key in values:
            raise ValueError(f"conflicting command-line values for {key}")
        values[key] = value
    return values


def _duration(seconds: object) -> str:
    if not isinstance(seconds, (int, float)):
        return "--:--"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _terminal_progress(event: Mapping[str, Any]) -> None:
    """Render stable progress lines to stderr while keeping stdout machine-readable."""
    kind = event["event"]
    has_cell = "cell_index" in event
    if kind == "cell_start":
        print(
            f"Cell {event['cell_index']}/{event['cell_count']} | {event['objective']} | "
            f"{event['architecture']} | SG {'on' if event['stop_gradient'] else 'off'} | "
            f"EMA {'on' if event['ema_enabled'] else 'off'}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "run_start" and not has_cell:
        print(
            f"Run | {event['objective']} | {event['architecture']} | "
            f"SG {'on' if event['stop_gradient'] else 'off'} | "
            f"EMA {'on' if event['ema_enabled'] else 'off'}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "epoch_complete":
        mse = event.get("train_mse")
        mse_label = "n/a" if mse is None else f"{float(mse):.6g}"
        print(
            f"  Epoch {event['epoch']}/{event['total_epochs']} | train MSE {mse_label} | "
            f"elapsed {_duration(event.get('elapsed_seconds'))} | "
            f"ETA {_duration(event.get('eta_seconds'))}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "cell_reused":
        print("  Reused completed run", file=sys.stderr, flush=True)
    elif kind == "cell_complete":
        print(
            f"  Complete | elapsed {_duration(event.get('wall_clock_seconds'))}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "cell_failed":
        print(
            f"  Failed | {event['error_type']}: {event['error_message']}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "run_complete" and not has_cell:
        print(
            f"Complete | elapsed {_duration(event.get('wall_clock_seconds'))}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "run_failed" and not has_cell:
        print(
            f"Failed | {event['error_type']}: {event['error_message']}",
            file=sys.stderr,
            flush=True,
        )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        type=_key_value,
        metavar="KEY=VALUE",
        help="strict dotted configuration override; may be repeated",
    )
    parser.add_argument("--output-root", help="artifact root directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="start a fresh timestamped run instead of reusing a compatible sweep",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jepa", description="Time-series JEPA experiments")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train one JEPA variant")
    _add_common(train)
    train.add_argument("--architecture", choices=("linear", "nonlinear"))
    train.add_argument("--sg", choices=("on", "off"))
    train.add_argument("--ema", choices=("on", "off"))
    train.add_argument("--system-kind", choices=("linear", "nonlinear"))
    train.add_argument("--objective", choices=("future_window", "masked_patches"))
    train.add_argument("--resume-from", help="checkpoint.pt from an incomplete CPU run")

    sweep = commands.add_parser("sweep", help="run the full JEPA policy/model matrix")
    _add_common(sweep)
    sweep.add_argument("--seeds", type=int, nargs="+", required=True)
    sweep.add_argument("--system-kind", choices=("linear", "nonlinear", "all"))
    sweep.add_argument(
        "--objectives",
        choices=("future_window", "masked_patches", "all"),
        default="configured",
        help="override the configured objective or run both objectives",
    )

    prepare = commands.add_parser("prepare-data", help="prepare an immutable dataset bundle")
    _add_common(prepare)
    prepare.add_argument("--offline", action="store_true")
    prepare.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, overrides=_overrides(args))
        if args.command == "prepare-data":
            if not isinstance(config.data, BinanceDataConfig):
                raise ValueError("prepare-data currently requires data.source: binance_spot")
            result = prepare_binance_data(
                config.data,
                offline=args.offline,
                force=args.force,
            )
            print(
                json.dumps(
                    {
                        "state": "complete",
                        "fingerprint": result.fingerprint,
                        "dataset_dir": str(result.directory),
                        "sample_counts": result.split_samples,
                        "reused": result.reused,
                    }
                )
            )
            return 0
        if args.command == "train":
            train_result = train_experiment(
                config,
                resume_from=args.resume_from,
                progress=_terminal_progress,
            )
            print(
                json.dumps(
                    {
                        "state": "complete",
                        "run_id": train_result.run_id,
                        "run_dir": str(train_result.run_dir),
                    }
                )
            )
            return 0
        sweep_result = run_sweep(
            config,
            seeds=args.seeds,
            system_kind=args.system_kind,
            objectives=args.objectives,
            progress=_terminal_progress,
        )
        print(
            json.dumps(
                {
                    "state": ("complete" if sweep_result.failed_count == 0 else "partial_failure"),
                    "sweep_dir": str(sweep_result.sweep_dir),
                    "completed_count": sweep_result.completed_count,
                    "failed_count": sweep_result.failed_count,
                }
            )
        )
        return 1 if sweep_result.failed_count else 0
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
