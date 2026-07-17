"""Command-line interface for single JEPA runs and experiment sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from jepa import __version__
from jepa.configs.base import load_config
from jepa.reporting import run_sweep
from jepa.training.core import train_experiment


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
        help="replace only the exact run or sweep directory",
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
    train.add_argument("--resume-from", help="checkpoint.pt from an incomplete CPU run")

    sweep = commands.add_parser("sweep", help="run the full JEPA policy/model matrix")
    _add_common(sweep)
    sweep.add_argument("--seeds", type=int, nargs="+", required=True)
    sweep.add_argument("--system-kind", choices=("linear", "nonlinear", "all"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, overrides=_overrides(args))
        if args.command == "train":
            train_result = train_experiment(config, resume_from=args.resume_from)
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
        sweep_result = run_sweep(config, seeds=args.seeds, system_kind=args.system_kind)
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
