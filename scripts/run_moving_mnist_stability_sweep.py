#!/usr/bin/env python3
"""Run the controlled uniform-stability and foreground-objective sweeps."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Variant:
    label: str
    display_name: str
    learning_rate: float
    ema_start: float
    foreground_weight: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/moving_mnist/stability_base.yaml",
    )
    parser.add_argument("--stage", choices=["optimizer", "foreground"], default="optimizer")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--view", choices=["compact", "full"], default="compact")
    parser.add_argument("--x-axis", default="clips_seen")
    parser.add_argument("--foreground-weights", default="1,2,4")
    parser.add_argument("--foreground-learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--foreground-ema-start", type=float, default=0.996)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def optimizer_variants() -> list[Variant]:
    return [
        Variant(
            label="lr5e4_ema990",
            display_name="LR 5e-4 · EMA .990",
            learning_rate=5e-4,
            ema_start=0.990,
        ),
        Variant(
            label="lr25e4_ema990",
            display_name="LR 2.5e-4 · EMA .990",
            learning_rate=2.5e-4,
            ema_start=0.990,
        ),
        Variant(
            label="lr5e4_ema996",
            display_name="LR 5e-4 · EMA .996",
            learning_rate=5e-4,
            ema_start=0.996,
        ),
        Variant(
            label="lr25e4_ema996",
            display_name="LR 2.5e-4 · EMA .996",
            learning_rate=2.5e-4,
            ema_start=0.996,
        ),
    ]


def foreground_variants(args: argparse.Namespace) -> list[Variant]:
    weights = [
        float(item.strip())
        for item in args.foreground_weights.split(",")
        if item.strip()
    ]
    if not weights:
        raise ValueError("foreground-weights must contain at least one value")
    return [
        Variant(
            label=f"fgw{weight:g}_lr{args.foreground_learning_rate:g}_ema{args.foreground_ema_start:g}",
            display_name=f"Foreground weight {weight:g}",
            learning_rate=float(args.foreground_learning_rate),
            ema_start=float(args.foreground_ema_start),
            foreground_weight=weight,
        )
        for weight in weights
    ]


def generated_config(
    base: dict,
    variant: Variant,
    *,
    seed: int,
    stage: str,
    output_root: Path,
) -> dict:
    config = copy.deepcopy(base)
    config.setdefault("experiment", {})
    config.setdefault("training", {})
    config.setdefault("objective", {})
    config.setdefault("sampling", {})
    config["experiment"]["seed"] = seed
    config["experiment"]["output_root"] = str(output_root)
    config["experiment"]["run_label"] = variant.label
    config["experiment"]["display_name"] = variant.display_name
    config["experiment"]["sweep_stage"] = stage
    config["training"]["learning_rate"] = variant.learning_rate
    config["training"]["encoder_learning_rate"] = variant.learning_rate
    config["training"]["predictor_learning_rate"] = variant.learning_rate
    config["training"]["ema_start"] = variant.ema_start
    if float(config["training"].get("ema_end", variant.ema_start)) < variant.ema_start:
        config["training"]["ema_end"] = variant.ema_start
    config["objective"]["foreground_weight"] = variant.foreground_weight
    config["objective"]["target_weighting_mode"] = (
        "foreground" if variant.foreground_weight > 1.0 else "uniform"
    )
    config["sampling"]["strategy"] = "uniform_shuffle"
    config["sampling"]["score_update_mode"] = "oracle_epochwise"
    return config


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    base = yaml.safe_load(config_path.read_text())
    base_output_root = Path(base["experiment"]["output_root"])
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else base_output_root / args.stage
    )
    generated_root = output_root / "_generated_configs"
    generated_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    variants = optimizer_variants() if args.stage == "optimizer" else foreground_variants(args)

    for variant_index, variant in enumerate(variants):
        for seed in seeds:
            config = generated_config(
                base,
                variant,
                seed=seed,
                stage=args.stage,
                output_root=output_root,
            )
            run_label = str(config["experiment"]["run_label"])
            run_name = f"{run_label}_seed{seed}"
            generated_path = generated_root / f"{run_name}.yaml"
            generated_path.write_text(yaml.safe_dump(config, sort_keys=False))
            run_dir = output_root / run_name

            if not args.skip_pretrain:
                command = [
                    sys.executable,
                    "scripts/pretrain_moving_mnist.py",
                    "--config",
                    str(generated_path),
                    "--device",
                    args.device,
                    "--run-name",
                    run_name,
                ]
                if args.overwrite:
                    command.append("--overwrite")
                run(command)

            if not args.skip_probes:
                command = [
                    sys.executable,
                    "scripts/eval_moving_mnist_probes.py",
                    "--run-dir",
                    str(run_dir),
                    "--checkpoints",
                    "all",
                    "--device",
                    args.device,
                ]
                if variant_index == 0:
                    command.append("--include-random-init")
                run(command)

    if not args.skip_plot:
        run(
            [
                sys.executable,
                "scripts/plot_moving_mnist_benchmark.py",
                "--root",
                str(output_root),
                "--x-axis",
                args.x_axis,
                "--view",
                args.view,
            ]
        )


if __name__ == "__main__":
    main()
