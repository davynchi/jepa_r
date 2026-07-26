#!/usr/bin/env python3
"""Run low-LR confirmation and predictor-LR × motion-weighting sweeps."""

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
    encoder_learning_rate: float
    predictor_learning_rate: float
    ema_start: float
    target_weighting_mode: str = "uniform"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/moving_mnist/motion_base.yaml")
    parser.add_argument("--stage", choices=["confirm", "objective"], default="confirm")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-lowshot", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--objective-ema-start", type=float, default=0.990)
    parser.add_argument("--checkpoint-epochs", default="1,4,7,10")
    parser.add_argument("--budgets", default="128,512,2048")
    parser.add_argument("--split-seeds", default="0,1,2")
    parser.add_argument("--random-init-seeds", default="0,1,2")
    parser.add_argument("--plot-budget", type=int, default=512)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def variants(args: argparse.Namespace) -> list[Variant]:
    if args.stage == "confirm":
        return [
            Variant(
                label="lr25e4_ema990",
                display_name="LR 2.5e-4 · EMA .990",
                encoder_learning_rate=2.5e-4,
                predictor_learning_rate=2.5e-4,
                ema_start=0.990,
            ),
            Variant(
                label="lr25e4_ema996",
                display_name="LR 2.5e-4 · EMA .996",
                encoder_learning_rate=2.5e-4,
                predictor_learning_rate=2.5e-4,
                ema_start=0.996,
            ),
        ]
    ema = float(args.objective_ema_start)
    return [
        Variant(
            label="enc25_pred25_uniform",
            display_name="Enc 2.5e-4 · Pred 2.5e-4 · uniform",
            encoder_learning_rate=2.5e-4,
            predictor_learning_rate=2.5e-4,
            ema_start=ema,
            target_weighting_mode="uniform",
        ),
        Variant(
            label="enc25_pred50_uniform",
            display_name="Enc 2.5e-4 · Pred 5e-4 · uniform",
            encoder_learning_rate=2.5e-4,
            predictor_learning_rate=5e-4,
            ema_start=ema,
            target_weighting_mode="uniform",
        ),
        Variant(
            label="enc25_pred25_motion",
            display_name="Enc 2.5e-4 · Pred 2.5e-4 · motion",
            encoder_learning_rate=2.5e-4,
            predictor_learning_rate=2.5e-4,
            ema_start=ema,
            target_weighting_mode="motion",
        ),
        Variant(
            label="enc25_pred50_motion",
            display_name="Enc 2.5e-4 · Pred 5e-4 · motion",
            encoder_learning_rate=2.5e-4,
            predictor_learning_rate=5e-4,
            ema_start=ema,
            target_weighting_mode="motion",
        ),
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
    config["training"]["encoder_learning_rate"] = variant.encoder_learning_rate
    config["training"]["predictor_learning_rate"] = variant.predictor_learning_rate
    config["training"]["learning_rate"] = variant.encoder_learning_rate
    config["training"]["ema_start"] = variant.ema_start
    if float(config["training"].get("ema_end", variant.ema_start)) < variant.ema_start:
        config["training"]["ema_end"] = variant.ema_start
    config["objective"]["target_weighting_mode"] = variant.target_weighting_mode
    config["sampling"]["strategy"] = "uniform_shuffle"
    config["sampling"]["score_update_mode"] = "oracle_epochwise"
    return config


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(Path(args.config).read_text())
    default_root = Path(base["experiment"]["output_root"])
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else default_root / args.stage
    )
    generated_root = output_root / "_generated_configs"
    generated_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    sweep_variants = variants(args)
    random_reference_written = False

    for variant in sweep_variants:
        for seed in seeds:
            config = generated_config(
                base,
                variant,
                seed=seed,
                stage=args.stage,
                output_root=output_root,
            )
            run_name = f"{variant.label}_seed{seed}"
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

            if not args.skip_lowshot:
                command = [
                    sys.executable,
                    "scripts/eval_moving_mnist_lowshot.py",
                    "--run-dir",
                    str(run_dir),
                    "--checkpoints",
                    "all",
                    "--checkpoint-epochs",
                    args.checkpoint_epochs,
                    "--budgets",
                    args.budgets,
                    "--split-seeds",
                    args.split_seeds,
                    "--device",
                    args.device,
                ]
                if not random_reference_written:
                    command.extend(
                        [
                            "--include-random-init",
                            "--random-init-seeds",
                            args.random_init_seeds,
                            "--include-raw-baselines",
                        ]
                    )
                    random_reference_written = True
                run(command)

    if not args.skip_plot:
        run(
            [
                sys.executable,
                "scripts/plot_moving_mnist_motion_benchmark.py",
                "--root",
                str(output_root),
                "--budget",
                str(args.plot_budget),
            ]
        )


if __name__ == "__main__":
    main()
