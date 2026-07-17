from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    experiment = config["experiment"]
    data = config["data"]
    model = config["model"]
    training = config["training"]
    sampling = config["sampling"]

    output_root = Path(experiment["output_root"])

    if args.overwrite and output_root.exists():
        import shutil
        shutil.rmtree(output_root)

    for strategy in sampling["strategies"]:
        cmd = [
            "uv", "run", "python", "scripts/run_surprise_sampling.py",
            "--architecture", str(model["architecture"]),
            "--epochs", str(training["epochs"]),
            "--batch-size", str(training["batch_size"]),
            "--learning-rate", str(training["learning_rate"]),
            "--ema-decay", str(training["ema_decay"]),
            "--train-easy-samples", str(data["train_easy_samples"]),
            "--train-hard-samples", str(data["train_hard_samples"]),
            "--train-noise-samples", str(data["train_noise_samples"]),
            "--validation-samples-per-group", str(data["validation_samples_per_group"]),
            "--test-samples-per-group", str(data["test_samples_per_group"]),
            "--sequence-length", str(data["sequence_length"]),
            "--context-steps", str(data["context_steps"]),
            "--latent-dim", str(model["latent_dim"]),
            "--hidden-dim", str(model["hidden_dim"]),
            "--hidden-layers", str(model["hidden_layers"]),
            "--lp-delta", str(training["lp_delta"]),
            "--lp-ema-beta", str(training["lp_ema_beta"]),
            "--sampling-strategy", str(strategy),
            "--sampling-strength", str(sampling["strength"]),
            "--sampling-max-ratio", str(sampling["max_ratio"]),
            "--warmup-epochs", str(sampling["warmup_epochs"]),
            "--hard-cap", str(sampling["hard_cap"]),
            "--noise-cap", str(sampling["noise_cap"]),
            "--target-update", str(experiment["target_update"]),
            "--seed", str(data["seed"]),
            "--output-root", str(output_root),
        ]

        print("running:", strategy, flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
