#!/usr/bin/env python3
"""Low-shot temporal-linear evaluation for Moving-MNIST JEPA checkpoints."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from jepa.moving_mnist import MovingMNISTDataset, MovingMNISTSpec, make_fake_digit_bank
from jepa.video_jepa import CausalVideoJEPA, CausalVideoJEPAConfig
from jepa.video_probes import (
    ProbeConfig,
    evaluate_low_shot_feature_variant,
    extract_frozen_features,
)
from jepa.video_utils import load_yaml, resolve_device, seed_everything, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoints", choices=["all", "latest"], default="all")
    parser.add_argument(
        "--checkpoint-epochs",
        help="optional comma-separated epoch subset, e.g. 1,4,7,10",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--budgets", help="comma-separated labeled budgets")
    parser.add_argument("--split-seeds", help="comma-separated low-shot split seeds")
    parser.add_argument("--include-random-init", action="store_true")
    parser.add_argument("--random-init-seeds", help="comma-separated random encoder seeds")
    parser.add_argument("--include-raw-baselines", action="store_true")
    parser.add_argument(
        "--controls-checkpoint",
        choices=["none", "first", "latest", "all"],
        default="latest",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return default
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one integer")
    return result


def make_model_config(config: dict[str, Any]) -> CausalVideoJEPAConfig:
    data = config["data"]
    model = config["model"]
    objective = config.get("objective", {})
    foreground_weight = float(objective.get("foreground_weight", 1.0))
    return CausalVideoJEPAConfig(
        image_size=int(data["canvas_size"]),
        total_frames=int(data["total_frames"]),
        context_frames=int(data["context_frames"]),
        in_channels=1,
        patch_size=int(model["patch_size"]),
        tubelet_size=int(model["tubelet_size"]),
        embed_dim=int(model["embed_dim"]),
        encoder_depth=int(model["encoder_depth"]),
        predictor_depth=int(model["predictor_depth"]),
        num_heads=int(model["num_heads"]),
        mlp_ratio=float(model.get("mlp_ratio", 4.0)),
        dropout=float(model.get("dropout", 0.0)),
        position_embedding_type=str(model.get("position_embedding_type", "separate_3d")),
        target_weighting_mode=str(
            objective.get(
                "target_weighting_mode",
                "foreground" if foreground_weight > 1.0 else "uniform",
            )
        ),
        foreground_weight=foreground_weight,
        foreground_threshold=float(objective.get("foreground_threshold", 0.05)),
        motion_weight_min=float(objective.get("motion_weight_min", 0.25)),
        motion_weight_max=float(objective.get("motion_weight_max", 4.0)),
        motion_weight_power=float(objective.get("motion_weight_power", 1.0)),
        motion_weight_epsilon=float(objective.get("motion_weight_epsilon", 1e-6)),
    )


def make_dataset(
    config: dict[str, Any],
    split: str,
    num_samples: int,
    seed: int,
) -> MovingMNISTDataset:
    data = config["data"]
    spec = MovingMNISTSpec(
        num_samples=num_samples,
        total_frames=int(data["total_frames"]),
        context_frames=int(data["context_frames"]),
        canvas_size=int(data["canvas_size"]),
        digit_size=int(data["digit_size"]),
        min_speed=float(data["min_speed"]),
        max_speed=float(data["max_speed"]),
        num_directions=int(data["num_directions"]),
        seed=seed,
    )
    kwargs: dict[str, Any] = {}
    source = str(data.get("source", "mnist"))
    if source == "fake":
        kwargs["digit_images"], kwargs["digit_labels"] = make_fake_digit_bank(
            samples_per_class=int(data.get("fake_samples_per_class", 2)),
            size=int(data["digit_size"]),
        )
        kwargs["download"] = False
    elif source == "mnist":
        kwargs["download"] = bool(data.get("download", True))
    else:
        raise ValueError(f"unknown data source: {source}")
    return MovingMNISTDataset(
        root=data["root"],
        split=split,  # type: ignore[arg-type]
        spec=spec,
        **kwargs,
    )


def make_loader(
    dataset: MovingMNISTDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader[dict[str, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def discover_checkpoints(
    run_dir: Path,
    mode: str,
    requested_epochs: set[int] | None,
) -> list[Path]:
    paths = sorted(
        run_dir.joinpath("checkpoints").glob("epoch_*.pt"),
        key=checkpoint_epoch,
    )
    if requested_epochs is not None:
        paths = [path for path in paths if checkpoint_epoch(path) in requested_epochs]
    if not paths:
        latest = run_dir / "checkpoints" / "latest.pt"
        if latest.exists() and requested_epochs is None:
            return [latest]
        raise FileNotFoundError(f"no matching checkpoints found in {run_dir}")
    return [paths[-1]] if mode == "latest" else paths


def make_probe_config(config: dict[str, Any], seed: int) -> ProbeConfig:
    section = config.get("probe", {})
    lowshot = config.get("lowshot", {})
    return ProbeConfig(
        classifier_epochs=int(lowshot.get("classifier_epochs", 50)),
        classifier_batch_size=int(lowshot.get("classifier_batch_size", 128)),
        classifier_lr=float(lowshot.get("classifier_lr", 0.01)),
        classifier_weight_decay=float(lowshot.get("classifier_weight_decay", 1e-4)),
        ridge_lambdas=tuple(
            float(value)
            for value in lowshot.get(
                "ridge_lambdas",
                section.get("ridge_lambdas", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
            )
        ),
        raw_spatial_pool=int(lowshot.get("raw_spatial_pool", 8)),
        seed=seed,
    )


def load_model(
    model_config: CausalVideoJEPAConfig,
    checkpoint_path: Path | None,
    device: torch.device,
) -> tuple[CausalVideoJEPA, dict[str, int]]:
    model = CausalVideoJEPA(model_config).to(device)
    if checkpoint_path is None:
        return model, {"epoch": 0, "step": 0, "clips_seen": 0, "scoring_clips": 0}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model, {
        "epoch": int(checkpoint["epoch"]),
        "step": int(checkpoint["step"]),
        "clips_seen": int(checkpoint["clips_seen"]),
        "scoring_clips": int(checkpoint.get("scoring_clips", 0)),
    }


def extract_splits(
    model: CausalVideoJEPA,
    loaders: dict[str, DataLoader[dict[str, Tensor]]],
    device: torch.device,
    *,
    probe_config: ProbeConfig,
    include_controls: bool,
    include_raw: bool,
) -> dict[str, dict[str, Tensor]]:
    return {
        split: extract_frozen_features(
            model,
            loader,
            device,
            include_temporal_features=True,
            include_raw_features=include_raw,
            include_shuffled_context_features=include_controls,
            raw_spatial_pool=probe_config.raw_spatial_pool,
        )
        for split, loader in loaders.items()
    }


def append_variant_rows(
    rows: list[dict[str, Any]],
    splits: dict[str, dict[str, Tensor]],
    *,
    feature_key: str,
    feature_variant: str,
    model_kind: str,
    checkpoint: str,
    metadata: dict[str, int],
    budgets: list[int],
    split_seeds: list[int],
    num_directions: int,
    probe_config: ProbeConfig,
    strategy: str,
    run_label: str,
    display_name: str,
    training_seed: int,
) -> None:
    for budget in budgets:
        for split_seed in split_seeds:
            metrics, hyperparameters = evaluate_low_shot_feature_variant(
                splits["probe_train"],
                splits["probe_validation"],
                splits["probe_evaluation"],
                feature_key=feature_key,
                budget=budget,
                split_seed=split_seed,
                num_directions=num_directions,
                config=probe_config,
            )
            common = {
                "checkpoint": checkpoint,
                "epoch": metadata["epoch"],
                "step": metadata["step"],
                "clips_seen": metadata["clips_seen"],
                "scoring_clips": metadata["scoring_clips"],
                "strategy": strategy,
                "run_label": run_label,
                "display_name": display_name,
                "training_seed": training_seed,
                "model_kind": model_kind,
                "feature_variant": feature_variant,
                "budget": budget,
                "split_seed": split_seed,
                "split": "probe_evaluation",
            }
            for metric, value in metrics.items():
                rows.append({**common, "metric": metric, "value": value})
            for metric, value in hyperparameters.items():
                rows.append(
                    {
                        **common,
                        "split": "hyperparameter",
                        "metric": metric,
                        "value": value,
                    }
                )


def controls_enabled(mode: str, index: int, total: int) -> bool:
    return (
        mode == "all"
        or (mode == "first" and index == 0)
        or (mode == "latest" and index == total - 1)
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    experiment = config["experiment"]
    data = config["data"]
    probe = config.get("probe", {})
    lowshot = config.get("lowshot", {})
    seed = int(experiment.get("seed", 0))
    strategy = str(config["sampling"]["strategy"])
    run_label = str(experiment.get("run_label", strategy))
    display_name = str(
        experiment.get("display_name", strategy.replace("_", " ").title())
    )
    device = resolve_device(args.device)
    seed_everything(seed)

    budgets = parse_int_list(
        args.budgets,
        [int(value) for value in lowshot.get("budgets", [128, 512, 2048])],
    )
    split_seeds = parse_int_list(
        args.split_seeds,
        [int(value) for value in lowshot.get("split_seeds", [0, 1, 2])],
    )
    random_init_seeds = parse_int_list(
        args.random_init_seeds,
        [int(value) for value in lowshot.get("random_init_seeds", [0, 1, 2])],
    )
    requested_epochs = (
        set(parse_int_list(args.checkpoint_epochs, []))
        if args.checkpoint_epochs is not None
        else None
    )
    checkpoints = discover_checkpoints(run_dir, args.checkpoints, requested_epochs)
    model_config = make_model_config(config)
    probe_config = make_probe_config(config, seed + 701)

    split_specs = {
        "probe_train": (int(probe.get("train_samples", 8_000)), seed + 307),
        "probe_validation": (
            int(probe.get("validation_samples", 2_000)),
            seed + 401,
        ),
        "probe_evaluation": (
            int(probe.get("evaluation_samples", 2_000)),
            seed + 503,
        ),
    }
    workers = int(lowshot.get("num_workers", probe.get("num_workers", 0)))
    batch_size = int(lowshot.get("batch_size", probe.get("batch_size", 256)))
    loaders = {
        split: make_loader(
            make_dataset(config, split, samples, split_seed),
            batch_size=batch_size,
            workers=workers,
            device=device,
        )
        for split, (samples, split_seed) in split_specs.items()
    }

    rows: list[dict[str, Any]] = []
    raw_done = False
    for index, checkpoint_path in enumerate(checkpoints):
        seed_everything(seed + 1_000 + index)
        model, metadata = load_model(model_config, checkpoint_path, device)
        include_controls = controls_enabled(args.controls_checkpoint, index, len(checkpoints))
        include_raw = args.include_raw_baselines and not raw_done
        splits = extract_splits(
            model,
            loaders,
            device,
            probe_config=probe_config,
            include_controls=include_controls,
            include_raw=include_raw,
        )
        label = checkpoint_path.stem
        append_variant_rows(
            rows,
            splits,
            feature_key="temporal_features",
            feature_variant="temporal_linear",
            model_kind="pretrained",
            checkpoint=label,
            metadata=metadata,
            budgets=budgets,
            split_seeds=split_seeds,
            num_directions=int(data["num_directions"]),
            probe_config=probe_config,
            strategy=strategy,
            run_label=run_label,
            display_name=display_name,
            training_seed=seed,
        )
        if include_controls:
            append_variant_rows(
                rows,
                splits,
                feature_key="last_token_features",
                feature_variant="last_token_linear",
                model_kind="pretrained_control",
                checkpoint=label,
                metadata=metadata,
                budgets=budgets,
                split_seeds=split_seeds,
                num_directions=int(data["num_directions"]),
                probe_config=probe_config,
                strategy=strategy,
                run_label=run_label,
                display_name=f"{display_name} · last token",
                training_seed=seed,
            )
            append_variant_rows(
                rows,
                splits,
                feature_key="shuffled_temporal_features",
                feature_variant="shuffled_input_temporal_linear",
                model_kind="pretrained_control",
                checkpoint=label,
                metadata=metadata,
                budgets=budgets,
                split_seeds=split_seeds,
                num_directions=int(data["num_directions"]),
                probe_config=probe_config,
                strategy=strategy,
                run_label=run_label,
                display_name=f"{display_name} · shuffled input",
                training_seed=seed,
            )
        if include_raw:
            raw_metadata = {"epoch": 0, "step": 0, "clips_seen": 0, "scoring_clips": 0}
            for feature_key, variant, name in [
                ("raw_temporal_features", "raw_temporal_linear", "Raw temporal pixels"),
                ("raw_last_frame_features", "raw_last_frame_linear", "Raw last frame"),
                ("raw_difference_features", "raw_difference_linear", "Raw frame differences"),
            ]:
                append_variant_rows(
                    rows,
                    splits,
                    feature_key=feature_key,
                    feature_variant=variant,
                    model_kind="raw_control",
                    checkpoint="raw_control",
                    metadata=raw_metadata,
                    budgets=budgets,
                    split_seeds=split_seeds,
                    num_directions=int(data["num_directions"]),
                    probe_config=probe_config,
                    strategy="raw_control",
                    run_label=variant,
                    display_name=name,
                    training_seed=seed,
                )
            raw_done = True
        print(
            f"{label}: low-shot temporal probe complete"
            + (" with controls" if include_controls else ""),
            flush=True,
        )

    if args.include_random_init:
        for random_seed in random_init_seeds:
            seed_everything(50_000 + random_seed)
            model, metadata = load_model(model_config, None, device)
            splits = extract_splits(
                model,
                loaders,
                device,
                probe_config=probe_config,
                include_controls=False,
                include_raw=False,
            )
            append_variant_rows(
                rows,
                splits,
                feature_key="temporal_features",
                feature_variant="random_temporal_linear",
                model_kind="random_encoder",
                checkpoint=f"random_init_{random_seed}",
                metadata=metadata,
                budgets=budgets,
                split_seeds=split_seeds,
                num_directions=int(data["num_directions"]),
                probe_config=probe_config,
                strategy="random_init",
                run_label="random_init",
                display_name="Random encoder",
                training_seed=random_seed,
            )

    output = Path(args.output) if args.output else run_dir / "lowshot_metrics.csv"
    write_csv_rows(output, rows)
    print(output)


if __name__ == "__main__":
    main()
