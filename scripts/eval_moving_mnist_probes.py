#!/usr/bin/env python3
"""Evaluate frozen Moving-MNIST representations and select one checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from jepa.moving_mnist import MovingMNISTDataset, MovingMNISTSpec, make_fake_digit_bank
from jepa.video_diagnostics import (
    accumulate_pairing_losses,
    finalize_pairing_metrics,
    pairing_context_batch,
)
from jepa.video_jepa import CausalVideoJEPA, CausalVideoJEPAConfig
from jepa.video_probes import (
    ProbeConfig,
    evaluate_attentive_classification_probes,
    evaluate_probe_suite,
    extract_frozen_features,
)
from jepa.video_selection import (
    CheckpointCandidate,
    SelectionCriterion,
    criteria_from_config,
    select_checkpoint,
)
from jepa.video_utils import (
    atomic_json_dump,
    load_yaml,
    read_csv_rows,
    resolve_device,
    seed_everything,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoints", choices=["all", "latest"], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--test-every-checkpoint", action="store_true")
    parser.add_argument("--test-final-too", action="store_true")
    parser.add_argument("--include-random-init", action="store_true")
    parser.add_argument(
        "--attentive-tasks",
        help="comma-separated subset of direction,bounce,digit; overrides config",
    )
    parser.add_argument("--skip-pairing-controls", action="store_true")
    parser.add_argument("--selection-metric")
    parser.add_argument("--selection-mode", choices=["min", "max"])
    return parser.parse_args()


def make_model_config(config: dict[str, Any]) -> CausalVideoJEPAConfig:
    data = config["data"]
    model = config["model"]
    objective = config.get("objective", {})
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
        position_embedding_type=str(
            model.get("position_embedding_type", "separate_3d")
        ),
        target_weighting_mode=str(
            objective.get(
                "target_weighting_mode",
                "foreground" if float(objective.get("foreground_weight", 1.0)) > 1.0 else "uniform",
            )
        ),
        foreground_weight=float(objective.get("foreground_weight", 1.0)),
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


def discover_checkpoints(run_dir: Path, mode: str) -> list[Path]:
    paths = sorted(
        run_dir.joinpath("checkpoints").glob("epoch_*.pt"),
        key=checkpoint_epoch,
    )
    if not paths:
        latest = run_dir / "checkpoints" / "latest.pt"
        if latest.exists():
            return [latest]
        raise FileNotFoundError(f"no checkpoints found in {run_dir}")
    return [paths[-1]] if mode == "latest" else paths


def rows_for_metrics(
    metrics: dict[str, float],
    *,
    checkpoint: str,
    checkpoint_role: str,
    epoch: int,
    step: int,
    clips_seen: int,
    scoring_clips: int,
    strategy: str,
    run_label: str,
    display_name: str,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": checkpoint,
            "checkpoint_role": checkpoint_role,
            "epoch": epoch,
            "step": step,
            "clips_seen": clips_seen,
            "scoring_clips": scoring_clips,
            "strategy": strategy,
            "run_label": run_label,
            "display_name": display_name,
            "seed": seed,
            "split": split,
            "metric": metric,
            "value": value,
        }
        for metric, value in metrics.items()
    ]


@torch.no_grad()
def evaluate_pairing_controls(
    model: CausalVideoJEPA,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    max_samples: int,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    count = 0
    for batch in loader:
        if count >= max_samples:
            break
        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        take = min(len(context), max_samples - count)
        losses = pairing_context_batch(
            model,
            context[:take],
            target[:take],
            digit_labels=batch["digit_label"][:take],
            direction_labels=batch["direction_label"][:take],
        )
        accumulate_pairing_losses(sums, counts, losses)
        count += take
    return finalize_pairing_metrics(sums, counts)


def pretrain_metrics_by_epoch(run_dir: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for row in read_csv_rows(run_dir / "pretrain_metrics.csv"):
        if row.get("split") != "validation":
            continue
        epoch = int(float(row["epoch"]))
        result.setdefault(epoch, {})[str(row["metric"])] = float(row["value"])
    return result


def make_probe_config(config: dict[str, Any], seed: int) -> ProbeConfig:
    section = config.get("probe", {})
    return ProbeConfig(
        classifier_epochs=int(section.get("classifier_epochs", 80)),
        classifier_batch_size=int(section.get("classifier_batch_size", 256)),
        classifier_lr=float(section.get("classifier_lr", 1e-2)),
        classifier_weight_decay=float(section.get("classifier_weight_decay", 1e-4)),
        ridge_lambdas=tuple(
            float(value)
            for value in section.get(
                "ridge_lambdas", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
            )
        ),
        attentive_epochs=int(section.get("attentive_epochs", 30)),
        attentive_batch_size=int(section.get("attentive_batch_size", 128)),
        attentive_lr=float(section.get("attentive_lr", 1e-3)),
        attentive_weight_decay=float(section.get("attentive_weight_decay", 1e-4)),
        attentive_heads=int(section.get("attentive_heads", 4)),
        attentive_mlp_ratio=float(section.get("attentive_mlp_ratio", 2.0)),
        attentive_spatial_pool=int(section.get("attentive_spatial_pool", 4)),
        raw_spatial_pool=int(section.get("raw_spatial_pool", 8)),
        seed=seed + 701,
    )


def load_model(
    model_config: CausalVideoJEPAConfig,
    checkpoint_path: Path | None,
    device: torch.device,
) -> tuple[CausalVideoJEPA, dict[str, int]]:
    model = CausalVideoJEPA(model_config).to(device)
    if checkpoint_path is None:
        return model, {
            "epoch": 0,
            "step": 0,
            "clips_seen": 0,
            "scoring_clips": 0,
        }
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model, {
        "epoch": int(checkpoint["epoch"]),
        "step": int(checkpoint["step"]),
        "clips_seen": int(checkpoint["clips_seen"]),
        "scoring_clips": int(checkpoint.get("scoring_clips", 0)),
    }


def evaluate_features(
    model: CausalVideoJEPA,
    loaders: dict[str, DataLoader[dict[str, Tensor]]],
    device: torch.device,
    *,
    probe_config: ProbeConfig,
    attentive_tasks: tuple[str, ...],
    num_directions: int,
    evaluation_split: str,
) -> tuple[dict[str, float], dict[str, float]]:
    extraction_kwargs = {
        "include_attentive_tokens": bool(attentive_tasks),
        "attentive_spatial_pool": probe_config.attentive_spatial_pool,
    }
    train_features = extract_frozen_features(
        model, loaders["probe_train"], device, **extraction_kwargs
    )
    validation_features = extract_frozen_features(
        model, loaders["probe_validation"], device, **extraction_kwargs
    )
    evaluation_features = extract_frozen_features(
        model, loaders[evaluation_split], device, **extraction_kwargs
    )
    metrics, hyperparameters = evaluate_probe_suite(
        train_features,
        validation_features,
        evaluation_features,
        num_directions=num_directions,
        config=probe_config,
    )
    if attentive_tasks:
        attentive_metrics, attentive_hyperparameters = (
            evaluate_attentive_classification_probes(
                train_features,
                validation_features,
                evaluation_features,
                num_directions=num_directions,
                tasks=attentive_tasks,
                config=probe_config,
                device=device,
            )
        )
        metrics.update(attentive_metrics)
        hyperparameters.update(attentive_hyperparameters)
    return metrics, hyperparameters


def append_hyperparameters(
    rows: list[dict[str, Any]],
    values: dict[str, float],
    *,
    checkpoint: str,
    checkpoint_role: str,
    epoch: int,
    strategy: str,
    run_label: str,
    display_name: str,
    seed: int,
    split: str,
) -> None:
    for metric, value in values.items():
        rows.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_role": checkpoint_role,
                "epoch": epoch,
                "strategy": strategy,
                "run_label": run_label,
                "display_name": display_name,
                "seed": seed,
                "split": split,
                "metric": metric,
                "value": value,
            }
        )


def selection_criteria(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[SelectionCriterion, ...]:
    criteria = list(criteria_from_config(config))
    if args.selection_metric is not None:
        mode = args.selection_mode or "max"
        criteria[0] = SelectionCriterion(
            metric=args.selection_metric,
            mode=mode,
            source="probe",
        )
    elif args.selection_mode is not None:
        criteria[0] = SelectionCriterion(
            metric=criteria[0].metric,
            mode=args.selection_mode,
            source=criteria[0].source,
        )
    return tuple(criteria)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    experiment = config["experiment"]
    data = config["data"]
    probe_section = config.get("probe", {})
    seed = int(experiment.get("seed", 0))
    strategy = str(config["sampling"]["strategy"])
    run_label = str(experiment.get("run_label", strategy))
    default_display = strategy.replace("_", " ").title()
    display_name = str(experiment.get("display_name", default_display))
    device = resolve_device(args.device)
    seed_everything(seed)

    attentive_tasks = tuple(
        task.strip()
        for task in (
            args.attentive_tasks.split(",")
            if args.attentive_tasks is not None
            else probe_section.get("attentive_tasks", ["direction"])
        )
        if task.strip()
    )
    probe_config = make_probe_config(config, seed)
    batch_size = int(probe_section.get("batch_size", 256))
    workers = int(
        probe_section.get("num_workers", config["training"].get("num_workers", 0))
    )

    split_specs = {
        "probe_train": (int(probe_section.get("train_samples", 8_000)), seed + 307),
        "probe_validation": (
            int(probe_section.get("validation_samples", 2_000)),
            seed + 401,
        ),
        "probe_evaluation": (
            int(probe_section.get("evaluation_samples", 2_000)),
            seed + 503,
        ),
        "test": (int(probe_section.get("test_samples", 5_000)), seed + 601),
    }
    datasets = {
        split: make_dataset(config, split, samples, split_seed)
        for split, (samples, split_seed) in split_specs.items()
    }
    loaders = {
        split: make_loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            device=device,
        )
        for split, dataset in datasets.items()
    }

    checkpoints = discover_checkpoints(run_dir, args.checkpoints)
    if "position_embedding_type" not in config.get("model", {}):
        metadata = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
        schema_version = int(metadata.get("schema_version", 1))
        config["model"]["position_embedding_type"] = (
            "legacy_sum" if schema_version < 2 else "separate_3d"
        )
        print(
            "position embedding inferred as "
            f"{config['model']['position_embedding_type']} from schema "
            f"{schema_version}",
            flush=True,
        )

    model_config = make_model_config(config)
    pretrain_lookup = pretrain_metrics_by_epoch(run_dir)
    metric_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    candidates: list[CheckpointCandidate] = []
    validation_results: dict[str, dict[str, float]] = {}

    evaluations: list[tuple[str, Path | None]] = []
    if args.include_random_init:
        evaluations.append(("random_init", None))
    evaluations.extend((path.stem, path) for path in checkpoints)

    for evaluation_index, (label, checkpoint_path) in enumerate(evaluations):
        seed_everything(seed + 900 + evaluation_index)
        model, metadata = load_model(model_config, checkpoint_path, device)
        role = "reference" if checkpoint_path is None else "candidate"
        evaluation_strategy = "random_init" if checkpoint_path is None else strategy
        evaluation_run_label = "random_init" if checkpoint_path is None else run_label
        evaluation_display_name = (
            "Random encoder" if checkpoint_path is None else display_name
        )

        metrics, hyperparameters = evaluate_features(
            model,
            loaders,
            device,
            probe_config=probe_config,
            attentive_tasks=attentive_tasks,
            num_directions=int(data["num_directions"]),
            evaluation_split="probe_evaluation",
        )
        validation_results[label] = metrics
        metric_rows.extend(
            rows_for_metrics(
                metrics,
                checkpoint=label,
                checkpoint_role=role,
                epoch=metadata["epoch"],
                step=metadata["step"],
                clips_seen=metadata["clips_seen"],
                scoring_clips=metadata["scoring_clips"],
                strategy=evaluation_strategy,
                run_label=evaluation_run_label,
                display_name=evaluation_display_name,
                seed=seed,
                split="validation",
            )
        )
        append_hyperparameters(
            hyperparameter_rows,
            hyperparameters,
            checkpoint=label,
            checkpoint_role=role,
            epoch=metadata["epoch"],
            strategy=evaluation_strategy,
            run_label=evaluation_run_label,
            display_name=evaluation_display_name,
            seed=seed,
            split="validation",
        )

        if not args.skip_pairing_controls:
            pairing_metrics = evaluate_pairing_controls(
                model,
                loaders["probe_evaluation"],
                device,
                max_samples=int(probe_section.get("pairing_samples", 512)),
            )
            pairing_rows.extend(
                rows_for_metrics(
                    pairing_metrics,
                    checkpoint=label,
                    checkpoint_role=role,
                    epoch=metadata["epoch"],
                    step=metadata["step"],
                    clips_seen=metadata["clips_seen"],
                    scoring_clips=metadata["scoring_clips"],
                    strategy=evaluation_strategy,
                    run_label=evaluation_run_label,
                    display_name=evaluation_display_name,
                    seed=seed,
                    split="pairing",
                )
            )

        if checkpoint_path is not None:
            candidates.append(
                CheckpointCandidate(
                    label=label,
                    path=str(checkpoint_path),
                    epoch=metadata["epoch"],
                    step=metadata["step"],
                    clips_seen=metadata["clips_seen"],
                    scoring_clips=metadata["scoring_clips"],
                    probe_metrics=metrics,
                    pretrain_metrics=pretrain_lookup.get(metadata["epoch"], {}),
                )
            )

        if args.test_every_checkpoint and checkpoint_path is not None:
            test_metrics, test_hyperparameters = evaluate_features(
                model,
                loaders,
                device,
                probe_config=probe_config,
                attentive_tasks=attentive_tasks,
                num_directions=int(data["num_directions"]),
                evaluation_split="test",
            )
            metric_rows.extend(
                rows_for_metrics(
                    test_metrics,
                    checkpoint=label,
                    checkpoint_role="debug_test",
                    epoch=metadata["epoch"],
                    step=metadata["step"],
                    clips_seen=metadata["clips_seen"],
                    scoring_clips=metadata["scoring_clips"],
                    strategy=strategy,
                    run_label=run_label,
                    display_name=display_name,
                    seed=seed,
                    split="test_debug",
                )
            )
            append_hyperparameters(
                hyperparameter_rows,
                test_hyperparameters,
                checkpoint=label,
                checkpoint_role="debug_test",
                epoch=metadata["epoch"],
                strategy=strategy,
                run_label=run_label,
                display_name=display_name,
                seed=seed,
                split="test_debug",
            )

        write_csv_rows(run_dir / "probe_metrics.csv", metric_rows)
        write_csv_rows(run_dir / "probe_hyperparameters.csv", hyperparameter_rows)
        if pairing_rows:
            write_csv_rows(run_dir / "pairing_metrics.csv", pairing_rows)
        print(
            f"{label}: direction(att.)="
            f"{metrics.get('attentive_direction_accuracy', float('nan')):.4f} "
            f"speed={metrics['speed_rmse']:.4f} FDE={metrics['future_fde']:.3f}",
            flush=True,
        )

    criteria = selection_criteria(config, args)
    selected = select_checkpoint(candidates, criteria)
    atomic_json_dump(
        run_dir / "checkpoint_selection.json",
        {
            "run_label": run_label,
            "display_name": display_name,
            "selected_checkpoint": selected.label,
            "selected_path": selected.path,
            "epoch": selected.epoch,
            "step": selected.step,
            "clips_seen": selected.clips_seen,
            "scoring_clips": selected.scoring_clips,
            "criteria": [
                {
                    "metric": criterion.metric,
                    "mode": criterion.mode,
                    "source": criterion.source,
                }
                for criterion in criteria
            ],
            "selected_values": {
                f"{criterion.source}:{criterion.metric}": selected.value(criterion)
                for criterion in criteria
            },
            "candidates": [
                {
                    "checkpoint": candidate.label,
                    "epoch": candidate.epoch,
                    "clips_seen": candidate.clips_seen,
                    "values": {
                        f"{criterion.source}:{criterion.metric}": candidate.value(
                            criterion
                        )
                        for criterion in criteria
                    },
                }
                for candidate in candidates
            ],
        },
    )

    test_targets: list[tuple[CheckpointCandidate, str]] = [(selected, "selected")]
    final_candidate = max(candidates, key=lambda candidate: candidate.epoch)
    if args.test_final_too and final_candidate.label != selected.label:
        test_targets.append((final_candidate, "final"))

    for target_index, (candidate, role) in enumerate(test_targets):
        seed_everything(seed + 1_900 + target_index)
        model, metadata = load_model(model_config, Path(candidate.path), device)
        test_metrics, test_hyperparameters = evaluate_features(
            model,
            loaders,
            device,
            probe_config=probe_config,
            attentive_tasks=attentive_tasks,
            num_directions=int(data["num_directions"]),
            evaluation_split="test",
        )
        split = "test" if role == "selected" else "test_final"
        metric_rows.extend(
            rows_for_metrics(
                test_metrics,
                checkpoint=candidate.label,
                checkpoint_role=role,
                epoch=metadata["epoch"],
                step=metadata["step"],
                clips_seen=metadata["clips_seen"],
                scoring_clips=metadata["scoring_clips"],
                strategy=strategy,
                run_label=run_label,
                display_name=display_name,
                seed=seed,
                split=split,
            )
        )
        append_hyperparameters(
            hyperparameter_rows,
            test_hyperparameters,
            checkpoint=candidate.label,
            checkpoint_role=role,
            epoch=metadata["epoch"],
            strategy=strategy,
            run_label=run_label,
            display_name=display_name,
            seed=seed,
            split=split,
        )

    write_csv_rows(run_dir / "probe_metrics.csv", metric_rows)
    write_csv_rows(run_dir / "probe_hyperparameters.csv", hyperparameter_rows)
    if pairing_rows:
        write_csv_rows(run_dir / "pairing_metrics.csv", pairing_rows)
    print(
        f"selected {selected.label} at epoch {selected.epoch} using "
        + " -> ".join(
            f"{criterion.source}:{criterion.metric}({criterion.mode})"
            for criterion in criteria
        ),
        flush=True,
    )
    print(run_dir / "probe_metrics.csv")


if __name__ == "__main__":
    main()
