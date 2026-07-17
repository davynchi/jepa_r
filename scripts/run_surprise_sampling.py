from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import run_surprise_synthetic as base

from jepa.models import build_model_pair
from jepa.surprise import (
    GROUP_NAMES,
    learnable_sampling_score,
    positive_learning_progress,
    render_surprise_summary_svg,
    write_csv,
    zscore,
)


def sigmoid_scalar(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def row_learnable_sampling_score(
    row: dict[str, object],
    lower: float = 0.0,
    upper: float = 1.5,
    temperature: float = 0.5,
) -> float:
    lp = float(row["positive_learning_progress"])
    z = float(row["normalized_surprise"])
    not_easy_gate = sigmoid_scalar((z - lower) / temperature)
    not_extreme_gate = sigmoid_scalar((upper - z) / temperature)
    return lp * not_easy_gate * not_extreme_gate



def row_sampling_score(
    row: dict[str, object],
    strategy: str,
    epoch: int,
    warmup_epochs: int,
) -> float:
    if strategy == "uniform":
        return 1.0

    if strategy == "soft_surprise":
        return float(row["raw_mse_surprise"])

    if strategy == "warmup_soft_surprise":
        if epoch <= warmup_epochs:
            return 1.0
        return float(row["raw_mse_surprise"])

    if strategy in {"warmup_learnable_surprise", "capped_learnable_surprise"}:
        if epoch <= warmup_epochs:
            return 1.0
        return row_learnable_sampling_score(row)

    return 1.0


def attach_sampling_scores(
    sample_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    strategy: str,
    epoch: int,
    warmup_epochs: int,
) -> None:
    scores = []

    for row in sample_rows:
        score = row_sampling_score(row, strategy, epoch, warmup_epochs)
        row["sampling_score"] = score
        scores.append(score)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    if mean_score <= 1.0e-12:
        mean_score = 1.0

    by_group_score: dict[int, list[float]] = {0: [], 1: [], 2: []}
    by_group_relative: dict[int, list[float]] = {0: [], 1: [], 2: []}

    for row in sample_rows:
        group_id = int(row["group_id"])
        relative = float(row["sampling_score"]) / mean_score
        row["sampling_score_relative"] = relative
        by_group_score[group_id].append(float(row["sampling_score"]))
        by_group_relative[group_id].append(relative)

    for row in summary_rows:
        group_id = int(row["group_id"])
        scores_g = by_group_score[group_id]
        relative_g = by_group_relative[group_id]
        row["sampling_score_mean"] = sum(scores_g) / len(scores_g) if scores_g else 0.0
        row["sampling_score_relative_mean"] = sum(relative_g) / len(relative_g) if relative_g else 0.0



def keep_keys(row: dict[str, object], keys: list[str]) -> dict[str, object]:
    return {key: row[key] for key in keys if key in row}


def compact_rows(
    sample_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sample_keys = [
        "epoch",
        "step",
        "split",
        "strategy",
        "sample_id",
        "group_id",
        "group",
        "raw_mse_surprise",
        "positive_learning_progress",
        "sampling_score",
        "sampling_score_relative",
    ]

    summary_keys = [
        "epoch",
        "step",
        "split",
        "strategy",
        "group_id",
        "group",
        "raw_mse_surprise_mean",
        "positive_learning_progress_mean",
        "sampling_score_mean",
        "sampling_score_relative_mean",
    ]

    return (
        [keep_keys(row, sample_keys) for row in sample_rows],
        [keep_keys(row, summary_keys) for row in summary_rows],
    )


def robust_learnable_surprise_score(
    current_surprise: torch.Tensor,
    previous_surprise: torch.Tensor,
    lower: float = 0.0,
    upper: float = 1.5,
    temperature: float = 0.5,
    gamma: float = 1.5,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    lp_pos = positive_learning_progress(previous_surprise, current_surprise)
    normalized = zscore(current_surprise, eps=eps)
    lower_gate = torch.sigmoid((normalized - lower) / temperature)
    over = (normalized - upper).clamp_min(0.0)
    extreme_penalty = torch.exp(-gamma * over)
    q95 = torch.quantile(current_surprise, 0.95)
    quantile_penalty = torch.where(
        current_surprise <= q95,
        torch.ones_like(current_surprise),
        torch.full_like(current_surprise, 0.1),
    )
    return lp_pos * lower_gate * extreme_penalty * quantile_penalty


def soft_weights_from_score(
    score: torch.Tensor,
    strength: float,
    max_ratio: float,
    eps: float = 1.0e-8,
) -> torch.Tensor | None:
    if score.numel() == 0:
        return None

    score = score.float()
    if not torch.isfinite(score).all():
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    total = score.sum()
    if float(total.item()) <= eps:
        return None

    scaled = score / (score.mean() + eps)
    weights = 1.0 + strength * scaled
    weights = weights.clamp(min=1.0 / max_ratio, max=max_ratio)
    return weights.double()


def apply_group_cap(
    weights: torch.Tensor,
    groups: torch.Tensor,
    group_id: int,
    cap: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if cap <= 0.0 or cap >= 1.0:
        return weights

    mask = groups == group_id
    group_mass = weights[mask].sum()
    other_mass = weights[~mask].sum()

    total = group_mass + other_mass
    if float(total.item()) <= eps:
        return weights

    current_fraction = group_mass / total
    if float(current_fraction.item()) <= cap:
        return weights

    scale = (cap * other_mass) / ((1.0 - cap) * group_mass + eps)
    weights = weights.clone()
    weights[mask] = weights[mask] * scale
    return weights


def build_sampling_weights(
    strategy: str,
    current_smoothed: torch.Tensor | None,
    reference: torch.Tensor | None,
    groups: torch.Tensor,
    epoch: int,
    warmup_epochs: int,
    strength: float,
    max_ratio: float,
    hard_cap: float,
    noise_cap: float,
) -> torch.Tensor | None:
    if strategy == "uniform":
        return None

    if current_smoothed is None:
        return None

    warmup = strategy in {"warmup_soft_surprise", "warmup_learnable_surprise", "capped_learnable_surprise"}
    if warmup and epoch <= warmup_epochs:
        return None

    if strategy in {"soft_surprise", "warmup_soft_surprise"}:
        score = current_smoothed.clamp_min(0.0)
        return soft_weights_from_score(score, strength=strength, max_ratio=max_ratio)

    if strategy in {"warmup_learnable_surprise", "capped_learnable_surprise"}:
        if reference is None:
            return None
        score = learnable_sampling_score(current_smoothed, reference).clamp_min(0.0)
        weights = soft_weights_from_score(score, strength=strength, max_ratio=max_ratio)
        if weights is None:
            return None

        if strategy == "capped_learnable_surprise":
            weights = apply_group_cap(weights, groups, group_id=1, cap=hard_cap)
            weights = apply_group_cap(weights, groups, group_id=2, cap=noise_cap)

        return weights

    raise ValueError(f"unknown sampling strategy: {strategy}")


def make_train_loader(
    dataset: base.SurpriseTimeSeriesDataset,
    batch_size: int,
    weights: torch.Tensor | None,
) -> DataLoader:
    if weights is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def subset_by_group_counts(
    dataset: base.SurpriseTimeSeriesDataset,
    counts: dict[int, int],
    seed: int,
) -> base.SurpriseTimeSeriesDataset:
    generator = torch.Generator().manual_seed(seed)
    selected = []

    for group_id, count in counts.items():
        group_indices = torch.where(dataset.groups == group_id)[0]
        order = torch.randperm(group_indices.numel(), generator=generator)
        chosen = group_indices[order[:count]]
        selected.append(chosen)

    indices = torch.cat(selected)
    order = torch.randperm(indices.numel(), generator=generator)
    indices = indices[order]

    return base.SurpriseTimeSeriesDataset(
        dataset.contexts[indices],
        dataset.targets[indices],
        dataset.groups[indices],
        dataset.split,
    )


def build_datasets(
    config: base.SurpriseSyntheticConfig,
    test_samples_per_group: int,
    train_counts: dict[int, int],
):
    max_train_count = max(train_counts.values())

    train_full, mean, std = base._make_split(
        samples_per_group=max_train_count,
        split="train",
        config=config,
    )
    train = subset_by_group_counts(
        train_full,
        counts=train_counts,
        seed=config.seed + 12345,
    )
    validation, _, _ = base._make_split(
        samples_per_group=config.validation_samples_per_group,
        split="validation",
        config=config,
        mean=mean,
        std=std,
    )
    test, _, _ = base._make_split(
        samples_per_group=test_samples_per_group,
        split="test",
        config=config,
        mean=mean,
        std=std,
    )
    return train, validation, test


def train_one_epoch(
    *,
    loader: DataLoader,
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stop_gradient: bool,
    ema_decay: float,
    target_update: str,
) -> tuple[float, dict[int, int]]:
    context_encoder.train()
    predictor.train()
    target_encoder.eval()

    total_loss = 0.0
    total_count = 0
    group_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    for context, target, _, group_ids in loader:
        for group_id in group_ids.tolist():
            group_counts[int(group_id)] = group_counts.get(int(group_id), 0) + 1

        context = context.to(device).flatten(1)
        target = target.to(device).flatten(1)

        optimizer.zero_grad(set_to_none=True)

        context_latent = context_encoder(context)
        target_latent = target_encoder(target)
        if stop_gradient:
            target_latent = target_latent.detach()

        prediction = predictor(context_latent)
        per_sample = base.per_sample_mse(prediction, target_latent)
        loss = per_sample.mean()

        loss.backward()
        optimizer.step()
        if target_update == "ema":
            base.update_ema(target_encoder, context_encoder, ema_decay)

        total_loss += float(loss.detach().cpu().item()) * context.shape[0]
        total_count += context.shape[0]

    return total_loss / max(total_count, 1), group_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/surprise_sampling")
    parser.add_argument("--architecture", choices=("linear", "nonlinear"), default="nonlinear")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--target-update", choices=("ema", "frozen"), default="ema")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context-steps", type=int, default=128)
    parser.add_argument("--train-samples-per-group", type=int, default=1024)
    parser.add_argument("--train-easy-samples", type=int, default=2048)
    parser.add_argument("--train-hard-samples", type=int, default=768)
    parser.add_argument("--train-noise-samples", type=int, default=256)
    parser.add_argument("--validation-samples-per-group", type=int, default=512)
    parser.add_argument("--test-samples-per-group", type=int, default=512)
    parser.add_argument("--lp-delta", type=int, default=5)
    parser.add_argument("--lp-ema-beta", type=float, default=0.9)
    parser.add_argument(
        "--sampling-strategy",
        choices=(
            "uniform",
            "soft_surprise",
            "warmup_soft_surprise",
            "warmup_learnable_surprise",
            "capped_learnable_surprise",
        ),
        default="uniform",
    )
    parser.add_argument("--sampling-strength", type=float, default=0.35)
    parser.add_argument("--sampling-max-ratio", type=float, default=3.0)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--hard-cap", type=float, default=0.45)
    parser.add_argument("--noise-cap", type=float, default=0.15)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    if args.sequence_length != 2 * args.context_steps:
        raise ValueError("sequence_length must be equal to 2 * context_steps")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_config = base.SurpriseSyntheticConfig(
        train_samples_per_group=args.train_samples_per_group,
        validation_samples_per_group=args.validation_samples_per_group,
        sequence_length=args.sequence_length,
        context_steps=args.context_steps,
        seed=args.seed,
    )

    train_counts = {
        0: args.train_easy_samples,
        1: args.train_hard_samples,
        2: args.train_noise_samples,
    }

    train_dataset, validation_dataset, test_dataset = build_datasets(
        data_config,
        args.test_samples_per_group,
        train_counts=train_counts,
    )

    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = base.resolve_device(args.device)

    input_dim = data_config.context_steps * data_config.observation_dim
    context_encoder, predictor = build_model_pair(
        args.architecture,
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
    )

    target_encoder = deepcopy(context_encoder)
    for parameter in target_encoder.parameters():
        parameter.requires_grad_(False)

    context_encoder.to(device)
    target_encoder.to(device)
    predictor.to(device)

    optimizer = torch.optim.Adam(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=args.learning_rate,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_root) / f"{timestamp}_{args.sampling_strategy}"
    run_dir.mkdir(parents=True, exist_ok=False)

    all_sample_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    all_sampling_rows: list[dict[str, object]] = []

    smoothed_by_split: dict[str, torch.Tensor | None] = {
        "train": None,
        "validation": None,
        "test": None,
    }
    smoothed_history_by_split: dict[str, list[torch.Tensor]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    started = time.time()
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)

    for epoch in range(1, args.epochs + 1):
        sampling_reference = (
            smoothed_history_by_split["train"][-args.lp_delta]
            if len(smoothed_history_by_split["train"]) >= args.lp_delta
            else None
        )

        sampling_weights = build_sampling_weights(
            strategy=args.sampling_strategy,
            current_smoothed=smoothed_by_split["train"],
            reference=sampling_reference,
            groups=train_dataset.groups,
            epoch=epoch,
            warmup_epochs=args.warmup_epochs,
            strength=args.sampling_strength,
            max_ratio=args.sampling_max_ratio,
            hard_cap=args.hard_cap,
            noise_cap=args.noise_cap,
        )

        train_loader = make_train_loader(
            train_dataset,
            batch_size=args.batch_size,
            weights=sampling_weights,
        )

        train_loss, sampling_group_counts = train_one_epoch(
            loader=train_loader,
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            optimizer=optimizer,
            device=device,
            stop_gradient=True,
            ema_decay=args.ema_decay,
            target_update=args.target_update,
        )

        selected_total = sum(sampling_group_counts.values())
        for group_id, group_name in GROUP_NAMES.items():
            count = int(sampling_group_counts.get(group_id, 0))
            all_sampling_rows.append(
                {
                    "epoch": epoch,
                    "step": epoch * steps_per_epoch,
                    "strategy": args.sampling_strategy,
                    "group_id": group_id,
                    "group": group_name,
                    "selected_count": count,
                    "selected_fraction": count / max(selected_total, 1),
                }
            )

        split_specs = [
            ("train", train_dataset, train_eval_loader),
            ("validation", validation_dataset, validation_loader),
            ("test", test_dataset, test_loader),
        ]

        for split, dataset, loader in split_specs:
            reference = (
                smoothed_history_by_split[split][-args.lp_delta]
                if len(smoothed_history_by_split[split]) >= args.lp_delta
                else None
            )

            sample_rows, summary_rows, _, smoothed = base.evaluate_split(
                epoch=epoch,
                split=split,
                dataset=dataset,
                loader=loader,
                context_encoder=context_encoder,
                target_encoder=target_encoder,
                predictor=predictor,
                device=device,
                lp_reference=reference,
                previous_smoothed=smoothed_by_split[split],
                lp_ema_beta=args.lp_ema_beta,
            )

            for row in sample_rows:
                row["step"] = epoch * steps_per_epoch
                row["strategy"] = args.sampling_strategy
            for row in summary_rows:
                row["step"] = epoch * steps_per_epoch
                row["strategy"] = args.sampling_strategy

            attach_sampling_scores(
                sample_rows,
                summary_rows,
                strategy=args.sampling_strategy,
                epoch=epoch,
                warmup_epochs=args.warmup_epochs,
            )

            sample_rows, summary_rows = compact_rows(sample_rows, summary_rows)

            all_sample_rows.extend(sample_rows)
            all_summary_rows.extend(summary_rows)

            smoothed_by_split[split] = smoothed
            smoothed_history_by_split[split].append(smoothed)

        selected_easy = sampling_group_counts.get(0, 0) / max(selected_total, 1)
        selected_hard = sampling_group_counts.get(1, 0) / max(selected_total, 1)
        selected_noise = sampling_group_counts.get(2, 0) / max(selected_total, 1)

        val_rows = [
            row for row in all_summary_rows
            if row["epoch"] == epoch and row["split"] == "validation"
        ]
        val_raw_mean = sum(float(row["raw_mse_surprise_mean"]) for row in val_rows) / max(len(val_rows), 1)

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"step={epoch * steps_per_epoch:04d} | "
            f"strategy={args.sampling_strategy} | "
            f"train_loss={train_loss:.6g} | "
            f"val_group_raw_mean={val_raw_mean:.6g} | "
            f"selected=({selected_easy:.2f},{selected_hard:.2f},{selected_noise:.2f})",
            flush=True,
        )

    write_csv(run_dir / "surprise_history.csv", all_sample_rows)
    write_csv(run_dir / "surprise_summary.csv", all_summary_rows)
    write_csv(run_dir / "sampling_history.csv", all_sampling_rows)

    for split in ("train", "validation", "test"):
        (run_dir / f"surprise_{split}.svg").write_text(
            render_surprise_summary_svg(
                all_summary_rows,
                split=split,
                title=f"Surprise sampling — {args.sampling_strategy} — {split}",
            )
        )

    status = {
        "state": "complete",
        "run_dir": str(run_dir),
        "elapsed_seconds": time.time() - started,
        "config": {
            "data": asdict(data_config),
            "test_samples_per_group": args.test_samples_per_group,
            "architecture": args.architecture,
            "epochs": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "learning_rate": args.learning_rate,
            "ema_decay": args.ema_decay,
            "lp_delta": args.lp_delta,
            "lp_ema_beta": args.lp_ema_beta,
            "seed": args.seed,
            "sampling_strategy": args.sampling_strategy,
            "sampling_strength": args.sampling_strength,
            "sampling_max_ratio": args.sampling_max_ratio,
            "warmup_epochs": args.warmup_epochs,
            "hard_cap": args.hard_cap,
            "noise_cap": args.noise_cap,
            "train_counts": train_counts,
            "device": str(device),
            "stop_gradient": True,
            "target_update": args.target_update,
        },
        "artifacts": [
            "surprise_history.csv",
            "surprise_summary.csv",
            "sampling_history.csv",
            "surprise_train.svg",
            "surprise_validation.svg",
            "surprise_test.svg",
        ],
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))

    print(json.dumps({"state": "complete", "run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
