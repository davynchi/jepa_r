from __future__ import annotations

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jepa.models import build_model_pair
from jepa.surprise import (
    GROUP_NAMES,
    bandpass_useful_surprise_score,
    per_sample_cosine_distance,
    per_sample_mse,
    per_sample_norm_gap,
    per_sample_relative_mse,
    render_surprise_summary_svg,
    useful_surprise_score,
    write_csv,
    zscore,
)


GroupName = Literal["easy", "hard", "noise"]


@dataclass(frozen=True, slots=True)
class SurpriseSyntheticConfig:
    train_samples_per_group: int = 1024
    validation_samples_per_group: int = 512
    sequence_length: int = 64
    context_steps: int = 32
    observation_dim: int = 1
    seed: int = 0


class SurpriseTimeSeriesDataset(Dataset):
    def __init__(
        self,
        contexts: torch.Tensor,
        targets: torch.Tensor,
        groups: torch.Tensor,
        split: str,
    ) -> None:
        if contexts.ndim != 3 or targets.ndim != 3:
            raise ValueError("contexts and targets must have shape [N, T, D]")
        if contexts.shape[0] != targets.shape[0] or contexts.shape[0] != groups.shape[0]:
            raise ValueError("contexts, targets, groups must have same N")
        self.contexts = contexts.float().contiguous()
        self.targets = targets.float().contiguous()
        self.groups = groups.long().contiguous()
        self.split = split

    def __len__(self) -> int:
        return self.contexts.shape[0]

    def __getitem__(self, index: int):
        return (
            self.contexts[index],
            self.targets[index],
            torch.tensor(index, dtype=torch.long),
            self.groups[index],
        )


def _easy_series(rng: np.random.Generator, steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    amp = rng.uniform(0.8, 1.2)
    freq = rng.uniform(1.0, 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    trend = rng.uniform(-0.15, 0.15) * t
    y = amp * np.sin(2.0 * np.pi * freq * t + phase) + trend
    y += rng.normal(0.0, 0.03, size=steps)
    return y.astype(np.float32)


def _hard_series(rng: np.random.Generator, steps: int, context_steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)

    amp = rng.uniform(0.7, 1.3)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    slope = rng.uniform(-0.9, 0.9)

    y = amp * np.sin(2.0 * np.pi * 1.0 * t + phase)
    y += 0.35 * np.sin(2.0 * np.pi * 2.7 * t + 0.5 * phase)
    y += slope * t

    context = y[:context_steps]
    context_mean = float(context.mean())
    context_std = float(context.std())
    context_last = float(context[-1])

    target_mask = t >= t[context_steps]
    target_t = t[target_mask]

    regime_freq = 3.0 + 2.0 * (context_mean > 0.0) + 1.5 * (slope > 0.0)
    regime_amp = 0.45 + 0.35 * (context_std > 0.55)
    regime_phase = phase + (np.pi / 3.0) * (context_last > 0.0)

    y[target_mask] += regime_amp * np.sin(
        2.0 * np.pi * regime_freq * target_t + regime_phase
    )

    centers = [0.62, 0.76, 0.90]
    signs = [
        1.0 if context_mean >= 0.0 else -1.0,
        1.0 if context_last >= 0.0 else -1.0,
        1.0 if slope >= 0.0 else -1.0,
    ]
    widths = [0.020, 0.026, 0.032]

    for center, sign, width in zip(centers, signs, widths):
        y += sign * 0.75 * np.exp(-0.5 * ((t - center) / width) ** 2)

    target_indices = np.arange(context_steps, steps)
    knot_count = 6
    knot_positions = np.linspace(context_steps, steps - 1, knot_count).astype(int)
    knot_values = rng.normal(0.0, 0.18, size=knot_count).astype(np.float32)
    smooth_component = np.interp(target_indices, knot_positions, knot_values)
    y[context_steps:] += smooth_component.astype(np.float32)

    y += rng.normal(0.0, 0.035, size=steps)
    return y.astype(np.float32)


def _noise_series(rng: np.random.Generator, steps: int, context_steps: int) -> np.ndarray:
    context = _easy_series(rng, context_steps)
    target = rng.normal(0.0, 1.0, size=steps - context_steps).astype(np.float32)
    return np.concatenate([context, target]).astype(np.float32)


def _make_split(
    *,
    samples_per_group: int,
    split: str,
    config: SurpriseSyntheticConfig,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[SurpriseTimeSeriesDataset, float, float]:
    seed_offset = {"train": 0, "validation": 10_000, "test": 20_000}[split]
    rng = np.random.default_rng(config.seed + seed_offset)

    sequences: list[np.ndarray] = []
    groups: list[int] = []

    generators = [
        (0, lambda: _easy_series(rng, config.sequence_length)),
        (1, lambda: _hard_series(rng, config.sequence_length, config.context_steps)),
        (2, lambda: _noise_series(rng, config.sequence_length, config.context_steps)),
    ]

    for group_id, generator in generators:
        for _ in range(samples_per_group):
            sequences.append(generator())
            groups.append(group_id)

    sequences_array = np.stack(sequences).astype(np.float32)

    order = rng.permutation(len(sequences_array))
    sequences_array = sequences_array[order]
    groups_array = np.array(groups, dtype=np.int64)[order]

    if mean is None or std is None:
        stat_array = sequences_array[:, : config.context_steps]
        mean = float(stat_array.mean())
        std = float(stat_array.std() + 1.0e-6)

    sequences_array = (sequences_array - mean) / std

    sequences_tensor = torch.from_numpy(sequences_array).unsqueeze(-1)
    groups_tensor = torch.from_numpy(groups_array)

    contexts = sequences_tensor[:, : config.context_steps]
    targets = sequences_tensor[:, config.context_steps :]

    return SurpriseTimeSeriesDataset(contexts, targets, groups_tensor, split), mean, std


def build_surprise_datasets(
    config: SurpriseSyntheticConfig,
) -> tuple[SurpriseTimeSeriesDataset, SurpriseTimeSeriesDataset]:
    train, mean, std = _make_split(
        samples_per_group=config.train_samples_per_group,
        split="train",
        config=config,
    )
    validation, _, _ = _make_split(
        samples_per_group=config.validation_samples_per_group,
        split="validation",
        config=config,
        mean=mean,
        std=std,
    )
    return train, validation


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(decay).add_(source_param, alpha=1.0 - decay)


def evaluate_split(
    *,
    epoch: int,
    split: str,
    dataset: SurpriseTimeSeriesDataset,
    loader: DataLoader,
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    device: torch.device,
    lp_reference: torch.Tensor | None,
    previous_smoothed: torch.Tensor | None,
    lp_ema_beta: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], torch.Tensor, torch.Tensor]:
    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()

    num_samples = len(dataset)
    raw = torch.empty(num_samples, dtype=torch.float32)
    raw_cosine = torch.empty(num_samples, dtype=torch.float32)
    raw_relative = torch.empty(num_samples, dtype=torch.float32)
    raw_norm_gap = torch.empty(num_samples, dtype=torch.float32)
    groups = torch.empty(num_samples, dtype=torch.long)

    with torch.no_grad():
        for context, target, sample_ids, group_ids in loader:
            context = context.to(device).flatten(1)
            target = target.to(device).flatten(1)
            sample_ids = sample_ids.long()

            context_latent = context_encoder(context)
            target_latent = target_encoder(target)
            prediction = predictor(context_latent)

            batch_surprise = per_sample_mse(prediction, target_latent).detach().cpu()
            batch_cosine = per_sample_cosine_distance(prediction, target_latent).detach().cpu()
            batch_relative = per_sample_relative_mse(prediction, target_latent).detach().cpu()
            batch_norm_gap = per_sample_norm_gap(prediction, target_latent).detach().cpu()

            raw[sample_ids] = batch_surprise
            raw_cosine[sample_ids] = batch_cosine
            raw_relative[sample_ids] = batch_relative
            raw_norm_gap[sample_ids] = batch_norm_gap
            groups[sample_ids] = group_ids.long()

    normalized = zscore(raw)

    if previous_smoothed is None:
        smoothed = raw.clone()
    else:
        smoothed = lp_ema_beta * previous_smoothed + (1.0 - lp_ema_beta) * raw

    if lp_reference is None:
        lp = torch.full_like(raw, float("nan"))
        lp_pos = torch.zeros_like(raw)
        useful = torch.zeros_like(raw)
        useful_band = torch.zeros_like(raw)
    else:
        lp = lp_reference - smoothed
        lp_pos = lp.clamp_min(0.0)
        useful = useful_surprise_score(smoothed, lp_reference)
        useful_band = bandpass_useful_surprise_score(smoothed, lp_reference)

    sample_rows: list[dict[str, object]] = []
    for sample_id in range(num_samples):
        group_id = int(groups[sample_id].item())
        lp_value = float(lp[sample_id].item())
        sample_rows.append(
            {
                "epoch": epoch,
                "split": split,
                "sample_id": sample_id,
                "group_id": group_id,
                "group": GROUP_NAMES[group_id],
                "raw_surprise": float(raw[sample_id].item()),
                "raw_mse_surprise": float(raw[sample_id].item()),
                "raw_cosine_surprise": float(raw_cosine[sample_id].item()),
                "raw_relative_surprise": float(raw_relative[sample_id].item()),
                "raw_norm_gap_surprise": float(raw_norm_gap[sample_id].item()),
                "normalized_surprise": float(normalized[sample_id].item()),
                "smoothed_surprise": float(smoothed[sample_id].item()),
                "learning_progress": "" if math.isnan(lp_value) else lp_value,
                "positive_learning_progress": float(lp_pos[sample_id].item()),
                "useful_surprise": float(useful[sample_id].item()),
                "useful_surprise_penalty": float(useful[sample_id].item()),
                "useful_surprise_band": float(useful_band[sample_id].item()),
            }
        )

    summary_rows: list[dict[str, object]] = []
    for group_id, group_name in GROUP_NAMES.items():
        mask = groups == group_id
        count = int(mask.sum().item())
        group_lp = lp[mask]
        finite_lp = group_lp[torch.isfinite(group_lp)]

        summary_rows.append(
            {
                "epoch": epoch,
                "split": split,
                "group_id": group_id,
                "group": group_name,
                "count": count,
                "raw_surprise_mean": float(raw[mask].mean().item()),
                "raw_mse_surprise_mean": float(raw[mask].mean().item()),
                "raw_cosine_surprise_mean": float(raw_cosine[mask].mean().item()),
                "raw_relative_surprise_mean": float(raw_relative[mask].mean().item()),
                "raw_norm_gap_surprise_mean": float(raw_norm_gap[mask].mean().item()),
                "normalized_surprise_mean": float(normalized[mask].mean().item()),
                "smoothed_surprise_mean": float(smoothed[mask].mean().item()),
                "learning_progress_mean": (
                    "" if finite_lp.numel() == 0 else float(finite_lp.mean().item())
                ),
                "positive_learning_progress_mean": float(lp_pos[mask].mean().item()),
                "useful_surprise_mean": float(useful[mask].mean().item()),
                "useful_surprise_penalty_mean": float(useful[mask].mean().item()),
                "useful_surprise_band_mean": float(useful_band[mask].mean().item()),
            }
        )

    return sample_rows, summary_rows, raw.clone(), smoothed.clone()


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
) -> float:
    context_encoder.train()
    predictor.train()
    target_encoder.eval()

    total_loss = 0.0
    total_count = 0

    for context, target, _, _ in loader:
        context = context.to(device).flatten(1)
        target = target.to(device).flatten(1)

        optimizer.zero_grad(set_to_none=True)

        context_latent = context_encoder(context)
        target_latent = target_encoder(target)
        if stop_gradient:
            target_latent = target_latent.detach()

        prediction = predictor(context_latent)
        per_sample = per_sample_mse(prediction, target_latent)
        loss = per_sample.mean()

        loss.backward()
        optimizer.step()
        update_ema(target_encoder, context_encoder, ema_decay)

        total_loss += float(loss.detach().cpu().item()) * context.shape[0]
        total_count += context.shape[0]

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/surprise_synthetic")
    parser.add_argument("--architecture", choices=("linear", "nonlinear"), default="nonlinear")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--context-steps", type=int, default=32)
    parser.add_argument("--train-samples-per-group", type=int, default=1024)
    parser.add_argument("--validation-samples-per-group", type=int, default=512)
    parser.add_argument("--lp-delta", type=int, default=5)
    parser.add_argument("--lp-ema-beta", type=float, default=0.85)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.sequence_length != 2 * args.context_steps:
        raise ValueError("sequence_length must be equal to 2 * context_steps")

    data_config = SurpriseSyntheticConfig(
        train_samples_per_group=args.train_samples_per_group,
        validation_samples_per_group=args.validation_samples_per_group,
        sequence_length=args.sequence_length,
        context_steps=args.context_steps,
        seed=args.seed,
    )
    train_dataset, validation_dataset = build_surprise_datasets(data_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = resolve_device(args.device)

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
    run_dir = Path(args.output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    all_sample_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []

    smoothed_by_split: dict[str, torch.Tensor | None] = {
        "train": None,
        "validation": None,
    }
    smoothed_history_by_split: dict[str, list[torch.Tensor]] = {
        "train": [],
        "validation": [],
    }

    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            loader=train_loader,
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            optimizer=optimizer,
            device=device,
            stop_gradient=True,
            ema_decay=args.ema_decay,
        )

        train_reference = (
            smoothed_history_by_split["train"][-args.lp_delta]
            if len(smoothed_history_by_split["train"]) >= args.lp_delta
            else None
        )
        validation_reference = (
            smoothed_history_by_split["validation"][-args.lp_delta]
            if len(smoothed_history_by_split["validation"]) >= args.lp_delta
            else None
        )

        train_sample_rows, train_summary_rows, train_raw, train_smoothed = evaluate_split(
            epoch=epoch,
            split="train",
            dataset=train_dataset,
            loader=train_eval_loader,
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            device=device,
            lp_reference=train_reference,
            previous_smoothed=smoothed_by_split["train"],
            lp_ema_beta=args.lp_ema_beta,
        )

        val_sample_rows, val_summary_rows, val_raw, val_smoothed = evaluate_split(
            epoch=epoch,
            split="validation",
            dataset=validation_dataset,
            loader=validation_loader,
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            device=device,
            lp_reference=validation_reference,
            previous_smoothed=smoothed_by_split["validation"],
            lp_ema_beta=args.lp_ema_beta,
        )

        smoothed_by_split["train"] = train_smoothed
        smoothed_by_split["validation"] = val_smoothed
        smoothed_history_by_split["train"].append(train_smoothed)
        smoothed_history_by_split["validation"].append(val_smoothed)

        all_sample_rows.extend(train_sample_rows)
        all_sample_rows.extend(val_sample_rows)
        all_summary_rows.extend(train_summary_rows)
        all_summary_rows.extend(val_summary_rows)

        val_raw_mean = np.mean(
            [
                float(row["raw_surprise_mean"])
                for row in val_summary_rows
            ]
        )
        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6g} | "
            f"val_group_raw_mean={val_raw_mean:.6g}",
            flush=True,
        )

    write_csv(run_dir / "surprise_history.csv", all_sample_rows)
    write_csv(run_dir / "surprise_summary.csv", all_summary_rows)

    (run_dir / "surprise_validation.svg").write_text(
        render_surprise_summary_svg(
            all_summary_rows,
            split="validation",
            title="Surprise synthetic experiment — validation",
        )
    )
    (run_dir / "surprise_train.svg").write_text(
        render_surprise_summary_svg(
            all_summary_rows,
            split="train",
            title="Surprise synthetic experiment — train",
        )
    )

    status = {
        "state": "complete",
        "run_dir": str(run_dir),
        "elapsed_seconds": time.time() - started,
        "config": {
            "data": asdict(data_config),
            "architecture": args.architecture,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "learning_rate": args.learning_rate,
            "ema_decay": args.ema_decay,
            "lp_delta": args.lp_delta,
            "lp_ema_beta": args.lp_ema_beta,
            "seed": args.seed,
            "device": str(device),
            "stop_gradient": True,
            "target_update": "ema",
        },
        "artifacts": [
            "surprise_history.csv",
            "surprise_summary.csv",
            "surprise_validation.svg",
            "surprise_train.svg",
        ],
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))

    print(json.dumps({"state": "complete", "run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()