#!/usr/bin/env python3
"""Pretrain a compact causal Video-JEPA on deterministic Moving-MNIST."""

from __future__ import annotations

import argparse
import contextlib
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from jepa.moving_mnist import MovingMNISTDataset, MovingMNISTSpec, make_fake_digit_bank
from jepa.video_jepa import CausalVideoJEPA, CausalVideoJEPAConfig, cosine_ema_momentum
from jepa.video_diagnostics import (
    accumulate_pairing_losses,
    finalize_pairing_metrics,
    pairing_context_batch,
    token_sequence_diagnostics,
)
from jepa.video_sampling import (
    ADAPTIVE_STRATEGIES,
    AdaptiveSampleState,
    VideoSamplingConfig,
    selection_diagnostics,
)
from jepa.video_utils import (
    atomic_json_dump,
    atomic_torch_save,
    atomic_yaml_dump,
    load_yaml,
    read_csv_rows,
    resolve_device,
    seed_everything,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/moving_mnist/base.yaml")
    parser.add_argument("--strategy")
    parser.add_argument("--score-update-mode")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-root")
    parser.add_argument("--run-name")
    parser.add_argument("--run-label")
    parser.add_argument("--display-name")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--encoder-learning-rate", type=float)
    parser.add_argument("--predictor-learning-rate", type=float)
    parser.add_argument("--ema-start", type=float)
    parser.add_argument("--target-weighting-mode", choices=["uniform", "foreground", "motion"])
    parser.add_argument("--foreground-weight", type=float)
    parser.add_argument("--motion-weight-min", type=float)
    parser.add_argument("--motion-weight-max", type=float)
    parser.add_argument("--resume")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def nested(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = config.get(section, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {section!r} must be a mapping")
    return value.get(key, default)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in config.items()
    }
    config.setdefault("experiment", {})
    config.setdefault("sampling", {})
    config.setdefault("training", {})
    config.setdefault("objective", {})
    if args.strategy is not None:
        config["sampling"]["strategy"] = args.strategy
    if args.score_update_mode is not None:
        config["sampling"]["score_update_mode"] = args.score_update_mode
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.device is not None:
        config["experiment"]["device"] = args.device
    if args.output_root is not None:
        config["experiment"]["output_root"] = args.output_root
    if args.run_label is not None:
        config["experiment"]["run_label"] = args.run_label
    if args.display_name is not None:
        config["experiment"]["display_name"] = args.display_name
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
        config["training"]["encoder_learning_rate"] = args.learning_rate
        config["training"]["predictor_learning_rate"] = args.learning_rate
    if args.encoder_learning_rate is not None:
        config["training"]["encoder_learning_rate"] = args.encoder_learning_rate
    if args.predictor_learning_rate is not None:
        config["training"]["predictor_learning_rate"] = args.predictor_learning_rate
    if args.ema_start is not None:
        config["training"]["ema_start"] = args.ema_start
    if args.target_weighting_mode is not None:
        config["objective"]["target_weighting_mode"] = args.target_weighting_mode
    if args.foreground_weight is not None:
        config["objective"]["foreground_weight"] = args.foreground_weight
    if args.motion_weight_min is not None:
        config["objective"]["motion_weight_min"] = args.motion_weight_min
    if args.motion_weight_max is not None:
        config["objective"]["motion_weight_max"] = args.motion_weight_max
    return config


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
    source = str(data.get("source", "mnist"))
    kwargs: dict[str, Any] = {}
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


def make_sampling_config(config: dict[str, Any]) -> VideoSamplingConfig:
    sampling = config["sampling"]
    return VideoSamplingConfig(
        strategy=str(sampling["strategy"]),  # type: ignore[arg-type]
        score_update_mode=str(  # type: ignore[arg-type]
            sampling.get("score_update_mode", "oracle_epochwise")
        ),
        warmup_epochs=int(sampling.get("warmup_epochs", 5)),
        uniform_mix=float(sampling.get("uniform_mix", 0.4)),
        loss_ema_beta=float(sampling.get("loss_ema_beta", 0.8)),
        progress_ema_beta=float(sampling.get("progress_ema_beta", 0.8)),
        surprise_alpha=float(sampling.get("surprise_alpha", 1.0)),
        band_center=float(sampling.get("band_center", 0.6)),
        band_width=float(sampling.get("band_width", 0.18)),
        progress_boost=float(sampling.get("progress_boost", 2.0)),
        progress_margin=float(sampling.get("progress_margin", 0.0)),
        quarantine_percentile=float(sampling.get("quarantine_percentile", 0.97)),
        min_weight=float(sampling.get("min_weight", 0.25)),
        max_weight=float(sampling.get("max_weight", 4.0)),
    )


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)


def base_loader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    workers = int(training.get("num_workers", 0))
    return {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "worker_init_fn": worker_seed if workers > 0 else None,
    }


def make_train_loader(
    dataset: MovingMNISTDataset,
    sampling: VideoSamplingConfig,
    weights: Tensor,
    *,
    batch_size: int,
    epoch: int,
    seed: int,
    loader_kwargs: dict[str, Any],
) -> DataLoader[dict[str, Tensor]]:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    common = {
        "dataset": dataset,
        "batch_size": batch_size,
        "drop_last": True,
        "generator": generator,
        **loader_kwargs,
    }
    if sampling.strategy == "uniform_shuffle":
        return DataLoader(shuffle=True, **common)
    if sampling.strategy == "uniform_replacement":
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=(len(dataset) // batch_size) * batch_size,
            generator=generator,
        )
        return DataLoader(sampler=sampler, shuffle=False, **common)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=(len(dataset) // batch_size) * batch_size,
        replacement=True,
        generator=generator,
    )
    return DataLoader(sampler=sampler, shuffle=False, **common)


def make_eval_loader(
    dataset: MovingMNISTDataset,
    *,
    batch_size: int,
    loader_kwargs: dict[str, Any],
) -> DataLoader[dict[str, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )


def autocast_context(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager[Any]:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def gradient_norm(parameters: list[Tensor]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


@torch.no_grad()
def evaluate_model(
    model: CausalVideoJEPA,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    amp: bool,
    diagnostics_samples: int,
    pairing_samples: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    unweighted_loss_sum = 0.0
    foreground_weighted_loss_sum = 0.0
    motion_weighted_loss_sum = 0.0
    foreground_loss_sum = 0.0
    background_loss_sum = 0.0
    motion_active_loss_sum = 0.0
    motion_inactive_loss_sum = 0.0
    foreground_fraction_sum = 0.0
    motion_active_fraction_sum = 0.0
    motion_activity_sum = 0.0
    motion_weight_std_sum = 0.0
    motion_weight_max_sum = 0.0
    sample_count = 0
    diagnostic_tokens: dict[str, list[Tensor]] = {
        "context": [],
        "predicted": [],
        "target": [],
    }
    diagnostic_count = 0
    pairing_sums: dict[str, float] = {}
    pairing_counts: dict[str, int] = {}
    pairing_count = 0

    for batch in loader:
        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            output = model(context, target)
        batch_size = len(context)
        loss_sum += float(output["per_sample_loss"].float().sum())
        unweighted_loss_sum += float(
            output["per_sample_unweighted_loss"].float().sum()
        )
        foreground_weighted_loss_sum += (
            float(output["foreground_weighted_loss"].float()) * batch_size
        )
        motion_weighted_loss_sum += (
            float(output["motion_weighted_loss"].float()) * batch_size
        )
        foreground_loss_sum += float(output["foreground_loss"].float()) * batch_size
        background_loss_sum += float(output["background_loss"].float()) * batch_size
        motion_active_loss_sum += float(output["motion_active_loss"].float()) * batch_size
        motion_inactive_loss_sum += float(output["motion_inactive_loss"].float()) * batch_size
        foreground_fraction_sum += (
            float(output["foreground_fraction"].float()) * batch_size
        )
        motion_active_fraction_sum += (
            float(output["motion_active_fraction"].float()) * batch_size
        )
        motion_activity_sum += float(output["motion_activity_mean"].float()) * batch_size
        motion_weight_std_sum += float(output["motion_weight_std"].float()) * batch_size
        motion_weight_max_sum += float(output["motion_weight_max"].float()) * batch_size
        sample_count += batch_size

        if diagnostic_count < diagnostics_samples:
            take = min(batch_size, diagnostics_samples - diagnostic_count)
            diagnostic_tokens["context"].append(
                output["context_tokens"][:take].float().cpu()
            )
            diagnostic_tokens["predicted"].append(
                output["predicted_tokens"][:take].float().cpu()
            )
            diagnostic_tokens["target"].append(
                output["target_tokens"][:take].float().cpu()
            )
            diagnostic_count += take

        if pairing_count < pairing_samples:
            take = min(batch_size, pairing_samples - pairing_count)
            with autocast_context(device, amp):
                pairing = pairing_context_batch(
                    model,
                    context[:take],
                    target[:take],
                    digit_labels=batch["digit_label"][:take],
                    direction_labels=batch["direction_label"][:take],
                )
            accumulate_pairing_losses(pairing_sums, pairing_counts, pairing)
            pairing_count += take

    if was_training:
        model.train()

    denominator = max(sample_count, 1)
    result = {
        "jepa_loss": loss_sum / denominator,
        "unweighted_jepa_loss": unweighted_loss_sum / denominator,
        "foreground_weighted_jepa_loss": foreground_weighted_loss_sum / denominator,
        "motion_weighted_jepa_loss": motion_weighted_loss_sum / denominator,
        "foreground_jepa_loss": foreground_loss_sum / denominator,
        "background_jepa_loss": background_loss_sum / denominator,
        "motion_active_jepa_loss": motion_active_loss_sum / denominator,
        "motion_inactive_jepa_loss": motion_inactive_loss_sum / denominator,
        "foreground_token_fraction": foreground_fraction_sum / denominator,
        "motion_active_token_fraction": motion_active_fraction_sum / denominator,
        "motion_activity_mean": motion_activity_sum / denominator,
        "motion_weight_std": motion_weight_std_sum / denominator,
        "motion_weight_max": motion_weight_max_sum / denominator,
    }
    for prefix, chunks in diagnostic_tokens.items():
        if chunks:
            result.update(token_sequence_diagnostics(torch.cat(chunks, dim=0), prefix))
    result.update(finalize_pairing_metrics(pairing_sums, pairing_counts))
    return result


@torch.no_grad()
def score_training_set(
    model: CausalVideoJEPA,
    loader: DataLoader[dict[str, Tensor]],
    state: AdaptiveSampleState,
    device: torch.device,
    *,
    amp: bool,
) -> int:
    was_training = model.training
    model.eval()
    scored = 0
    for batch in loader:
        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            output = model(context, target)
        state.update(batch["sample_id"], output["per_sample_loss"])
        scored += len(context)
    if was_training:
        model.train()
    return scored


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    minimum_ratios: list[float],
) -> torch.optim.lr_scheduler.LambdaLR:
    if len(minimum_ratios) != len(optimizer.param_groups):
        raise ValueError("one minimum LR ratio is required per optimizer group")

    def build_schedule(minimum_ratio: float):
        def schedule(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max((step + 1) / warmup_steps, 1e-8)
            denominator = max(total_steps - warmup_steps, 1)
            progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        return schedule

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [build_schedule(value) for value in minimum_ratios],
    )


def metric_rows(
    values: dict[str, float],
    *,
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
            "epoch": epoch,
            "step": step,
            "clips_seen": clips_seen,
            "scoring_clips": scoring_clips,
            "strategy": strategy,
            "run_label": run_label,
            "display_name": display_name,
            "seed": seed,
            "split": split,
            "metric": key,
            "value": value,
        }
        for key, value in values.items()
    ]


def checkpoint_payload(
    model: CausalVideoJEPA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    state: AdaptiveSampleState,
    config: dict[str, Any],
    *,
    epoch: int,
    step: int,
    clips_seen: int,
    scoring_clips: int,
) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "epoch": epoch,
        "step": step,
        "clips_seen": clips_seen,
        "scoring_clips": scoring_clips,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "sampler_state": state.state_dict(),
    }


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_yaml(args.config), args)
    experiment = config["experiment"]
    training = config["training"]
    sampling_config = make_sampling_config(config)
    sampling_config.validate()
    seed = int(experiment.get("seed", 0))
    device = resolve_device(str(experiment.get("device", "auto")))
    seed_everything(seed)

    output_root = Path(experiment["output_root"])
    default_name = f"{sampling_config.strategy}_{sampling_config.score_update_mode}_seed{seed}"
    run_name = args.run_name or default_name
    run_label = str(experiment.get("run_label", sampling_config.strategy))
    default_display = sampling_config.strategy.replace("_", " ").title()
    display_name = str(experiment.get("display_name", default_display))
    run_dir = output_root / run_name
    if run_dir.exists() and args.resume is None:
        if not args.overwrite:
            raise SystemExit(f"run directory already exists: {run_dir}; use --overwrite")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    atomic_yaml_dump(run_dir / "config.yaml", config)
    atomic_json_dump(
        run_dir / "status.json",
        {"state": "running", "run_dir": str(run_dir), "config": config},
    )

    train_dataset = make_dataset(
        config,
        "pretrain_train",
        int(config["data"]["train_samples"]),
        seed + 101,
    )
    validation_dataset = make_dataset(
        config,
        "pretrain_validation",
        int(config["data"]["validation_samples"]),
        seed + 211,
    )
    loader_kwargs = base_loader_kwargs(config, device)
    batch_size = int(training["batch_size"])
    eval_batch_size = int(training.get("eval_batch_size", batch_size))
    validation_loader = make_eval_loader(
        validation_dataset,
        batch_size=eval_batch_size,
        loader_kwargs=loader_kwargs,
    )
    scoring_loader = make_eval_loader(
        train_dataset,
        batch_size=eval_batch_size,
        loader_kwargs=loader_kwargs,
    )

    model = CausalVideoJEPA(make_model_config(config)).to(device)
    encoder_parameters = list(model.context_encoder.parameters())
    predictor_parameters = list(model.predictor.parameters())
    trainable_parameters = encoder_parameters + predictor_parameters
    legacy_learning_rate = float(training.get("learning_rate", 2.5e-4))
    encoder_learning_rate = float(
        training.get("encoder_learning_rate", legacy_learning_rate)
    )
    predictor_learning_rate = float(
        training.get("predictor_learning_rate", legacy_learning_rate)
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": encoder_learning_rate,
                "group_name": "encoder",
            },
            {
                "params": predictor_parameters,
                "lr": predictor_learning_rate,
                "group_name": "predictor",
            },
        ],
        weight_decay=float(training.get("weight_decay", 0.05)),
        betas=(0.9, 0.95),
    )
    epochs = int(training["epochs"])
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(training.get("lr_warmup_epochs", 0)) * steps_per_epoch
    shared_minimum = float(training.get("minimum_learning_rate", 1e-6))
    minimum_encoder = float(
        training.get("minimum_encoder_learning_rate", shared_minimum)
    )
    minimum_predictor = float(
        training.get("minimum_predictor_learning_rate", shared_minimum)
    )
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        minimum_ratios=[
            minimum_encoder / encoder_learning_rate,
            minimum_predictor / predictor_learning_rate,
        ],
    )
    sampler_state = AdaptiveSampleState(len(train_dataset), sampling_config)

    pretrain_rows: list[dict[str, Any]] = (
        read_csv_rows(run_dir / "pretrain_metrics.csv") if args.resume is not None else []
    )
    sampler_rows: list[dict[str, Any]] = (
        read_csv_rows(run_dir / "sampler_metrics.csv") if args.resume is not None else []
    )
    start_epoch = 1
    global_step = 0
    clips_seen = 0
    scoring_clips = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        sampler_state.load_state_dict(checkpoint["sampler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])
        clips_seen = int(checkpoint["clips_seen"])
        scoring_clips = int(checkpoint.get("scoring_clips", 0))

    amp = bool(training.get("amp", True))
    grad_clip = float(training.get("gradient_clip", 1.0))
    diagnostics_samples = int(training.get("diagnostics_samples", 256))
    pairing_samples = int(training.get("pairing_diagnostics_samples", diagnostics_samples))
    checkpoint_every = int(training.get("checkpoint_every", 1))
    ema_start = float(training.get("ema_start", 0.99))
    ema_end = float(training.get("ema_end", 0.9999))
    run_started = time.time()

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.time()
            weights = sampler_state.weights(epoch)
            weight_diagnostics = sampler_state.diagnostics(weights)
            train_loader = make_train_loader(
                train_dataset,
                sampling_config,
                weights,
                batch_size=batch_size,
                epoch=epoch,
                seed=seed,
                loader_kwargs=loader_kwargs,
            )
            model.train()
            loss_sum = 0.0
            unweighted_loss_sum = 0.0
            foreground_weighted_loss_sum = 0.0
            motion_weighted_loss_sum = 0.0
            foreground_loss_sum = 0.0
            background_loss_sum = 0.0
            motion_active_loss_sum = 0.0
            motion_inactive_loss_sum = 0.0
            foreground_fraction_sum = 0.0
            motion_active_fraction_sum = 0.0
            motion_activity_sum = 0.0
            motion_weight_std_sum = 0.0
            motion_weight_max_sum = 0.0
            trained_samples = 0
            gradient_norm_sum = 0.0
            encoder_gradient_norm_sum = 0.0
            predictor_gradient_norm_sum = 0.0
            selected_ids: list[Tensor] = []
            selected_bounce: list[Tensor] = []
            selected_speed: list[Tensor] = []

            for batch in train_loader:
                context = batch["context"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, amp):
                    output = model(context, target)
                    loss = output["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite training loss at epoch {epoch}, step {global_step}"
                    )
                loss.backward()
                raw_gradient_norm = gradient_norm(trainable_parameters)
                raw_encoder_gradient_norm = gradient_norm(encoder_parameters)
                raw_predictor_gradient_norm = gradient_norm(predictor_parameters)
                if grad_clip > 0.0:
                    clip_grad_norm_(trainable_parameters, grad_clip)
                optimizer.step()
                scheduler.step()
                momentum = cosine_ema_momentum(
                    global_step,
                    total_steps,
                    ema_start,
                    ema_end,
                )
                model.update_target_encoder(momentum)

                if (
                    sampling_config.is_adaptive
                    and sampling_config.score_update_mode == "online_cached"
                ):
                    sampler_state.update(batch["sample_id"], output["per_sample_loss"])

                current_batch = len(context)
                loss_sum += float(output["per_sample_loss"].detach().float().sum())
                unweighted_loss_sum += float(
                    output["per_sample_unweighted_loss"].detach().float().sum()
                )
                foreground_weighted_loss_sum += (
                    float(output["foreground_weighted_loss"].detach().float())
                    * current_batch
                )
                motion_weighted_loss_sum += (
                    float(output["motion_weighted_loss"].detach().float())
                    * current_batch
                )
                foreground_loss_sum += (
                    float(output["foreground_loss"].detach().float()) * current_batch
                )
                background_loss_sum += (
                    float(output["background_loss"].detach().float()) * current_batch
                )
                motion_active_loss_sum += (
                    float(output["motion_active_loss"].detach().float()) * current_batch
                )
                motion_inactive_loss_sum += (
                    float(output["motion_inactive_loss"].detach().float()) * current_batch
                )
                foreground_fraction_sum += (
                    float(output["foreground_fraction"].detach().float())
                    * current_batch
                )
                motion_active_fraction_sum += (
                    float(output["motion_active_fraction"].detach().float())
                    * current_batch
                )
                motion_activity_sum += (
                    float(output["motion_activity_mean"].detach().float())
                    * current_batch
                )
                motion_weight_std_sum += (
                    float(output["motion_weight_std"].detach().float())
                    * current_batch
                )
                motion_weight_max_sum += (
                    float(output["motion_weight_max"].detach().float())
                    * current_batch
                )
                trained_samples += current_batch
                clips_seen += current_batch
                global_step += 1
                gradient_norm_sum += raw_gradient_norm
                encoder_gradient_norm_sum += raw_encoder_gradient_norm
                predictor_gradient_norm_sum += raw_predictor_gradient_norm
                selected_ids.append(batch["sample_id"].cpu())
                selected_bounce.append(batch["bounce_target"].cpu())
                selected_speed.append(batch["speed"].cpu())

            trained_denominator = max(trained_samples, 1)
            train_metrics = {
                "jepa_loss": loss_sum / trained_denominator,
                "unweighted_jepa_loss": unweighted_loss_sum / trained_denominator,
                "foreground_weighted_jepa_loss": (
                    foreground_weighted_loss_sum / trained_denominator
                ),
                "motion_weighted_jepa_loss": (
                    motion_weighted_loss_sum / trained_denominator
                ),
                "foreground_jepa_loss": foreground_loss_sum / trained_denominator,
                "background_jepa_loss": background_loss_sum / trained_denominator,
                "motion_active_jepa_loss": motion_active_loss_sum / trained_denominator,
                "motion_inactive_jepa_loss": (
                    motion_inactive_loss_sum / trained_denominator
                ),
                "foreground_token_fraction": (
                    foreground_fraction_sum / trained_denominator
                ),
                "motion_active_token_fraction": (
                    motion_active_fraction_sum / trained_denominator
                ),
                "motion_activity_mean": motion_activity_sum / trained_denominator,
                "motion_weight_std": motion_weight_std_sum / trained_denominator,
                "motion_weight_max": motion_weight_max_sum / trained_denominator,
                "foreground_weight": model.config.foreground_weight,
                "gradient_norm": gradient_norm_sum / max(len(train_loader), 1),
                "encoder_gradient_norm": (
                    encoder_gradient_norm_sum / max(len(train_loader), 1)
                ),
                "predictor_gradient_norm": (
                    predictor_gradient_norm_sum / max(len(train_loader), 1)
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "predictor_learning_rate": float(optimizer.param_groups[1]["lr"]),
                "ema_momentum": cosine_ema_momentum(
                    max(global_step - 1, 0), total_steps, ema_start, ema_end
                ),
                "epoch_seconds": time.time() - epoch_started,
            }
            validation_metrics = evaluate_model(
                model,
                validation_loader,
                device,
                amp=amp,
                diagnostics_samples=diagnostics_samples,
                pairing_samples=pairing_samples,
            )

            if (
                sampling_config.is_adaptive
                and sampling_config.score_update_mode == "oracle_epochwise"
            ):
                scoring_clips += score_training_set(
                    model,
                    scoring_loader,
                    sampler_state,
                    device,
                    amp=amp,
                )

            selection = selection_diagnostics(
                torch.cat(selected_ids),
                dataset_size=len(train_dataset),
                selected_bounce=torch.cat(selected_bounce),
                selected_speed=torch.cat(selected_speed),
            )
            current_weights = sampler_state.weights(epoch + 1)
            sampler_metrics = {
                **weight_diagnostics,
                **selection,
                **{
                    f"next_{key}": value
                    for key, value in sampler_state.diagnostics(current_weights).items()
                },
            }
            pretrain_rows.extend(
                metric_rows(
                    train_metrics,
                    epoch=epoch,
                    step=global_step,
                    clips_seen=clips_seen,
                    scoring_clips=scoring_clips,
                    strategy=sampling_config.strategy,
                    run_label=run_label,
                    display_name=display_name,
                    seed=seed,
                    split="train",
                )
            )
            pretrain_rows.extend(
                metric_rows(
                    validation_metrics,
                    epoch=epoch,
                    step=global_step,
                    clips_seen=clips_seen,
                    scoring_clips=scoring_clips,
                    strategy=sampling_config.strategy,
                    run_label=run_label,
                    display_name=display_name,
                    seed=seed,
                    split="validation",
                )
            )
            sampler_rows.extend(
                metric_rows(
                    sampler_metrics,
                    epoch=epoch,
                    step=global_step,
                    clips_seen=clips_seen,
                    scoring_clips=scoring_clips,
                    strategy=sampling_config.strategy,
                    run_label=run_label,
                    display_name=display_name,
                    seed=seed,
                    split="sampler",
                )
            )
            write_csv_rows(run_dir / "pretrain_metrics.csv", pretrain_rows)
            write_csv_rows(run_dir / "sampler_metrics.csv", sampler_rows)

            if epoch % checkpoint_every == 0 or epoch == epochs:
                payload = checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    sampler_state,
                    config,
                    epoch=epoch,
                    step=global_step,
                    clips_seen=clips_seen,
                    scoring_clips=scoring_clips,
                )
                epoch_path = checkpoints_dir / f"epoch_{epoch:04d}.pt"
                atomic_torch_save(epoch_path, payload)
                atomic_torch_save(checkpoints_dir / "latest.pt", payload)

            print(
                f"epoch {epoch:03d}/{epochs} | train {train_metrics['jepa_loss']:.5f} "
                f"| val {validation_metrics['jepa_loss']:.5f} "
                f"| pooled-rank {validation_metrics.get('context_pooled_effective_rank', float('nan')):.1f} "
                f"| token-rank {validation_metrics.get('context_token_effective_rank', float('nan')):.1f} "
                f"| clips {clips_seen} | score clips {scoring_clips}",
                flush=True,
            )

        final_validation = {
            row["metric"]: float(row["value"])
            for row in pretrain_rows
            if int(float(row["epoch"])) == epochs and row["split"] == "validation"
        }
        atomic_json_dump(
            run_dir / "status.json",
            {
                "state": "completed",
                "run_dir": str(run_dir),
                "config": config,
                "epoch": epochs,
                "step": global_step,
                "clips_seen": clips_seen,
                "scoring_clips": scoring_clips,
                "wall_seconds": time.time() - run_started,
                "final_validation": final_validation,
            },
        )
    except Exception as exc:
        atomic_json_dump(
            run_dir / "status.json",
            {
                "state": "failed",
                "run_dir": str(run_dir),
                "config": config,
                "epoch": locals().get("epoch", start_epoch - 1),
                "step": global_step,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    print(run_dir)


if __name__ == "__main__":
    main()
