"""Frame-weighting curriculum helpers for spatial I-JEPA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from jepa.configs.base import derive_seed
from jepa.training.images.ijepa_spatial import (
    MaskConfig,
    SpatialIJEPACore,
    encode_samples_pooled,
    sample_masks,
    spatial_ijepa_per_sample_loss,
)

SpatialWeightingMethod = Literal["uniform", "loss", "ras", "coord"]
SpatialRichnessFunctional = Literal["logdet", "rbar", "pr"]
RASScoreGranularity = Literal["sample", "batch"]
CoordinateImportanceMethod = Literal["covariance", "transformation", "dynamics"]


@dataclass(frozen=True, slots=True)
class SpatialWeightingConfig:
    method: SpatialWeightingMethod = "uniform"
    warmup_epochs: int = 0
    update_every_epochs: int = 1
    temperature: float = 1.0
    replay_beta: float = 1.0
    uniform_mix: float = 0.05
    score_batch_size: int = 0
    ref_size: int = 1024
    richness_functional: SpatialRichnessFunctional = "logdet"
    richness_delta: float = 1.0e-4
    richness_trace_target: float = 1.0
    richness_trace_beta: float = 0.01
    ras_score_granularity: RASScoreGranularity = "sample"
    coordinate_importance: CoordinateImportanceMethod = "covariance"
    coordinate_ema_beta: float = 0.1
    coordinate_delta: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class SpatialWeightingState:
    memory: torch.Tensor
    probabilities: torch.Tensor
    last_scores: torch.Tensor
    coordinate_importance: torch.Tensor | None = None
    coordinate_previous_ref_latents: torch.Tensor | None = None


def init_spatial_weighting(num_frames: int) -> SpatialWeightingState:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    probabilities = torch.full((num_frames,), 1.0 / num_frames, dtype=torch.float64)
    return SpatialWeightingState(
        memory=torch.zeros(num_frames, dtype=torch.float64),
        probabilities=probabilities,
        last_scores=torch.zeros(num_frames, dtype=torch.float64),
    )


def _validate_config(config: SpatialWeightingConfig) -> None:
    if config.method not in {"uniform", "loss", "ras", "coord"}:
        raise ValueError(f"unknown spatial weighting method: {config.method!r}")
    if config.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if config.update_every_epochs <= 0:
        raise ValueError("update_every_epochs must be positive")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= config.replay_beta <= 1:
        raise ValueError("replay_beta must be in [0, 1]")
    if not 0 <= config.uniform_mix < 1:
        raise ValueError("uniform_mix must be in [0, 1)")
    if config.score_batch_size < 0:
        raise ValueError("score_batch_size must be non-negative")
    if config.ref_size <= 0:
        raise ValueError("ref_size must be positive")
    if config.richness_delta <= 0:
        raise ValueError("richness_delta must be positive")
    if config.richness_functional not in {"logdet", "rbar", "pr"}:
        raise ValueError(f"unknown richness functional: {config.richness_functional!r}")
    if config.richness_trace_target <= 0:
        raise ValueError("richness_trace_target must be positive")
    if config.richness_trace_beta < 0:
        raise ValueError("richness_trace_beta must be non-negative")
    if config.ras_score_granularity not in {"sample", "batch"}:
        raise ValueError(f"unknown RAS score granularity: {config.ras_score_granularity!r}")
    if config.coordinate_importance not in {"covariance", "transformation", "dynamics"}:
        raise ValueError(f"unknown coordinate importance: {config.coordinate_importance!r}")
    if not 0 <= config.coordinate_ema_beta <= 1:
        raise ValueError("coordinate_ema_beta must be in [0, 1]")
    if config.coordinate_delta <= 0:
        raise ValueError("coordinate_delta must be positive")


def should_update_weights(epoch: int, config: SpatialWeightingConfig) -> bool:
    _validate_config(config)
    return (
        config.method != "uniform"
        and epoch >= config.warmup_epochs
        and epoch % config.update_every_epochs == 0
    )


def sample_frame_indices(
    state: SpatialWeightingState,
    *,
    num_draws: int,
    seed: int,
    replacement: bool = True,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.multinomial(
        state.probabilities.float(),
        num_draws,
        replacement=replacement,
        generator=generator,
    )


def update_spatial_weights(
    state: SpatialWeightingState,
    scores: torch.Tensor,
    config: SpatialWeightingConfig,
) -> SpatialWeightingState:
    _validate_config(config)
    scores64 = scores.detach().cpu().to(torch.float64)
    if scores64.shape != state.memory.shape:
        raise ValueError(
            f"scores shape must be {tuple(state.memory.shape)}, got {tuple(scores64.shape)}"
        )
    if not torch.isfinite(scores64).all():
        raise ValueError("spatial weighting scores must be finite")

    std = scores64.std(unbiased=False)
    normalized = scores64 - scores64.mean()
    if std > 0:
        normalized = normalized / std
    else:
        normalized.zero_()

    memory = (1.0 - config.replay_beta) * state.memory + config.replay_beta * normalized
    probabilities = torch.softmax(memory / config.temperature, dim=0)
    if config.uniform_mix > 0:
        uniform = torch.full_like(probabilities, 1.0 / probabilities.numel())
        probabilities = (1.0 - config.uniform_mix) * probabilities + config.uniform_mix * uniform
    probabilities = probabilities / probabilities.sum()
    return SpatialWeightingState(
        memory=memory,
        probabilities=probabilities,
        last_scores=scores64,
        coordinate_importance=state.coordinate_importance,
        coordinate_previous_ref_latents=state.coordinate_previous_ref_latents,
    )


def update_coordinate_weighting_state(
    state: SpatialWeightingState,
    *,
    coordinate_importance: torch.Tensor,
    previous_ref_latents: torch.Tensor,
) -> SpatialWeightingState:
    return SpatialWeightingState(
        memory=state.memory,
        probabilities=state.probabilities,
        last_scores=state.last_scores,
        coordinate_importance=coordinate_importance.detach().cpu().to(torch.float64),
        coordinate_previous_ref_latents=previous_ref_latents.detach().cpu().to(torch.float64),
    )


def weighting_diagnostics(state: SpatialWeightingState) -> dict[str, float]:
    scores = state.last_scores
    probabilities = state.probabilities
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
    normalized_entropy = entropy / torch.log(torch.tensor(float(probabilities.numel())))
    sorted_probabilities = torch.sort(probabilities, descending=True).values

    def top_mass(fraction: float) -> float:
        count = max(1, int(round(probabilities.numel() * fraction)))
        return float(sorted_probabilities[:count].sum().item())

    return {
        "weighting/score_mean": float(scores.mean().item()),
        "weighting/score_std": float(scores.std(unbiased=False).item()),
        "weighting/score_min": float(scores.min().item()),
        "weighting/score_max": float(scores.max().item()),
        "weighting/prob_entropy": float(entropy.item()),
        "weighting/prob_entropy_normalized": float(normalized_entropy.item()),
        "weighting/prob_min": float(probabilities.min().item()),
        "weighting/prob_max": float(probabilities.max().item()),
        "weighting/effective_sample_size": float((1.0 / (probabilities.square().sum())).item()),
        "weighting/top_1pct_mass": top_mass(0.01),
        "weighting/top_5pct_mass": top_mass(0.05),
        "weighting/top_10pct_mass": top_mass(0.10),
    }


def select_reference_indices(num_frames: int, *, ref_size: int, seed: int) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if ref_size <= 0:
        raise ValueError("ref_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    count = min(ref_size, num_frames)
    return torch.randperm(num_frames, generator=generator)[:count]


def _latent_covariance_from_images(
    core: SpatialIJEPACore,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = encode_samples_pooled(core, images).to(torch.float64)
    centered = latents - latents.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(latents.shape[0] - 1, 1)
    return (covariance + covariance.T) / 2, latents


def richness_from_images(
    core: SpatialIJEPACore,
    images: torch.Tensor,
    *,
    functional: SpatialRichnessFunctional,
    delta: float,
    trace_target: float,
    trace_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if delta <= 0:
        raise ValueError("delta must be positive")
    if trace_target <= 0:
        raise ValueError("trace_target must be positive")
    if trace_beta < 0:
        raise ValueError("trace_beta must be non-negative")
    covariance, _ = _latent_covariance_from_images(core, images)
    trace = torch.trace(covariance)
    eye = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    regularized = covariance + delta * eye
    sign, logabsdet = torch.linalg.slogdet(regularized)
    if torch.any(sign <= 0):
        raise RuntimeError("richness covariance regularization did not produce a positive matrix")

    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    squared_trace = covariance.square().sum()
    participation_ratio = trace.square() / squared_trace.clamp_min(delta)
    trace_penalty = trace_beta * (trace - trace_target) ** 2

    if functional == "logdet":
        richness = logabsdet
    elif functional == "rbar":
        barrier_penalty = trace - logabsdet - covariance.shape[0]
        richness = -barrier_penalty
    elif functional == "pr":
        richness = participation_ratio - trace_penalty
    else:
        raise ValueError(f"unknown richness functional: {functional!r}")

    metadata = {
        "ras/richness_value": float(richness.detach().cpu().item()),
        "ras/richness_logdet": float(logabsdet.detach().cpu().item()),
        "ras/richness_trace": float(trace.detach().cpu().item()),
        "ras/richness_pr": float(participation_ratio.detach().cpu().item()),
        "ras/richness_trace_penalty": float(trace_penalty.detach().cpu().item()),
        "ras/richness_top_eigenvalue": float(eigenvalues[-1].detach().cpu().item()),
    }
    return richness, metadata


def _normalize_coordinate_importance(weights: torch.Tensor, *, delta: float) -> torch.Tensor:
    weights = weights.detach().to(torch.float64).clamp_min(0)
    mean = weights.mean().clamp_min(delta)
    weights = weights / mean
    return weights.clamp_min(delta)


def _coordinate_importance_from_covariance(
    latents: torch.Tensor,
    *,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    latents64 = latents.detach().to(torch.float64)
    centered = latents64 - latents64.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(latents64.shape[0] - 1, 1)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0)
    weights = _normalize_coordinate_importance(
        eigenvalues / eigenvalues.sum().clamp_min(delta), delta=delta
    )
    metadata = {
        "coord/importance_min": float(weights.min().item()),
        "coord/importance_max": float(weights.max().item()),
        "coord/importance_std": float(weights.std(unbiased=False).item()),
        "coord/covariance_top_eigenvalue": float(eigenvalues[-1].item()),
        "coord/covariance_trace": float(eigenvalues.sum().item()),
    }
    return weights, eigenvectors, metadata


def _coordinate_importance_from_transformation(
    source_latents: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    source64 = source_latents.detach().to(torch.float64)
    target64 = target_latents.detach().to(torch.float64)
    source_centered = source64 - source64.mean(dim=0, keepdim=True)
    target_centered = target64 - target64.mean(dim=0, keepdim=True)
    dim = source64.shape[1]
    eye = torch.eye(dim, dtype=source64.dtype, device=source64.device)
    denom = max(source64.shape[0] - 1, 1)
    source_cov = source_centered.T @ source_centered / denom
    target_cov = target_centered.T @ target_centered / denom
    cross_cov = target_centered.T @ source_centered / denom
    source_eigs, source_basis = torch.linalg.eigh((source_cov + source_cov.T) / 2)
    target_eigs, target_basis = torch.linalg.eigh((target_cov + target_cov.T) / 2)
    source_inv_sqrt = (
        source_basis
        @ torch.diag(source_eigs.clamp_min(delta).rsqrt())
        @ source_basis.T
    )
    target_inv_sqrt = (
        target_basis
        @ torch.diag(target_eigs.clamp_min(delta).rsqrt())
        @ target_basis.T
    )
    operator = target_inv_sqrt @ cross_cov @ source_inv_sqrt
    operator = torch.nan_to_num(operator, nan=0.0, posinf=0.0, neginf=0.0)
    _, singular_values, vh = torch.linalg.svd(operator + delta * eye, full_matrices=False)
    singular_values = singular_values.clamp(max=1.0)
    weights = _normalize_coordinate_importance(
        (1.0 - singular_values.abs()).clamp_min(0), delta=delta
    )
    basis = vh.T
    metadata = {
        "coord/importance_min": float(weights.min().item()),
        "coord/importance_max": float(weights.max().item()),
        "coord/importance_std": float(weights.std(unbiased=False).item()),
        "coord/transform_singular_min": float(singular_values.min().item()),
        "coord/transform_singular_max": float(singular_values.max().item()),
    }
    return weights, basis, metadata


def _coordinate_importance_from_dynamics(
    latents: torch.Tensor,
    state: SpatialWeightingState,
    *,
    ema_beta: float,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    latents64 = latents.detach().to(torch.float64)
    dim = latents64.shape[1]
    basis = torch.eye(dim, dtype=latents64.dtype, device=latents64.device)
    previous = state.coordinate_previous_ref_latents
    if previous is None or previous.shape != latents64.shape:
        instantaneous = torch.ones(dim, dtype=latents64.dtype, device=latents64.device)
        cold_start = 1.0
    else:
        previous = previous.to(device=latents64.device, dtype=latents64.dtype)
        instantaneous = (latents64 - previous).abs().mean(dim=0)
        cold_start = 0.0
    old = state.coordinate_importance
    if old is None or old.shape != instantaneous.shape:
        weights_raw = instantaneous
    else:
        old = old.to(device=latents64.device, dtype=latents64.dtype)
        weights_raw = (1.0 - ema_beta) * old + ema_beta * instantaneous
    weights = _normalize_coordinate_importance(weights_raw, delta=delta)
    metadata = {
        "coord/importance_min": float(weights.min().item()),
        "coord/importance_max": float(weights.max().item()),
        "coord/importance_std": float(weights.std(unbiased=False).item()),
        "coord/dynamics_cold_start": cold_start,
        "coord/dynamics_delta_mean": float(instantaneous.mean().item()),
    }
    return weights, basis, metadata


def _encoder_parameters(core: SpatialIJEPACore) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter for parameter in core.context_encoder.parameters() if parameter.requires_grad
    )


def _dot_gradients(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    total = torch.zeros((), dtype=right[0].dtype, device=right[0].device)
    for left_gradient, right_gradient in zip(left, right, strict=True):
        if left_gradient is None:
            continue
        total = total + (left_gradient.to(right_gradient.dtype) * right_gradient).sum()
    return total


def score_frames_by_loss(
    core: SpatialIJEPACore,
    train_images: torch.Tensor,
    *,
    grid: int,
    mask_config: MaskConfig,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("score batch_size must be positive")
    context_was_training = core.context_encoder.training
    predictor_was_training = core.predictor.training
    target_was_training = core.target_encoder.training
    core.context_encoder.eval()
    core.predictor.eval()
    core.target_encoder.eval()

    scores = torch.empty(train_images.shape[0], dtype=torch.float64)
    mask_generator = torch.Generator().manual_seed(derive_seed(seed, "weighting-loss-masks"))
    try:
        with torch.no_grad():
            for indices in torch.arange(train_images.shape[0], device=train_images.device).split(
                batch_size
            ):
                batch = train_images[indices].to(device)
                context_masks, target_masks = sample_masks(
                    grid,
                    grid,
                    mask_config,
                    mask_generator,
                    batch_size=batch.shape[0],
                )
                batch_losses = spatial_ijepa_per_sample_loss(
                    core, batch, context_masks, target_masks
                )
                scores[indices.detach().cpu()] = batch_losses.detach().cpu().to(torch.float64)
    finally:
        core.context_encoder.train(context_was_training)
        core.predictor.train(predictor_was_training)
        core.target_encoder.train(target_was_training)
    return scores


def score_frames_by_ras(
    core: SpatialIJEPACore,
    train_images: torch.Tensor,
    *,
    ref_indices: torch.Tensor,
    grid: int,
    mask_config: MaskConfig,
    batch_size: int,
    seed: int,
    device: torch.device,
    richness_functional: SpatialRichnessFunctional,
    richness_delta: float,
    richness_trace_target: float,
    richness_trace_beta: float,
    score_granularity: RASScoreGranularity = "sample",
) -> tuple[torch.Tensor, dict[str, float]]:
    if batch_size <= 0:
        raise ValueError("score batch_size must be positive")
    if score_granularity not in {"sample", "batch"}:
        raise ValueError(f"unknown RAS score granularity: {score_granularity!r}")
    context_was_training = core.context_encoder.training
    predictor_was_training = core.predictor.training
    target_was_training = core.target_encoder.training
    core.context_encoder.eval()
    core.predictor.eval()
    core.target_encoder.eval()

    parameters = _encoder_parameters(core)
    if not parameters:
        raise ValueError("RAS requires trainable context encoder parameters")

    scores = torch.empty(train_images.shape[0], dtype=torch.float64)
    mask_generator = torch.Generator().manual_seed(derive_seed(seed, "weighting-ras-masks"))
    try:
        ref_images = train_images[ref_indices.to(train_images.device)].to(device)
        richness, richness_metadata = richness_from_images(
            core,
            ref_images,
            functional=richness_functional,
            delta=richness_delta,
            trace_target=richness_trace_target,
            trace_beta=richness_trace_beta,
        )
        richness_gradients_raw = torch.autograd.grad(richness, parameters, retain_graph=False)
        richness_gradients = tuple(gradient.detach() for gradient in richness_gradients_raw)
        grad_norm = torch.sqrt(
            sum(
                gradient.detach().to(torch.float64).square().sum()
                for gradient in richness_gradients
            )
        )

        score_order = torch.randperm(
            train_images.shape[0],
            generator=torch.Generator().manual_seed(derive_seed(seed, "weighting-ras-order")),
        ).to(train_images.device)
        score_groups = score_order.split(batch_size)
        for indices in score_groups:
            batch = train_images[indices].to(device)
            context_masks, target_masks = sample_masks(
                grid,
                grid,
                mask_config,
                mask_generator,
                batch_size=batch.shape[0],
            )
            if score_granularity == "batch":
                group_loss = spatial_ijepa_per_sample_loss(
                    core,
                    batch,
                    context_masks,
                    target_masks,
                ).mean()
                loss_gradients = torch.autograd.grad(
                    group_loss,
                    parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
                group_score = float(
                    (-_dot_gradients(loss_gradients, richness_gradients))
                    .detach()
                    .cpu()
                    .item()
                )
                batch_scores = torch.full(
                    (batch.shape[0],),
                    group_score,
                    dtype=torch.float64,
                )
            else:
                batch_scores = torch.empty(batch.shape[0], dtype=torch.float64)
                for local_index in range(batch.shape[0]):
                    sample = batch[local_index : local_index + 1]
                    sample_context_masks = [
                        mask[local_index : local_index + 1] for mask in context_masks
                    ]
                    sample_target_masks = [
                        mask[local_index : local_index + 1] for mask in target_masks
                    ]
                    sample_loss = spatial_ijepa_per_sample_loss(
                        core,
                        sample,
                        sample_context_masks,
                        sample_target_masks,
                    ).mean()
                    loss_gradients = torch.autograd.grad(
                        sample_loss,
                        parameters,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    batch_scores[local_index] = float(
                        (-_dot_gradients(loss_gradients, richness_gradients))
                        .detach()
                        .cpu()
                        .item()
                    )
            scores[indices.detach().cpu()] = batch_scores
    finally:
        core.context_encoder.train(context_was_training)
        core.predictor.train(predictor_was_training)
        core.target_encoder.train(target_was_training)

    metadata = {
        **richness_metadata,
        "ras/grad_richness_norm": float(grad_norm.detach().cpu().item()),
        "ras/positive_fraction": float((scores > 0).to(torch.float64).mean().item()),
        "ras/negative_fraction": float((scores < 0).to(torch.float64).mean().item()),
        "ras/score_granularity_batch": float(score_granularity == "batch"),
        "ras/score_groups": float(math.ceil(train_images.shape[0] / batch_size)),
    }
    return scores, metadata


def score_frames_by_coordinate_importance(
    core: SpatialIJEPACore,
    train_images: torch.Tensor,
    *,
    ref_indices: torch.Tensor,
    state: SpatialWeightingState,
    grid: int,
    mask_config: MaskConfig,
    batch_size: int,
    seed: int,
    device: torch.device,
    coordinate_importance: CoordinateImportanceMethod,
    coordinate_ema_beta: float,
    coordinate_delta: float,
    transform_pair_images: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("score batch_size must be positive")
    context_was_training = core.context_encoder.training
    predictor_was_training = core.predictor.training
    target_was_training = core.target_encoder.training
    core.context_encoder.eval()
    core.predictor.eval()
    core.target_encoder.eval()

    parameters = _encoder_parameters(core)
    if not parameters:
        raise ValueError("coordinate weighting requires trainable context encoder parameters")

    scores = torch.empty(train_images.shape[0], dtype=torch.float64)
    mask_generator = torch.Generator().manual_seed(derive_seed(seed, "weighting-coord-masks"))
    try:
        ref_images = train_images[ref_indices.to(train_images.device)].to(device)
        ref_latents = encode_samples_pooled(core, ref_images)
        if coordinate_importance == "covariance":
            coord_weights, basis, coord_metadata = _coordinate_importance_from_covariance(
                ref_latents,
                delta=coordinate_delta,
            )
        elif coordinate_importance == "transformation":
            if transform_pair_images is None:
                raise ValueError("transformation coordinate importance requires paired images")
            source_images, target_images = transform_pair_images
            source_latents = encode_samples_pooled(core, source_images.to(device))
            target_latents = encode_samples_pooled(core, target_images.to(device))
            coord_weights, basis, coord_metadata = _coordinate_importance_from_transformation(
                source_latents,
                target_latents,
                delta=coordinate_delta,
            )
        elif coordinate_importance == "dynamics":
            coord_weights, basis, coord_metadata = _coordinate_importance_from_dynamics(
                ref_latents,
                state,
                ema_beta=coordinate_ema_beta,
                delta=coordinate_delta,
            )
        else:
            raise ValueError(f"unknown coordinate importance: {coordinate_importance!r}")

        projected_ref_mean = (ref_latents.to(torch.float64) @ basis.to(device)).mean(dim=0)
        coordinate_gradients: list[tuple[torch.Tensor | None, ...]] = []
        for coordinate in projected_ref_mean:
            coordinate_gradients.append(
                torch.autograd.grad(coordinate, parameters, retain_graph=True, allow_unused=True)
            )
        coord_weights = coord_weights.to(device=device, dtype=torch.float64)
        coord_grad_norm = torch.sqrt(
            sum(
                sum(
                    torch.zeros((), dtype=torch.float64, device=device)
                    if gradient is None
                    else gradient.detach().to(torch.float64).square().sum()
                    for gradient in gradients
                )
                for gradients in coordinate_gradients
            )
        )

        for indices in torch.arange(train_images.shape[0], device=train_images.device).split(
            batch_size
        ):
            batch = train_images[indices].to(device)
            context_masks, target_masks = sample_masks(
                grid,
                grid,
                mask_config,
                mask_generator,
                batch_size=batch.shape[0],
            )
            batch_scores = torch.empty(batch.shape[0], dtype=torch.float64)
            for local_index in range(batch.shape[0]):
                sample = batch[local_index : local_index + 1]
                sample_loss = torch.zeros((), device=device)
                sample_context_masks = [
                    mask[local_index : local_index + 1] for mask in context_masks
                ]
                sample_target_masks = [
                    mask[local_index : local_index + 1] for mask in target_masks
                ]
                sample_loss = spatial_ijepa_per_sample_loss(
                    core,
                    sample,
                    sample_context_masks,
                    sample_target_masks,
                ).mean()
                loss_gradients = torch.autograd.grad(
                    sample_loss,
                    parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
                impacts = torch.empty(coord_weights.shape[0], dtype=torch.float64, device=device)
                for coordinate_index, gradients in enumerate(coordinate_gradients):
                    impacts[coordinate_index] = -_dot_gradients(
                        loss_gradients,
                        tuple(
                            torch.zeros_like(parameter) if gradient is None else gradient.detach()
                            for gradient, parameter in zip(gradients, parameters, strict=True)
                        ),
                    ).to(torch.float64)
                batch_scores[local_index] = float(
                    (coord_weights * impacts.abs()).sum().detach().cpu().item()
                )
            scores[indices.detach().cpu()] = batch_scores
    finally:
        core.context_encoder.train(context_was_training)
        core.predictor.train(predictor_was_training)
        core.target_encoder.train(target_was_training)

    metadata = {
        **coord_metadata,
        "coord/grad_coordinate_norm": float(coord_grad_norm.detach().cpu().item()),
        "coord/score_positive_fraction": float((scores > 0).to(torch.float64).mean().item()),
    }
    return scores, metadata, coord_weights.detach().cpu(), ref_latents.detach().cpu()
