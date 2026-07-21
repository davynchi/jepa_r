"""Frame-weighting curriculum helpers for spatial I-JEPA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from jepa.configs.base import derive_seed
from jepa.training.images.ijepa_spatial import (
    MaskConfig,
    SpatialIJEPACore,
    sample_masks,
    spatial_ijepa_per_sample_loss,
)

SpatialWeightingMethod = Literal["uniform", "loss", "ras"]
SpatialRichnessFunctional = Literal["logdet", "rbar", "pr"]


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


@dataclass(frozen=True, slots=True)
class SpatialWeightingState:
    memory: torch.Tensor
    probabilities: torch.Tensor
    last_scores: torch.Tensor


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
    if config.method not in {"uniform", "loss", "ras"}:
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
    return SpatialWeightingState(memory=memory, probabilities=probabilities, last_scores=scores64)


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


def _latent_covariance_from_patches(
    core: SpatialIJEPACore,
    patches: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = core.context_encoder(patches).mean(dim=1).to(torch.float64)
    centered = latents - latents.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(latents.shape[0] - 1, 1)
    return (covariance + covariance.T) / 2, latents


def richness_from_patches(
    core: SpatialIJEPACore,
    patches: torch.Tensor,
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
    covariance, _ = _latent_covariance_from_patches(core, patches)
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
    train_patches: torch.Tensor,
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

    scores = torch.empty(train_patches.shape[0], dtype=torch.float64)
    mask_generator = torch.Generator().manual_seed(derive_seed(seed, "weighting-loss-masks"))
    try:
        with torch.no_grad():
            for indices in torch.arange(train_patches.shape[0]).split(batch_size):
                batch = train_patches[indices].to(device)
                context_masks, target_masks = sample_masks(grid, grid, mask_config, mask_generator)
                batch_losses = torch.zeros(batch.shape[0], device=device)
                for context_mask in context_masks:
                    for target_mask in target_masks:
                        batch_losses = batch_losses + spatial_ijepa_per_sample_loss(
                            core, batch, context_mask.to(device), target_mask.to(device)
                        )
                batch_losses = batch_losses / (len(context_masks) * len(target_masks))
                scores[indices] = batch_losses.detach().cpu().to(torch.float64)
    finally:
        core.context_encoder.train(context_was_training)
        core.predictor.train(predictor_was_training)
        core.target_encoder.train(target_was_training)
    return scores


def score_frames_by_ras(
    core: SpatialIJEPACore,
    train_patches: torch.Tensor,
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
) -> tuple[torch.Tensor, dict[str, float]]:
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
        raise ValueError("RAS requires trainable context encoder parameters")

    scores = torch.empty(train_patches.shape[0], dtype=torch.float64)
    mask_generator = torch.Generator().manual_seed(derive_seed(seed, "weighting-ras-masks"))
    try:
        ref_patches = train_patches[ref_indices].to(device)
        richness, richness_metadata = richness_from_patches(
            core,
            ref_patches,
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

        for indices in torch.arange(train_patches.shape[0]).split(batch_size):
            batch = train_patches[indices].to(device)
            context_masks, target_masks = sample_masks(grid, grid, mask_config, mask_generator)
            batch_scores = torch.empty(batch.shape[0], dtype=torch.float64)
            for local_index in range(batch.shape[0]):
                sample = batch[local_index : local_index + 1]
                sample_loss = torch.zeros((), device=device)
                for context_mask in context_masks:
                    for target_mask in target_masks:
                        sample_loss = (
                            sample_loss
                            + spatial_ijepa_per_sample_loss(
                                core, sample, context_mask.to(device), target_mask.to(device)
                            ).mean()
                        )
                sample_loss = sample_loss / (len(context_masks) * len(target_masks))
                loss_gradients = torch.autograd.grad(
                    sample_loss,
                    parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
                batch_scores[local_index] = float(
                    (-_dot_gradients(loss_gradients, richness_gradients)).detach().cpu().item()
                )
            scores[indices] = batch_scores
    finally:
        core.context_encoder.train(context_was_training)
        core.predictor.train(predictor_was_training)
        core.target_encoder.train(target_was_training)

    metadata = {
        **richness_metadata,
        "ras/grad_richness_norm": float(grad_norm.detach().cpu().item()),
        "ras/positive_fraction": float((scores > 0).to(torch.float64).mean().item()),
        "ras/negative_fraction": float((scores < 0).to(torch.float64).mean().item()),
    }
    return scores, metadata
