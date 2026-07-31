"""Residual-Q17 auxiliary objective for transformation geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

ResidualQ17Mode = Literal["matched", "shuffled"]
ResidualQ17Transform = Literal["flip", "blur", "color"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class CenteredResidualBatch:
    source: torch.Tensor
    residuals: dict[str, torch.Tensor]
    fit_indices: torch.Tensor
    holdout_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class GradientAlignment:
    jepa_norm: float
    auxiliary_norm: float
    cosine: float
    loss_weight: float
    actual_ratio: float


def _channel_constants(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return mean, std


def _gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("blur sigma must be positive")
    radius = max(1, int(round(3 * sigma)))
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=images.device,
        dtype=images.dtype,
    )
    kernel = torch.exp(-coordinates.square() / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    channels = images.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    blurred = F.conv2d(
        F.pad(images, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=channels,
    )
    return F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=channels,
    )


def _fixed_color_transform(images: torch.Tensor, strength: float) -> torch.Tensor:
    if strength < 0:
        raise ValueError("color strength must be non-negative")
    factor = 1.0 + strength
    result = images * factor
    spatial_mean = result.mean(dim=(1, 2, 3), keepdim=True)
    result = (result - spatial_mean) * factor + spatial_mean
    grayscale = 0.2989 * result[:, 0:1] + 0.5870 * result[:, 1:2] + 0.1140 * result[:, 2:3]
    return ((result - grayscale) * factor + grayscale).clamp(0, 1)


def apply_fixed_transform(
    normalized_images: torch.Tensor,
    transform: ResidualQ17Transform,
    *,
    blur_sigma: float,
    color_strength: float,
) -> torch.Tensor:
    """Apply a deterministic transform while preserving I-JEPA normalization."""
    if normalized_images.ndim != 4 or normalized_images.shape[1] != 3:
        raise ValueError("images must have shape [batch, 3, height, width]")
    mean, std = _channel_constants(normalized_images)
    pixels = (normalized_images * std + mean).clamp(0, 1)
    if transform == "flip":
        transformed = pixels.flip(-1)
    elif transform == "blur":
        transformed = _gaussian_blur(pixels, blur_sigma)
    elif transform == "color":
        transformed = _fixed_color_transform(pixels, color_strength)
    else:  # pragma: no cover - guarded by CLI and constructor validation.
        raise ValueError(f"unsupported residual-Q17 transform: {transform!r}")
    return (transformed - mean) / std


def _pooled_normalized_tokens(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    tokens = encoder(images)
    if tokens.ndim != 3:
        raise RuntimeError("encoder must return [batch, tokens, dimensions]")
    return F.layer_norm(tokens, (tokens.shape[-1],)).mean(dim=1).float()


class ResidualQ17Regularizer(nn.Module):
    """Cross-fitted full linear prediction of centered transformation residuals."""

    def __init__(
        self,
        embed_dim: int,
        transforms: tuple[ResidualQ17Transform, ...],
        *,
        statistics_decay: float = 0.99,
        epsilon: float = 1.0e-6,
        blur_sigma: float = 1.0,
        color_strength: float = 0.2,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if not transforms or len(set(transforms)) != len(transforms):
            raise ValueError("transforms must be non-empty and unique")
        if any(name not in {"flip", "blur", "color"} for name in transforms):
            raise ValueError("residual-Q17 supports flip, blur, and color")
        if not 0 <= statistics_decay < 1:
            raise ValueError("statistics_decay must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if blur_sigma <= 0 or color_strength < 0:
            raise ValueError("transform strengths are invalid")
        self.embed_dim = embed_dim
        self.transforms = transforms
        self.statistics_decay = statistics_decay
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma
        self.color_strength = color_strength
        self.operators = nn.ModuleDict(
            {name: nn.Linear(embed_dim, embed_dim, bias=False) for name in transforms}
        )
        for operator in self.operators.values():
            nn.init.zeros_(operator.weight)
        self.register_buffer("source_mean", torch.zeros(embed_dim, dtype=torch.float32))
        self.register_buffer(
            "residual_means",
            torch.zeros(len(transforms), embed_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "residual_energies",
            torch.ones(len(transforms), dtype=torch.float32),
        )
        self.register_buffer("statistics_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def target_residuals(
        self,
        target_encoder: nn.Module,
        normalized_images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        source = _pooled_normalized_tokens(target_encoder, normalized_images)
        residuals: dict[str, torch.Tensor] = {}
        for name in self.transforms:
            transformed = apply_fixed_transform(
                normalized_images,
                name,
                blur_sigma=self.blur_sigma,
                color_strength=self.color_strength,
            )
            residuals[name] = _pooled_normalized_tokens(target_encoder, transformed) - source
        return residuals

    def encode_batch(
        self,
        online_encoder: nn.Module,
        target_encoder: nn.Module,
        normalized_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source = _pooled_normalized_tokens(online_encoder, normalized_images)
        residuals = self.target_residuals(target_encoder, normalized_images)
        return source, residuals

    @torch.no_grad()
    def update_statistics(
        self,
        source: torch.Tensor,
        residuals: dict[str, torch.Tensor],
    ) -> None:
        self._validate_representations(source, residuals)
        source_mean = source.detach().float().mean(dim=0)
        residual_means = torch.stack(
            [residuals[name].detach().float().mean(dim=0) for name in self.transforms]
        )
        centered = torch.stack(
            [
                (residuals[name].detach().float() - residual_means[index])
                .square()
                .sum(dim=-1)
                .mean()
                for index, name in enumerate(self.transforms)
            ]
        )
        if int(self.statistics_updates.item()) == 0:
            self.source_mean.copy_(source_mean)
            self.residual_means.copy_(residual_means)
            self.residual_energies.copy_(centered.clamp_min(self.epsilon))
        else:
            beta = self.statistics_decay
            self.source_mean.mul_(beta).add_(source_mean, alpha=1.0 - beta)
            self.residual_means.mul_(beta).add_(residual_means, alpha=1.0 - beta)
            self.residual_energies.mul_(beta).add_(centered, alpha=1.0 - beta)
        self.statistics_updates.add_(1)

    def center_batch(
        self,
        source: torch.Tensor,
        residuals: dict[str, torch.Tensor],
    ) -> CenteredResidualBatch:
        self._validate_representations(source, residuals)
        if source.shape[0] < 4:
            raise ValueError("residual-Q17 requires at least four images")
        indices = torch.arange(source.shape[0], device=source.device)
        fit_indices = indices[::2]
        holdout_indices = indices[1::2]
        return CenteredResidualBatch(
            source=source.float() - self.source_mean.detach(),
            residuals={
                name: residuals[name].float() - self.residual_means[index].detach()
                for index, name in enumerate(self.transforms)
            },
            fit_indices=fit_indices,
            holdout_indices=holdout_indices,
        )

    def operator_loss(
        self,
        batch: CenteredResidualBatch,
        *,
        mode: ResidualQ17Mode,
    ) -> torch.Tensor:
        return self._loss(
            batch,
            batch.fit_indices,
            mode=mode,
            detach_operators=False,
            detach_inputs=True,
        )[0]

    def encoder_loss(
        self,
        batch: CenteredResidualBatch,
        *,
        mode: ResidualQ17Mode,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self._loss(
            batch,
            batch.holdout_indices,
            mode=mode,
            detach_operators=True,
            detach_inputs=False,
        )

    def _loss(
        self,
        batch: CenteredResidualBatch,
        indices: torch.Tensor,
        *,
        mode: ResidualQ17Mode,
        detach_operators: bool,
        detach_inputs: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if mode not in {"matched", "shuffled"}:
            raise ValueError(f"unsupported residual-Q17 mode: {mode!r}")
        source = batch.source[indices]
        if detach_inputs:
            source = source.detach()
        losses: list[torch.Tensor] = []
        metadata: dict[str, float] = {}
        for transform_index, name in enumerate(self.transforms):
            target = batch.residuals[name][indices]
            if detach_inputs:
                target = target.detach()
            if mode == "shuffled":
                target = target.roll(1, dims=0)
            weight = self.operators[name].weight
            if detach_operators:
                weight = weight.detach()
            prediction = F.linear(source, weight)
            prediction_mse = (prediction - target).square().sum(dim=-1).mean()
            identity_mse = target.square().sum(dim=-1).mean()
            scale = self.residual_energies[transform_index].detach().clamp_min(self.epsilon)
            normalized_loss = prediction_mse / scale
            losses.append(normalized_loss)
            prefix = f"residual_q17/{name}"
            metadata[f"{prefix}/normalized_loss"] = float(normalized_loss.detach().item())
            metadata[f"{prefix}/prediction_mse"] = float(prediction_mse.detach().item())
            metadata[f"{prefix}/identity_mse"] = float(identity_mse.detach().item())
            metadata[f"{prefix}/gain"] = float(
                (1.0 - prediction_mse / identity_mse.clamp_min(self.epsilon)).detach().item()
            )
            metadata[f"{prefix}/ema_energy"] = float(scale.item())
        loss = torch.stack(losses).mean()
        metadata["residual_q17/loss"] = float(loss.detach().item())
        return loss, metadata

    def _validate_representations(
        self,
        source: torch.Tensor,
        residuals: dict[str, torch.Tensor],
    ) -> None:
        expected = (source.shape[0], self.embed_dim)
        if source.ndim != 2 or source.shape[1] != self.embed_dim:
            raise ValueError(f"source must have shape [batch, {self.embed_dim}]")
        if set(residuals) != set(self.transforms):
            raise ValueError("residual keys do not match configured transforms")
        if any(value.shape != expected for value in residuals.values()):
            raise ValueError("all residuals must match the source shape")


def gradient_alignment_and_weight(
    jepa_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
    *,
    target_ratio: float,
    epsilon: float = 1.0e-12,
) -> GradientAlignment:
    """Measure gradient alignment and choose a detached auxiliary loss weight."""
    if target_ratio < 0:
        raise ValueError("target_ratio must be non-negative")
    jepa_gradients = torch.autograd.grad(
        jepa_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    device = jepa_loss.device
    jepa_squared = torch.zeros((), device=device, dtype=torch.float64)
    auxiliary_squared = torch.zeros((), device=device, dtype=torch.float64)
    dot = torch.zeros((), device=device, dtype=torch.float64)
    for jepa_gradient, auxiliary_gradient in zip(jepa_gradients, auxiliary_gradients, strict=True):
        if jepa_gradient is not None:
            jepa_squared += jepa_gradient.detach().double().square().sum()
        if auxiliary_gradient is not None:
            auxiliary_squared += auxiliary_gradient.detach().double().square().sum()
        if jepa_gradient is not None and auxiliary_gradient is not None:
            dot += (jepa_gradient.detach().double() * auxiliary_gradient.detach().double()).sum()
    jepa_norm = jepa_squared.sqrt()
    auxiliary_norm = auxiliary_squared.sqrt()
    denominator = jepa_norm * auxiliary_norm
    cosine = torch.where(
        denominator > epsilon,
        dot / denominator,
        torch.zeros_like(dot),
    )
    weight = torch.where(
        auxiliary_norm > epsilon,
        target_ratio * jepa_norm / auxiliary_norm,
        torch.zeros_like(auxiliary_norm),
    )
    actual_ratio = torch.where(
        jepa_norm > epsilon,
        weight * auxiliary_norm / jepa_norm,
        torch.zeros_like(jepa_norm),
    )
    return GradientAlignment(
        jepa_norm=float(jepa_norm.item()),
        auxiliary_norm=float(auxiliary_norm.item()),
        cosine=float(cosine.item()),
        loss_weight=float(weight.item()),
        actual_ratio=float(actual_ratio.item()),
    )


__all__ = [
    "CenteredResidualBatch",
    "GradientAlignment",
    "ResidualQ17Mode",
    "ResidualQ17Regularizer",
    "ResidualQ17Transform",
    "apply_fixed_transform",
    "gradient_alignment_and_weight",
]
