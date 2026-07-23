"""Online batch-RAS rewards computed from normal training gradients."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from jepa.training.images.ijepa_spatial import SpatialIJEPACore
from jepa.training.images.spatial_curriculum import (
    RASAlignment,
    SpatialRichnessFunctional,
    richness_from_images,
)


@dataclass(frozen=True, slots=True)
class RichnessGradientSnapshot:
    """A fixed richness direction aligned with context-encoder parameters."""

    gradients: tuple[torch.Tensor, ...]
    metadata: dict[str, float]
    step: int


def _trainable_encoder_parameters(
    core: SpatialIJEPACore,
) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter for parameter in core.context_encoder.parameters() if parameter.requires_grad
    )


def capture_richness_gradient(
    core: SpatialIJEPACore,
    reference_images: torch.Tensor,
    *,
    functional: SpatialRichnessFunctional,
    delta: float,
    trace_target: float,
    trace_beta: float,
    step: int,
) -> RichnessGradientSnapshot:
    """Evaluate richness and cache its parameter gradient without touching ``.grad``."""
    if step < 0:
        raise ValueError("step must be non-negative")
    parameters = _trainable_encoder_parameters(core)
    if not parameters:
        raise ValueError("online RAS requires trainable context encoder parameters")
    was_training = core.context_encoder.training
    core.context_encoder.eval()
    try:
        richness, metadata = richness_from_images(
            core,
            reference_images,
            functional=functional,
            delta=delta,
            trace_target=trace_target,
            trace_beta=trace_beta,
        )
        gradients = torch.autograd.grad(richness, parameters, retain_graph=False)
    finally:
        core.context_encoder.train(was_training)
    detached = tuple(gradient.detach() for gradient in gradients)
    gradient_norm_squared = torch.zeros((), device=detached[0].device, dtype=torch.float32)
    for gradient in detached:
        gradient_norm_squared = gradient_norm_squared + gradient.float().square().sum()
    gradient_norm = torch.sqrt(gradient_norm_squared)
    return RichnessGradientSnapshot(
        gradients=detached,
        metadata={
            **metadata,
            "ras/grad_richness_norm": float(gradient_norm.cpu().item()),
        },
        step=step,
    )


def batch_ras_from_parameter_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
    snapshot: RichnessGradientSnapshot,
    *,
    alignment: RASAlignment = "dot",
) -> torch.Tensor:
    """Return ``-<grad R, grad loss>`` from an already completed train backward."""
    if alignment not in {"dot", "cosine"}:
        raise ValueError(f"unknown RAS alignment: {alignment!r}")
    if len(parameters) != len(snapshot.gradients):
        raise ValueError("parameter list does not match richness gradient snapshot")
    dot = torch.zeros((), device=snapshot.gradients[0].device, dtype=torch.float32)
    loss_norm_squared = torch.zeros_like(dot)
    richness_norm_squared = torch.zeros_like(dot)
    for parameter, richness_gradient in zip(parameters, snapshot.gradients, strict=True):
        richness_norm_squared = richness_norm_squared + richness_gradient.float().square().sum()
        if parameter.grad is None:
            continue
        if parameter.grad.shape != richness_gradient.shape:
            raise ValueError("parameter gradient shape changed after richness capture")
        loss_gradient = parameter.grad.detach().float()
        dot = dot + torch.sum(loss_gradient * richness_gradient.float())
        loss_norm_squared = loss_norm_squared + loss_gradient.square().sum()
    score = -dot
    if alignment == "cosine":
        denominator = torch.sqrt(loss_norm_squared * richness_norm_squared)
        if denominator <= torch.finfo(denominator.dtype).eps:
            return torch.zeros_like(score)
        score = score / denominator
    return score
