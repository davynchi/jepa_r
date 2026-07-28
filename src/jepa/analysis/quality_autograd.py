"""Exact, non-mutating autograd diagnostics for spatial I-JEPA."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from jepa.analysis.quality_metrics import q15_virtual_interference
from jepa.analysis.spatial_decomposition import fit_label_free_partition
from jepa.training.images.ijepa_spatial import SpatialIJEPACore


class VirtualStateMutationError(RuntimeError):
    pass


def _fingerprint_module(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    values = {
        f"parameter:{name}": value.detach().cpu().clone()
        for name, value in module.named_parameters()
    }
    values.update(
        {f"buffer:{name}": value.detach().cpu().clone() for name, value in module.named_buffers()}
    )
    return values


def _core_fingerprint(core: SpatialIJEPACore) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "context": _fingerprint_module(core.context_encoder),
        "predictor": _fingerprint_module(core.predictor),
        "target": _fingerprint_module(core.target_encoder),
    }


def _assert_unchanged(
    before: Mapping[str, Mapping[str, torch.Tensor]], core: SpatialIJEPACore
) -> None:
    after = _core_fingerprint(core)
    if before.keys() != after.keys():
        raise VirtualStateMutationError("virtual_state_mutation")
    for module_name, values in before.items():
        if values.keys() != after[module_name].keys() or any(
            not torch.equal(value, after[module_name][name]) for name, value in values.items()
        ):
            raise VirtualStateMutationError("virtual_state_mutation")


@dataclass(frozen=True, slots=True)
class VirtualContextStep:
    parameters: Mapping[str, torch.Tensor]
    buffers: Mapping[str, torch.Tensor]
    loss: float


def virtual_context_step(
    core: SpatialIJEPACore,
    patches: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    eta: float = 1e-3,
) -> VirtualContextStep:
    """Compute one strict functional context-encoder SGD step."""
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and non-negative")
    before = _core_fingerprint(core)
    parameters = dict(core.context_encoder.named_parameters())
    buffers = dict(core.context_encoder.named_buffers())
    context_patches = patches[:, context_mask, :]
    target_patches = patches[:, target_mask, :]
    with torch.no_grad():
        target = core.target_encoder(target_patches)
        target = F.layer_norm(target, (target.shape[-1],))
    context = torch.func.functional_call(
        core.context_encoder,
        cast(tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]], (parameters, buffers)),
        (context_patches,),
        strict=True,
    )
    prediction = core.predictor(context.mean(dim=1), target_mask)
    loss = F.smooth_l1_loss(prediction, target.detach())
    gradients = torch.autograd.grad(
        loss, tuple(parameters.values()), create_graph=False, retain_graph=False
    )
    updated = {
        name: (parameter - eta * gradient).detach()
        for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True)
    }
    detached_buffers = {name: value.detach() for name, value in buffers.items()}
    _assert_unchanged(before, core)
    return VirtualContextStep(updated, detached_buffers, float(loss.detach()))


def _functional_encode(
    core: SpatialIJEPACore,
    step: VirtualContextStep,
    patches: torch.Tensor,
    *,
    microbatch_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in patches.split(microbatch_size):
            encoded = torch.func.functional_call(
                core.context_encoder,
                (dict(step.parameters), dict(step.buffers)),
                (batch,),
                strict=True,
            )
            rows.append(encoded.mean(dim=1).detach().cpu())
    return torch.cat(rows)


@dataclass(frozen=True, slots=True)
class FusedVirtualMetrics:
    q8: float
    q15: float
    per_sample_deltas: torch.Tensor
    gradient_calls: int


def fused_q8_q15(
    core: SpatialIJEPACore,
    conditioning_patches: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
    refit_view_a_patches: torch.Tensor,
    refit_view_b_patches: torch.Tensor,
    replay_patches: torch.Tensor,
    *,
    checkpoint_id: str,
    eta: float = 1e-3,
    microbatch_size: int = 32,
) -> FusedVirtualMetrics:
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    before = _core_fingerprint(core)
    with torch.no_grad():
        baseline_a = core.context_encoder(refit_view_a_patches).mean(dim=1).cpu()
        baseline_b = core.context_encoder(refit_view_b_patches).mean(dim=1).cpu()
        baseline_replay = core.context_encoder(replay_patches).mean(dim=1).cpu()
    baseline_partition = fit_label_free_partition(
        baseline_a, baseline_b, checkpoint_id=f"{checkpoint_id}:baseline"
    )
    delta_rows: list[torch.Tensor] = []
    interference: list[float] = []
    for sample in conditioning_patches:
        step = virtual_context_step(core, sample.unsqueeze(0), context_mask, target_mask, eta=eta)
        if eta == 0:
            # Preserve the exact identity invariant. Re-encoding in different
            # microbatch shapes can otherwise introduce harmless BLAS roundoff.
            updated_a = baseline_a
            updated_b = baseline_b
        else:
            updated_a = _functional_encode(
                core, step, refit_view_a_patches, microbatch_size=microbatch_size
            )
            updated_b = _functional_encode(
                core, step, refit_view_b_patches, microbatch_size=microbatch_size
            )
        updated_partition = fit_label_free_partition(
            updated_a, updated_b, checkpoint_id=f"{checkpoint_id}:virtual"
        )
        deltas = torch.tensor(
            [
                float(
                    torch.linalg.matrix_norm(updated.projector - baseline.projector)
                    / math.sqrt(2 * baseline.rank)
                )
                for baseline, updated in zip(
                    baseline_partition.bands, updated_partition.bands, strict=True
                )
            ],
            dtype=torch.float64,
        )
        delta_rows.append(deltas)
        updated_replay = (
            baseline_replay
            if eta == 0
            else _functional_encode(core, step, replay_patches, microbatch_size=microbatch_size)
        )
        value = q15_virtual_interference(baseline_replay, updated_replay)
        if value.value is None:
            raise ValueError(value.reason)
        interference.append(value.value)
    _assert_unchanged(before, core)
    all_deltas = torch.stack(delta_rows)
    return FusedVirtualMetrics(
        q8=float(all_deltas.sum(dim=1).mean()),
        q15=sum(interference) / len(interference),
        per_sample_deltas=all_deltas,
        gradient_calls=len(conditioning_patches),
    )


def predictor_jacobian(
    core: SpatialIJEPACore, context_summary: torch.Tensor, target_index: int
) -> torch.Tensor:
    index = torch.tensor([target_index], dtype=torch.long, device=context_summary.device)

    def predict(summary: torch.Tensor) -> torch.Tensor:
        return core.predictor(summary.unsqueeze(0), index)[0, 0]

    return torch.func.jacrev(predict)(context_summary)


def context_latent_gradients(
    core: SpatialIJEPACore,
    patches: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Return dL/d(context patch latents), without changing model state."""
    before = _core_fingerprint(core)
    context = core.context_encoder(patches[:, context_mask, :])
    with torch.no_grad():
        target = core.target_encoder(patches[:, target_mask, :])
        target = F.layer_norm(target, (target.shape[-1],))
    prediction = core.predictor(context.mean(dim=1), target_mask)
    loss = F.smooth_l1_loss(prediction, target.detach())
    (gradients,) = torch.autograd.grad(loss, (context,), create_graph=False)
    _assert_unchanged(before, core)
    return gradients.detach()


__all__ = [
    "FusedVirtualMetrics",
    "VirtualContextStep",
    "VirtualStateMutationError",
    "context_latent_gradients",
    "fused_q8_q15",
    "predictor_jacobian",
    "virtual_context_step",
]
