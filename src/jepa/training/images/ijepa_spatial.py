"""I-JEPA-style *spatial* masked prediction, with a CNN encoder instead of a ViT.

Motivation: the temporal (V-JEPA-style) framing does not obviously fit Shapes3D
-- its six generative factors (floor/wall/object hue, scale, shape,
orientation) are all *global* scene properties, and the object never moves, so
a patch at a fixed grid position does not track any persistent local content
across frames the way it would in real video. I-JEPA's framing needs no
temporal structure at all: mask blocks *within a single frame* and predict
their representations from the visible remainder.

Adapted from https://github.com/facebookresearch/ijepa (CC BY-NC 4.0, see
third_party/ijepa/). Kept from upstream: the multi-block mask sampling, the
target-side ``F.layer_norm`` over the feature dim, and ``smooth_l1_loss``.
Changed: a CNN encodes each patch independently and the visible patches are
mean-pooled into a context summary (a ViT would instead mix them with
self-attention), and the predictor is an MLP conditioned on a learned
positional embedding rather than a transformer over mask tokens.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from jepa.models.encoders import Architecture, build_model_pair
from jepa.models.patches import patchify
from jepa.training.core import (
    SCHEMA_VERSION,
    OptimizationPolicy,
    _atomic_torch_save,
    resolve_policy,
)


@dataclass(frozen=True, slots=True)
class MaskConfig:
    """Upstream defaults (configs/in1k_vith14_ep300.yaml) unless noted."""

    enc_mask_scale: tuple[float, float] = (0.85, 1.0)
    pred_mask_scale: tuple[float, float] = (0.15, 0.2)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    num_enc_masks: int = 1
    num_pred_masks: int = 4
    min_keep: int = 4
    allow_overlap: bool = False


def _sample_block_size(
    grid_h: int,
    grid_w: int,
    scale: tuple[float, float],
    aspect_ratio: tuple[float, float],
    generator: torch.Generator,
) -> tuple[int, int]:
    """Port of MaskCollator._sample_block_size (third_party/ijepa)."""
    rand = torch.rand(1, generator=generator).item()
    min_s, max_s = scale
    mask_scale = min_s + rand * (max_s - min_s)
    max_keep = int(grid_h * grid_w * mask_scale)
    min_ar, max_ar = aspect_ratio
    ratio = min_ar + rand * (max_ar - min_ar)
    h = int(round(math.sqrt(max_keep * ratio)))
    w = int(round(math.sqrt(max_keep / ratio)))
    h = min(max(h, 1), grid_h)
    w = min(max(w, 1), grid_w)
    return h, w


def _sample_block_mask(
    grid_h: int,
    grid_w: int,
    block_size: tuple[int, int],
    generator: torch.Generator,
    acceptable: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of MaskCollator._sample_block_mask. Returns (kept patch indices,
    complement grid). ``acceptable`` is a [grid_h, grid_w] 0/1 grid the block
    is intersected with (used to keep the context block off the target blocks)."""
    h, w = block_size
    min_keep = 1
    for _ in range(50):
        top = int(torch.randint(0, max(grid_h - h, 1), (1,), generator=generator).item())
        left = int(torch.randint(0, max(grid_w - w, 1), (1,), generator=generator).item())
        grid = torch.zeros((grid_h, grid_w), dtype=torch.int32)
        grid[top : top + h, left : left + w] = 1
        if acceptable is not None:
            grid = grid * acceptable
        indices = torch.nonzero(grid.flatten(), as_tuple=False).squeeze(-1)
        if indices.numel() > min_keep:
            complement = torch.ones((grid_h, grid_w), dtype=torch.int32)
            complement[top : top + h, left : left + w] = 0
            return indices, complement
    # Fall back to the unconstrained block if the constrained sampler kept
    # failing (upstream logs a warning and relaxes the constraint instead).
    grid = torch.zeros((grid_h, grid_w), dtype=torch.int32)
    grid[top : top + h, left : left + w] = 1
    indices = torch.nonzero(grid.flatten(), as_tuple=False).squeeze(-1)
    complement = torch.ones((grid_h, grid_w), dtype=torch.int32)
    complement[top : top + h, left : left + w] = 0
    return indices, complement


def sample_masks(
    grid_h: int, grid_w: int, config: MaskConfig, generator: torch.Generator
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """One (context masks, target masks) draw, shared across the batch.

    Upstream samples per-image masks inside a collate_fn and truncates them to
    a common length so they can be stacked; we hold our whole split in memory
    as one tensor and draw a single mask set per batch instead, which keeps
    every image's mask the same length by construction.
    """
    pred_size = _sample_block_size(
        grid_h, grid_w, config.pred_mask_scale, config.aspect_ratio, generator
    )
    enc_size = _sample_block_size(grid_h, grid_w, config.enc_mask_scale, (1.0, 1.0), generator)

    target_masks: list[torch.Tensor] = []
    complements: list[torch.Tensor] = []
    for _ in range(config.num_pred_masks):
        mask, complement = _sample_block_mask(grid_h, grid_w, pred_size, generator)
        target_masks.append(mask)
        complements.append(complement)

    acceptable = None
    if not config.allow_overlap:
        acceptable = torch.ones((grid_h, grid_w), dtype=torch.int32)
        for complement in complements:
            acceptable = acceptable * complement

    context_masks: list[torch.Tensor] = []
    for _ in range(config.num_enc_masks):
        mask, _ = _sample_block_mask(grid_h, grid_w, enc_size, generator, acceptable=acceptable)
        context_masks.append(mask)
    return context_masks, target_masks


class SpatialPositionalPredictor(nn.Module):
    """Predicts the latents of the patches named by a target mask, from the
    pooled context summary plus a learned embedding of each target position."""

    def __init__(
        self,
        num_patches: int,
        context_dim: int,
        patch_latent_dim: int,
        *,
        position_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(num_patches, position_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_latent_dim),
        )

    def forward(self, context_summary: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """``context_summary``: [B, context_dim]; ``target_indices``: [k].
        Returns [B, k, patch_latent_dim]."""
        k = target_indices.shape[0]
        position = self.position_embedding(target_indices)
        context_expanded = context_summary.unsqueeze(1).expand(-1, k, -1)
        position_expanded = position.unsqueeze(0).expand(context_summary.shape[0], -1, -1)
        return self.mlp(torch.cat([context_expanded, position_expanded], dim=-1))


@dataclass(slots=True)
class SpatialIJEPACore:
    context_encoder: nn.Module
    predictor: SpatialPositionalPredictor
    target_encoder: nn.Module
    policy: OptimizationPolicy


def build_spatial_ijepa_core(
    architecture: Architecture,
    *,
    patch_dim: int,
    patch_latent_dim: int,
    num_patches: int,
    stop_gradient: bool = True,
    ema_enabled: bool = True,
    hidden_dim: int = 64,
    position_dim: int = 32,
    predictor_hidden_dim: int = 128,
) -> SpatialIJEPACore:
    encoder, _ = build_model_pair(
        architecture, input_dim=patch_dim, latent_dim=patch_latent_dim, hidden_dim=hidden_dim
    )
    predictor = SpatialPositionalPredictor(
        num_patches,
        context_dim=patch_latent_dim,
        patch_latent_dim=patch_latent_dim,
        position_dim=position_dim,
        hidden_dim=predictor_hidden_dim,
    )
    policy = resolve_policy(stop_gradient=stop_gradient, ema_enabled=ema_enabled)
    target_encoder = deepcopy(encoder) if policy.separate_target else encoder
    if policy.separate_target and not policy.optimize_target:
        target_encoder.requires_grad_(False)
    return SpatialIJEPACore(encoder, predictor, target_encoder, policy)


def spatial_ijepa_per_sample_loss(
    core: SpatialIJEPACore,
    patches: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one spatial I-JEPA loss value per frame in the batch.

    ``patches``: [B, num_patches, patch_dim]; masks are 1-D index tensors.

    Target-side ``layer_norm`` over the feature dim and ``smooth_l1_loss`` are
    taken from upstream's src/train.py (forward_target / loss_fn).
    """
    context_patches = patches[:, context_mask, :]
    target_patches = patches[:, target_mask, :]

    with torch.no_grad() if not core.policy.optimize_target else torch.enable_grad():
        target_latents = core.target_encoder(target_patches)
        target_latents = F.layer_norm(target_latents, (target_latents.size(-1),))
    if core.policy.stop_gradient:
        target_latents = target_latents.detach()

    context_latents = core.context_encoder(context_patches)
    context_summary = context_latents.mean(dim=1)
    predicted = core.predictor(context_summary, target_mask)
    per_element = F.smooth_l1_loss(predicted, target_latents, reduction="none")
    return per_element.mean(dim=(1, 2))


def spatial_ijepa_loss(
    core: SpatialIJEPACore,
    patches: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean spatial I-JEPA loss over the batch."""
    return spatial_ijepa_per_sample_loss(core, patches, context_mask, target_mask).mean()


@torch.no_grad()
def encode_frames_pooled(
    core: SpatialIJEPACore, frames: torch.Tensor, *, patch_size: int
) -> torch.Tensor:
    """Frame-level representation for post-hoc analysis: encode every patch
    with the frozen context encoder and mean-pool. ``frames``: [..., C, H, W]."""
    core.context_encoder.eval()
    device = next(core.context_encoder.parameters()).device
    patches = patchify(frames, patch_size).to(device)
    return core.context_encoder(patches).mean(dim=-2)


def save_spatial_checkpoint(
    path: str | Path,
    core: SpatialIJEPACore,
    *,
    epoch: int,
    global_step: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "context_encoder": core.context_encoder.state_dict(),
        "predictor": core.predictor.state_dict(),
        "target_encoder": core.target_encoder.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    if extra_state is not None:
        payload["extra_state"] = dict(extra_state)
    _atomic_torch_save(
        Path(path),
        payload,
    )


def load_spatial_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint has an incompatible schema")
    return checkpoint
