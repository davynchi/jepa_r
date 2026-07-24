"""Spatial I-JEPA built on the upstream Meta I-JEPA model semantics.

The encoder, predictor, positional embeddings, target construction, and loss
follow https://github.com/facebookresearch/ijepa. Project-specific dataset
sampling and weighting live outside this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from jepa.models import ijepa as upstream_ijepa
from jepa.models.ijepa_utils import apply_masks, repeat_interleave_batch, trunc_normal_
from jepa.training.core import (
    SCHEMA_VERSION,
    OptimizationPolicy,
    _atomic_torch_save,
    resolve_policy,
)

IJEPAModelName = Literal["vit_tiny", "vit_small", "vit_base", "vit_large"]


@dataclass(frozen=True, slots=True)
class MaskConfig:
    """Multi-block mask settings from upstream I-JEPA."""

    enc_mask_scale: tuple[float, float] = (0.85, 1.0)
    pred_mask_scale: tuple[float, float] = (0.15, 0.2)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    num_enc_masks: int = 1
    num_pred_masks: int = 4
    # Upstream uses 10 on a 16x16 grid. Four is the corresponding safe
    # minimum for our 8x8 grid, where a 15% target block can contain 9 tokens.
    min_keep: int = 4
    allow_overlap: bool = False


def _sample_block_size(
    grid_h: int,
    grid_w: int,
    scale: tuple[float, float],
    aspect_ratio: tuple[float, float],
    generator: torch.Generator,
) -> tuple[int, int]:
    """Equivalent to upstream ``MaskCollator._sample_block_size``."""
    rand = torch.rand(1, generator=generator).item()
    min_s, max_s = scale
    mask_scale = min_s + rand * (max_s - min_s)
    max_keep = int(grid_h * grid_w * mask_scale)
    min_ar, max_ar = aspect_ratio
    ratio = min_ar + rand * (max_ar - min_ar)
    h = int(round(math.sqrt(max_keep * ratio)))
    w = int(round(math.sqrt(max_keep / ratio)))
    while h >= grid_h:
        h -= 1
    while w >= grid_w:
        w -= 1
    return max(h, 1), max(w, 1)


def _sample_block_mask(
    grid_h: int,
    grid_w: int,
    block_size: tuple[int, int],
    generator: torch.Generator,
    *,
    acceptable_regions: list[torch.Tensor] | None = None,
    min_keep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent to upstream ``MaskCollator._sample_block_mask``."""
    h, w = block_size
    tries = 0
    timeout = original_timeout = 20
    while True:
        top = int(torch.randint(0, grid_h - h, (1,), generator=generator).item())
        left = int(torch.randint(0, grid_w - w, (1,), generator=generator).item())
        mask = torch.zeros((grid_h, grid_w), dtype=torch.int32)
        mask[top : top + h, left : left + w] = 1
        if acceptable_regions is not None:
            for region in acceptable_regions[: max(len(acceptable_regions) - tries, 0)]:
                mask *= region
        indices = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(-1)
        if len(indices) > min_keep:
            complement = torch.ones((grid_h, grid_w), dtype=torch.int32)
            complement[top : top + h, left : left + w] = 0
            return indices, complement
        timeout -= 1
        if timeout == 0:
            tries += 1
            timeout = original_timeout


def sample_masks(
    grid_h: int,
    grid_w: int,
    config: MaskConfig,
    generator: torch.Generator,
    *,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Sample upstream multi-block masks with independent locations per image."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pred_size = _sample_block_size(
        grid_h, grid_w, config.pred_mask_scale, config.aspect_ratio, generator
    )
    enc_size = _sample_block_size(grid_h, grid_w, config.enc_mask_scale, (1.0, 1.0), generator)

    targets_by_image: list[list[torch.Tensor]] = []
    contexts_by_image: list[list[torch.Tensor]] = []
    min_pred = grid_h * grid_w
    min_enc = grid_h * grid_w
    for _ in range(batch_size):
        image_targets: list[torch.Tensor] = []
        complements: list[torch.Tensor] = []
        for _ in range(config.num_pred_masks):
            mask, complement = _sample_block_mask(
                grid_h,
                grid_w,
                pred_size,
                generator,
                min_keep=config.min_keep,
            )
            image_targets.append(mask)
            complements.append(complement)
            min_pred = min(min_pred, len(mask))
        targets_by_image.append(image_targets)

        acceptable = None if config.allow_overlap else complements
        image_contexts: list[torch.Tensor] = []
        for _ in range(config.num_enc_masks):
            mask, _ = _sample_block_mask(
                grid_h,
                grid_w,
                enc_size,
                generator,
                acceptable_regions=acceptable,
                min_keep=config.min_keep,
            )
            image_contexts.append(mask)
            min_enc = min(min_enc, len(mask))
        contexts_by_image.append(image_contexts)

    target_masks = [
        torch.stack([targets_by_image[b][m][:min_pred] for b in range(batch_size)])
        for m in range(config.num_pred_masks)
    ]
    context_masks = [
        torch.stack([contexts_by_image[b][m][:min_enc] for b in range(batch_size)])
        for m in range(config.num_enc_masks)
    ]
    return context_masks, target_masks


@dataclass(slots=True)
class SpatialIJEPACore:
    context_encoder: nn.Module
    predictor: nn.Module
    target_encoder: nn.Module
    policy: OptimizationPolicy
    model_name: IJEPAModelName
    image_size: int
    patch_size: int
    embed_dim: int


def build_spatial_ijepa_core(
    model_name: IJEPAModelName = "vit_tiny",
    *,
    image_size: int = 64,
    patch_size: int = 8,
    predictor_embed_dim: int = 192,
    predictor_depth: int = 6,
    stop_gradient: bool = True,
    ema_enabled: bool = True,
) -> SpatialIJEPACore:
    """Build the same encoder/predictor/target structure as upstream I-JEPA."""
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by patch_size")
    if model_name not in upstream_ijepa.VIT_EMBED_DIMS:
        raise ValueError(f"unknown I-JEPA model: {model_name!r}")
    encoder_factory = getattr(upstream_ijepa, model_name)
    encoder = encoder_factory(img_size=[image_size], patch_size=patch_size)
    predictor = upstream_ijepa.vit_predictor(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=encoder.num_heads,
    )

    # Upstream helper.init_model performs this second initialization pass after
    # constructing both modules.
    def init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    encoder.apply(init_weights)
    predictor.apply(init_weights)
    policy = resolve_policy(stop_gradient=stop_gradient, ema_enabled=ema_enabled)
    target_encoder = deepcopy(encoder) if policy.separate_target else encoder
    if policy.separate_target and not policy.optimize_target:
        target_encoder.requires_grad_(False)
    return SpatialIJEPACore(
        context_encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        policy=policy,
        model_name=model_name,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=encoder.embed_dim,
    )


def build_spatial_ijepa_core_from_metadata(metadata: Mapping[str, Any]) -> SpatialIJEPACore:
    if metadata.get("model_family") != "upstream_ijepa":
        raise ValueError(
            "checkpoint predates the upstream I-JEPA integration and requires the legacy code"
        )
    return build_spatial_ijepa_core(
        str(metadata.get("model_name", "vit_tiny")),  # type: ignore[arg-type]
        image_size=int(metadata.get("image_size", 64)),
        patch_size=int(metadata.get("patch_size", 8)),
        predictor_embed_dim=int(metadata.get("predictor_embed_dim", 192)),
        predictor_depth=int(metadata.get("predictor_depth", 6)),
    )


def _masks_to_device(masks: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [mask.to(device=device, dtype=torch.long, non_blocking=True) for mask in masks]


def normalize_ijepa_images(images: torch.Tensor, *, inplace: bool = False) -> torch.Tensor:
    """Apply the ImageNet normalization used by upstream I-JEPA transforms."""
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    if inplace:
        return images.sub_(mean).div_(std)
    return (images - mean) / std


def spatial_ijepa_per_sample_loss(
    core: SpatialIJEPACore,
    images: torch.Tensor,
    context_masks: list[torch.Tensor] | torch.Tensor,
    target_masks: list[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Return the upstream I-JEPA loss reduced to one value per input image."""
    losses, _ = spatial_ijepa_per_sample_loss_with_context(
        core, images, context_masks, target_masks
    )
    return losses


def spatial_ijepa_per_sample_loss_with_context(
    core: SpatialIJEPACore,
    images: torch.Tensor,
    context_masks: list[torch.Tensor] | torch.Tensor,
    target_masks: list[torch.Tensor] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image loss and pooled masked-context representations."""
    prediction, target, context, num_context_masks, num_target_masks = (
        spatial_ijepa_prediction_targets(core, images, context_masks, target_masks)
    )
    batch_size = images.shape[0]
    pooled_context = context.reshape(
        num_context_masks, batch_size, context.shape[-2], context.shape[-1]
    ).mean(dim=(0, 2))
    per_element = F.smooth_l1_loss(prediction, target, reduction="none")
    losses = per_element.reshape(
        num_target_masks, num_context_masks, batch_size, -1
    ).mean(dim=(0, 1, 3))
    return losses, pooled_context


def spatial_ijepa_prediction_targets(
    core: SpatialIJEPACore,
    images: torch.Tensor,
    context_masks: list[torch.Tensor] | torch.Tensor,
    target_masks: list[torch.Tensor] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return aligned predictor/EMA-target tokens and the encoded context."""
    if isinstance(context_masks, torch.Tensor):
        context_masks = [context_masks]
    if isinstance(target_masks, torch.Tensor):
        target_masks = [target_masks]
    device = images.device
    context_masks = _masks_to_device(context_masks, device)
    target_masks = _masks_to_device(target_masks, device)
    batch_size = images.shape[0]

    with torch.no_grad() if not core.policy.optimize_target else torch.enable_grad():
        target = core.target_encoder(images)
        target = F.layer_norm(target, (target.size(-1),))
        target = apply_masks(target, target_masks)
        target = repeat_interleave_batch(target, batch_size, repeat=len(context_masks))
    if core.policy.stop_gradient:
        target = target.detach()

    context = core.context_encoder(images, context_masks)
    prediction = core.predictor(context, context_masks, target_masks)
    return (
        prediction,
        target,
        context,
        len(context_masks),
        len(target_masks),
    )


def spatial_ijepa_loss(
    core: SpatialIJEPACore,
    images: torch.Tensor,
    context_masks: list[torch.Tensor] | torch.Tensor,
    target_masks: list[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    return spatial_ijepa_per_sample_loss(core, images, context_masks, target_masks).mean()


def spatial_ijepa_loss_with_context(
    core: SpatialIJEPACore,
    images: torch.Tensor,
    context_masks: list[torch.Tensor] | torch.Tensor,
    target_masks: list[torch.Tensor] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses, contexts = spatial_ijepa_per_sample_loss_with_context(
        core, images, context_masks, target_masks
    )
    return losses.mean(), contexts


def encode_samples_pooled(core: SpatialIJEPACore, images: torch.Tensor) -> torch.Tensor:
    """Mean-pool contextualized full-image encoder tokens."""
    return core.context_encoder(images).mean(dim=1)


@torch.no_grad()
def encode_frames_pooled(
    core: SpatialIJEPACore,
    frames: torch.Tensor,
    *,
    patch_size: int | None = None,
) -> torch.Tensor:
    if patch_size is not None and patch_size != core.patch_size:
        raise ValueError(
            f"requested patch_size={patch_size}, checkpoint model uses {core.patch_size}"
        )
    core.context_encoder.eval()
    device = next(core.context_encoder.parameters()).device
    return encode_samples_pooled(core, frames.to(device))


@torch.no_grad()
def encode_frames_pooled_batched(
    core: SpatialIJEPACore,
    frames: torch.Tensor,
    *,
    patch_size: int | None = None,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return torch.cat(
        [
            encode_frames_pooled(core, batch, patch_size=patch_size).detach().cpu()
            for batch in frames.split(batch_size)
        ],
        dim=0,
    )


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
    _atomic_torch_save(Path(path), payload)


def load_spatial_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint has an incompatible schema")
    return checkpoint
