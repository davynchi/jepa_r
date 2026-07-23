"""Deterministic image corruptions for controlled weighting experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

CorruptionMode = Literal["mixed", "noise", "blur", "occlusion", "blank"]

CORRUPTION_NAMES = ("clean", "noise", "blur", "occlusion", "blank")
_MODE_TO_ID = {name: index for index, name in enumerate(CORRUPTION_NAMES)}


@dataclass(frozen=True, slots=True)
class CorruptedImageBatch:
    images: torch.Tensor
    corrupted: torch.Tensor
    kind_ids: torch.Tensor


def corrupt_images(
    images: torch.Tensor,
    *,
    fraction: float,
    seed: int,
    mode: CorruptionMode = "mixed",
    noise_std: float = 0.75,
) -> CorruptedImageBatch:
    """Corrupt a fixed subset while retaining sample order and clean labels."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be [N, 3, H, W], got {tuple(images.shape)}")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    if mode not in {"mixed", "noise", "blur", "occlusion", "blank"}:
        raise ValueError(f"unknown corruption mode: {mode!r}")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    num_images = images.shape[0]
    count = int(round(num_images * fraction))
    if fraction > 0 and count == 0:
        count = 1
    count = min(count, max(num_images - 1, 0))
    generator = torch.Generator().manual_seed(seed)
    corrupted_indices = torch.randperm(num_images, generator=generator)[:count]
    corrupted = torch.zeros(num_images, dtype=torch.bool)
    corrupted[corrupted_indices] = True
    kind_ids = torch.zeros(num_images, dtype=torch.long)
    output = images.clone()

    if mode == "mixed":
        selected_kind_ids = torch.arange(count, dtype=torch.long) % 4 + 1
        selected_kind_ids = selected_kind_ids[
            torch.randperm(count, generator=generator)
        ]
    else:
        selected_kind_ids = torch.full(
            (count,), _MODE_TO_ID[mode], dtype=torch.long
        )
    kind_ids[corrupted_indices] = selected_kind_ids

    noise_indices = corrupted_indices[selected_kind_ids == _MODE_TO_ID["noise"]]
    if noise_indices.numel() > 0:
        noise = torch.randn(
            output[noise_indices].shape,
            generator=generator,
            dtype=output.dtype,
        )
        output[noise_indices] = (output[noise_indices] + noise_std * noise).clamp(0, 1)

    blur_indices = corrupted_indices[selected_kind_ids == _MODE_TO_ID["blur"]]
    if blur_indices.numel() > 0:
        output[blur_indices] = F.avg_pool2d(
            output[blur_indices], kernel_size=11, stride=1, padding=5
        )

    occlusion_indices = corrupted_indices[
        selected_kind_ids == _MODE_TO_ID["occlusion"]
    ]
    if occlusion_indices.numel() > 0:
        height, width = output.shape[-2:]
        top, bottom = height // 4, height - height // 4
        left, right = width // 4, width - width // 4
        output[occlusion_indices, :, top:bottom, left:right] = 0.5

    blank_indices = corrupted_indices[selected_kind_ids == _MODE_TO_ID["blank"]]
    if blank_indices.numel() > 0:
        output[blank_indices] = 0.5

    return CorruptedImageBatch(
        images=output,
        corrupted=corrupted,
        kind_ids=kind_ids,
    )
