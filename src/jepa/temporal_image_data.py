"""Rendered-image entity/context world (Section 2.4 of the experiment spec).

Same hidden-factor principle as :mod:`jepa.temporal_data` -- a slow entity
``E_t in {none, circle, square, triangle}`` and a fast context ``C_t``
(position, scale, rotation, color, visibility/opacity, background) -- but the
observation map ``h`` is a small vectorized rasterizer instead of a random
matrix or MLP.

Datasets here deliberately expose ``.observations`` (flattened frames),
``.entities``, ``.contexts``, and ``config.trajectory_length`` /
``config.observation_dim`` with the exact same meaning as
:class:`jepa.temporal_data.EntityContextTrajectoryDataset`. That duck-typed
compatibility lets every pair-builder and per-epoch training helper in
:mod:`jepa.temporal_data` / :mod:`jepa.temporal_training` (``make_temporal_pairs``,
``make_hierarchical_pairs``, ``_standard_epoch_loss``, ``_hierarchical_epoch_loss``,
``encode_split_standard``, ``encode_split_hierarchical``) run unmodified on
image trajectories -- nothing about JEPA training needed to be re-implemented
for pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from jepa.config import derive_seed
from jepa.temporal_image_config import ENTITY_NAMES, ImageDataConfig, SpatialMaskingConfig

Split = str

_X_RANGE = (-0.15, 1.15)
_Y_RANGE = (-0.15, 1.15)
_S_RANGE = (0.12, 0.30)
_COLOR_RANGE = (0.2, 1.0)
_VISIBILITY_RANGE = (0.3, 1.0)
_BACKGROUND_RANGE = (0.05, 0.95)
_SUPERSAMPLE = 2


def _cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _sample_uniform(generator: torch.Generator, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * torch.rand((), generator=generator).item())


def _sample_context(generator: torch.Generator) -> list[float]:
    return [
        _sample_uniform(generator, *_X_RANGE),
        _sample_uniform(generator, *_Y_RANGE),
        _sample_uniform(generator, *_S_RANGE),
        _sample_uniform(generator, 0.0, 2 * math.pi),
        _sample_uniform(generator, *_COLOR_RANGE),
        _sample_uniform(generator, *_COLOR_RANGE),
        _sample_uniform(generator, *_COLOR_RANGE),
        _sample_uniform(generator, *_VISIBILITY_RANGE),
        _sample_uniform(generator, *_BACKGROUND_RANGE),
    ]


def _ar_step_bounded(
    value: float, lo: float, hi: float, rho: float, generator: torch.Generator
) -> float:
    mean = (lo + hi) / 2.0
    span = (hi - lo) / 2.0
    noise = (1.0 - rho) * span * float(torch.randn((), generator=generator).item())
    updated = rho * (value - mean) + mean + noise
    return float(min(max(updated, lo), hi))


def _ar_step_angle(value: float, rho: float, generator: torch.Generator) -> float:
    noise = (1.0 - rho) * math.pi * float(torch.randn((), generator=generator).item())
    return float((value + noise) % (2 * math.pi))


def _step_context(
    context: list[float], config: ImageDataConfig, generator: torch.Generator
) -> list[float]:
    x, y, s, theta, r, g, b, visibility, background = context
    x = _ar_step_bounded(x, *_X_RANGE, config.context_smoothness, generator)
    y = _ar_step_bounded(y, *_Y_RANGE, config.context_smoothness, generator)
    s = _ar_step_bounded(s, *_S_RANGE, config.context_smoothness, generator)
    theta = _ar_step_angle(theta, config.context_smoothness, generator)
    r = _ar_step_bounded(r, *_COLOR_RANGE, config.context_smoothness, generator)
    g = _ar_step_bounded(g, *_COLOR_RANGE, config.context_smoothness, generator)
    b = _ar_step_bounded(b, *_COLOR_RANGE, config.context_smoothness, generator)
    if torch.rand((), generator=generator).item() < config.visibility_change_probability:
        visibility = _sample_uniform(generator, *_VISIBILITY_RANGE)
    if torch.rand((), generator=generator).item() < config.background_change_probability:
        background = _sample_uniform(generator, *_BACKGROUND_RANGE)
    return [x, y, s, theta, r, g, b, visibility, background]


def _edge_sign(
    px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float
) -> np.ndarray:
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _triangle_mask(rx: np.ndarray, ry: np.ndarray, s: float) -> np.ndarray:
    # Equilateral triangle, circumradius s, centered at origin, pointing up (before rotation).
    v0 = (0.0, -s)
    v1 = (-s * math.sqrt(3) / 2, s * 0.5)
    v2 = (s * math.sqrt(3) / 2, s * 0.5)
    d1 = _edge_sign(rx, ry, *v0, *v1)
    d2 = _edge_sign(rx, ry, *v1, *v2)
    d3 = _edge_sign(rx, ry, *v2, *v0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def render_frame(
    entity_index: int,
    context: np.ndarray,
    *,
    image_size: int,
    num_channels: int,
    supersample: int = _SUPERSAMPLE,
) -> np.ndarray:
    """Rasterize X_t = h(E_t, C_t) for one frame. Returns float32 [C, H, W] in [0, 1]."""
    x, y, s, theta, r, g, b, visibility, background = (float(v) for v in context)
    size = image_size * supersample
    rows, cols = np.mgrid[0:size, 0:size]
    xs = (cols + 0.5) / size
    ys = (rows + 0.5) / size
    canvas = np.full((size, size, num_channels), background, dtype=np.float32)

    if entity_index != 0:
        dx = xs - x
        dy = ys - y
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rx = dx * cos_t + dy * sin_t
        ry = -dx * sin_t + dy * cos_t
        if entity_index == 1:  # circle
            mask = rx**2 + ry**2 <= s**2
        elif entity_index == 2:  # square
            mask = (np.abs(rx) <= s) & (np.abs(ry) <= s)
        elif entity_index == 3:  # triangle
            mask = _triangle_mask(rx, ry, s)
        else:
            raise ValueError(f"unknown entity index: {entity_index}")
        color = np.array([r, g, b][:num_channels], dtype=np.float32)
        canvas[mask] = visibility * color + (1.0 - visibility) * canvas[mask]

    downsampled = canvas.reshape(
        image_size, supersample, image_size, supersample, num_channels
    ).mean(axis=(1, 3))
    return np.ascontiguousarray(downsampled.transpose(2, 0, 1).astype(np.float32))


def _generate_entity_context_trajectory(
    config: ImageDataConfig, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _cpu_generator(seed)
    length = config.trajectory_length
    entities = torch.empty(length, dtype=torch.long)
    contexts = torch.empty(length, config.context_dim)

    entity = int(torch.randint(config.num_entities, (), generator=generator).item())
    context = _sample_context(generator)
    for t in range(length):
        entities[t] = entity
        contexts[t] = torch.tensor(context, dtype=torch.float32)
        if t + 1 < length:
            if torch.rand((), generator=generator).item() < config.entity_switch_probability:
                offset = 1 + int(
                    torch.randint(config.num_entities - 1, (), generator=generator).item()
                )
                entity = (entity + offset) % config.num_entities
            context = _step_context(context, config, generator)
    return entities, contexts


def _split_size_and_seed(config: ImageDataConfig, split: Split, *, static: bool) -> tuple[int, int]:
    if split == "train":
        return (
            config.num_train_samples if static else config.num_train_trajectories
        ), config.train_sample_seed
    if split == "validation":
        return (
            config.num_val_samples if static else config.num_val_trajectories
        ), config.validation_sample_seed
    if split == "test":
        return (
            config.num_test_samples if static else config.num_test_trajectories
        ), config.test_sample_seed
    raise ValueError(f"unknown split: {split!r}")


class EntityContextImageTrajectoryDataset(Dataset):
    """Rendered analogue of ``EntityContextTrajectoryDataset``.

    Exposes ``.observations`` (flattened frames) alongside ``.frames`` (the
    unflattened [N, T, C, H, W] tensor used for visualization) so the
    vector-world pair builders and per-epoch training helpers work unchanged.
    """

    def __init__(self, config: ImageDataConfig, split: Split) -> None:
        self.config = config
        self.split = split
        size, split_seed = _split_size_and_seed(config, split, static=False)

        entities = torch.empty(size, config.trajectory_length, dtype=torch.long)
        contexts = torch.empty(size, config.trajectory_length, config.context_dim)
        frames = torch.empty(
            size,
            config.trajectory_length,
            config.num_channels,
            config.image_size,
            config.image_size,
        )
        for index in range(size):
            traj_seed = derive_seed(split_seed, "trajectory", index)
            entity_seq, context_seq = _generate_entity_context_trajectory(config, seed=traj_seed)
            entities[index] = entity_seq
            contexts[index] = context_seq
            for t in range(config.trajectory_length):
                frame = render_frame(
                    int(entity_seq[t]),
                    context_seq[t].numpy(),
                    image_size=config.image_size,
                    num_channels=config.num_channels,
                )
                frames[index, t] = torch.from_numpy(frame)

        self.entities = entities
        self.contexts = contexts
        self.frames = frames
        self.observations = frames.reshape(size, config.trajectory_length, -1)

    def __len__(self) -> int:
        return self.entities.shape[0]

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "frames": self.frames[index],
            "entity": self.entities[index],
            "context": self.contexts[index],
            "trajectory_id": index,
        }


@dataclass(frozen=True, slots=True)
class ImageDatasetSplits:
    train: EntityContextImageTrajectoryDataset
    validation: EntityContextImageTrajectoryDataset
    test: EntityContextImageTrajectoryDataset


def build_image_dataset_splits(config: ImageDataConfig) -> ImageDatasetSplits:
    return ImageDatasetSplits(
        train=EntityContextImageTrajectoryDataset(config, "train"),
        validation=EntityContextImageTrajectoryDataset(config, "validation"),
        test=EntityContextImageTrajectoryDataset(config, "test"),
    )


class StaticImageDataset(Dataset):
    """Independent-sample rendered images: ``I = h(E, C)``, no temporal correlation."""

    def __init__(
        self, config: ImageDataConfig, split: Split, *, exclude_none: bool = False
    ) -> None:
        self.config = config
        self.split = split
        size, split_seed = _split_size_and_seed(config, split, static=True)
        low = 1 if exclude_none else 0

        entities = torch.empty(size, dtype=torch.long)
        contexts = torch.empty(size, config.context_dim)
        images = torch.empty(size, config.num_channels, config.image_size, config.image_size)
        for index in range(size):
            generator = _cpu_generator(derive_seed(split_seed, "sample", index))
            entity = low + int(
                torch.randint(config.num_entities - low, (), generator=generator).item()
            )
            context = _sample_context(generator)
            frame = render_frame(
                entity,
                np.array(context, dtype=np.float32),
                image_size=config.image_size,
                num_channels=config.num_channels,
            )
            entities[index] = entity
            contexts[index] = torch.tensor(context, dtype=torch.float32)
            images[index] = torch.from_numpy(frame)

        self.entities = entities
        self.contexts = contexts
        self.images = images
        self.observations = images.reshape(size, -1)

    def __len__(self) -> int:
        return self.entities.shape[0]

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": self.images[index],
            "entity": int(self.entities[index]),
            "context": self.contexts[index],
            "sample_id": index,
        }


def block_mask(
    image: torch.Tensor, block_fraction: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """Split an image into a complementary (visible, target-block) pair.

    ``visible`` zeroes out one contiguous rectangular block (the spatial-JEPA
    "masked region"); ``target`` keeps only that block's original pixels and
    zeroes everything else -- so the predictor is asked to reconstruct the
    *target encoder's* representation of specifically the masked content from
    the *context encoder's* representation of everything else, while both
    views share the same [C, H, W] shape (reusing ``build_jepa_core`` as-is).
    """
    channels, height, width = image.shape
    block_h = max(1, int(round(height * math.sqrt(block_fraction))))
    block_w = max(1, int(round(width * math.sqrt(block_fraction))))
    top = int(torch.randint(height - block_h + 1, (), generator=generator).item())
    left = int(torch.randint(width - block_w + 1, (), generator=generator).item())

    visible = image.clone()
    visible[:, top : top + block_h, left : left + block_w] = 0.0
    target = torch.zeros_like(image)
    target[:, top : top + block_h, left : left + block_w] = image[
        :, top : top + block_h, left : left + block_w
    ]
    return visible, target, (top, left, block_h, block_w)


@dataclass(frozen=True, slots=True)
class StaticSpatialDataset:
    """Block-masked (visible, target) views over a :class:`StaticImageDataset`."""

    base: StaticImageDataset
    visible: torch.Tensor
    target: torch.Tensor
    bounds: tuple[tuple[int, int, int, int], ...]

    def __len__(self) -> int:
        return len(self.base)


def build_static_spatial_dataset(
    base: StaticImageDataset, spatial: SpatialMaskingConfig, *, seed: int
) -> StaticSpatialDataset:
    generator = _cpu_generator(seed)
    visible = torch.empty_like(base.images)
    target = torch.empty_like(base.images)
    bounds: list[tuple[int, int, int, int]] = []
    for index in range(len(base)):
        v, t, b = block_mask(base.images[index], spatial.block_fraction, generator)
        visible[index] = v
        target[index] = t
        bounds.append(b)
    return StaticSpatialDataset(base=base, visible=visible, target=target, bounds=tuple(bounds))


@dataclass(frozen=True, slots=True)
class ImageCounterfactualPairs:
    same_entity: torch.Tensor
    same_entity_x1: torch.Tensor
    same_entity_x2: torch.Tensor
    diff_entity_entities: torch.Tensor
    diff_entity_x1: torch.Tensor
    diff_entity_x2: torch.Tensor


def build_image_counterfactual_pairs(
    config: ImageDataConfig, *, num_pairs: int, seed: int
) -> ImageCounterfactualPairs:
    """Section 2.4.8: controlled image pairs changing exactly one hidden factor."""
    generator = _cpu_generator(seed)

    same_entity = torch.empty(num_pairs, dtype=torch.long)
    same_x1 = torch.empty(num_pairs, config.num_channels, config.image_size, config.image_size)
    same_x2 = torch.empty_like(same_x1)
    for i in range(num_pairs):
        entity = int(torch.randint(config.num_entities, (), generator=generator).item())
        context_1 = _sample_context(generator)
        context_2 = _sample_context(generator)
        same_entity[i] = entity
        same_x1[i] = torch.from_numpy(
            render_frame(
                entity,
                np.array(context_1, dtype=np.float32),
                image_size=config.image_size,
                num_channels=config.num_channels,
            )
        )
        same_x2[i] = torch.from_numpy(
            render_frame(
                entity,
                np.array(context_2, dtype=np.float32),
                image_size=config.image_size,
                num_channels=config.num_channels,
            )
        )

    diff_entities = torch.empty(num_pairs, 2, dtype=torch.long)
    diff_x1 = torch.empty_like(same_x1)
    diff_x2 = torch.empty_like(same_x1)
    for i in range(num_pairs):
        entity_1 = int(torch.randint(config.num_entities, (), generator=generator).item())
        offset = 1 + int(torch.randint(config.num_entities - 1, (), generator=generator).item())
        entity_2 = (entity_1 + offset) % config.num_entities
        context = _sample_context(generator)
        diff_entities[i] = torch.tensor([entity_1, entity_2])
        diff_x1[i] = torch.from_numpy(
            render_frame(
                entity_1,
                np.array(context, dtype=np.float32),
                image_size=config.image_size,
                num_channels=config.num_channels,
            )
        )
        diff_x2[i] = torch.from_numpy(
            render_frame(
                entity_2,
                np.array(context, dtype=np.float32),
                image_size=config.image_size,
                num_channels=config.num_channels,
            )
        )

    return ImageCounterfactualPairs(
        same_entity=same_entity,
        same_entity_x1=same_x1,
        same_entity_x2=same_x2,
        diff_entity_entities=diff_entities,
        diff_entity_x1=diff_x1,
        diff_entity_x2=diff_x2,
    )


__all__ = [
    "ENTITY_NAMES",
    "EntityContextImageTrajectoryDataset",
    "ImageCounterfactualPairs",
    "ImageDatasetSplits",
    "StaticImageDataset",
    "StaticSpatialDataset",
    "block_mask",
    "build_image_counterfactual_pairs",
    "build_image_dataset_splits",
    "build_static_spatial_dataset",
    "render_frame",
]
