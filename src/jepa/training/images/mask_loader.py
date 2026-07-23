"""Asynchronous input pipelines for upstream-style I-JEPA training."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from jepa.configs.base import derive_seed
from jepa.training.images.ijepa_spatial import MaskConfig, sample_masks


@dataclass(slots=True)
class PreparedIJEPABatch:
    indices: torch.Tensor
    context_masks: list[torch.Tensor]
    target_masks: list[torch.Tensor]
    images: torch.Tensor | None = None
    crop_theta: torch.Tensor | None = None
    horizontal_flip: torch.Tensor | None = None

    def pin_memory(self) -> PreparedIJEPABatch:
        self.indices = self.indices.pin_memory()
        self.context_masks = [mask.pin_memory() for mask in self.context_masks]
        self.target_masks = [mask.pin_memory() for mask in self.target_masks]
        if self.images is not None:
            self.images = self.images.pin_memory()
        if self.crop_theta is not None:
            self.crop_theta = self.crop_theta.pin_memory()
        if self.horizontal_flip is not None:
            self.horizontal_flip = self.horizontal_flip.pin_memory()
        return self


class _SharedEpochOrderDataset(Dataset):
    """Expose a mutable epoch order to persistent DataLoader workers."""

    def __init__(self, *, num_draws: int, batch_size: int) -> None:
        if num_draws <= 0:
            raise ValueError("num_draws must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.num_draws = num_draws
        self.batch_size = batch_size
        self.order = torch.empty(num_draws, dtype=torch.long).share_memory_()
        self.mask_seeds = torch.empty(
            math.ceil(num_draws / batch_size),
            dtype=torch.long,
        ).share_memory_()

    def set_epoch(self, order: torch.Tensor, *, seed: int) -> None:
        order = order.detach().cpu().to(torch.long)
        if order.shape != self.order.shape:
            raise ValueError(
                f"epoch order must have shape {tuple(self.order.shape)}, got {tuple(order.shape)}"
            )
        self.order.copy_(order)
        seeds = torch.tensor(
            [
                derive_seed(seed, "mask-batch", batch_index)
                for batch_index in range(self.mask_seeds.numel())
            ],
            dtype=torch.long,
        )
        self.mask_seeds.copy_(seeds)

    def __len__(self) -> int:
        return self.num_draws

    def __getitem__(self, position: int) -> tuple[int, int, int]:
        batch_index = position // self.batch_size
        return position, int(self.order[position]), int(self.mask_seeds[batch_index])


def _sample_resized_crop_theta(
    *,
    batch_size: int,
    source_size: tuple[int, int],
    scale: tuple[float, float],
    horizontal_flip_probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_h, source_w = source_size
    area = source_h * source_w
    crop_h = torch.full((batch_size,), source_h, dtype=torch.long)
    crop_w = torch.full((batch_size,), source_w, dtype=torch.long)
    valid = torch.zeros(batch_size, dtype=torch.bool)
    for _ in range(10):
        target_area = area * (
            scale[0] + torch.rand(batch_size, generator=generator) * (scale[1] - scale[0])
        )
        log_ratio = math.log(3 / 4) + torch.rand(batch_size, generator=generator) * (
            math.log(4 / 3) - math.log(3 / 4)
        )
        ratio = log_ratio.exp()
        proposed_w = (target_area * ratio).sqrt().round().to(torch.long)
        proposed_h = (target_area / ratio).sqrt().round().to(torch.long)
        accepted = (
            ~valid
            & (proposed_h > 0)
            & (proposed_h <= source_h)
            & (proposed_w > 0)
            & (proposed_w <= source_w)
        )
        crop_h[accepted] = proposed_h[accepted]
        crop_w[accepted] = proposed_w[accepted]
        valid |= accepted
        if bool(valid.all()):
            break

    top = (torch.rand(batch_size, generator=generator) * (source_h - crop_h + 1)).floor()
    left = (torch.rand(batch_size, generator=generator) * (source_w - crop_w + 1)).floor()
    theta = torch.zeros((batch_size, 2, 3), dtype=torch.float32)
    theta[:, 0, 0] = crop_w / source_w
    theta[:, 1, 1] = crop_h / source_h
    theta[:, 0, 2] = 2.0 * (left + crop_w / 2.0) / source_w - 1.0
    theta[:, 1, 2] = 2.0 * (top + crop_h / 2.0) / source_h - 1.0
    flip = torch.rand(batch_size, generator=generator) < horizontal_flip_probability
    return theta, flip


class _MaskCollator:
    """Generate upstream-style masks inside a DataLoader worker."""

    def __init__(
        self,
        *,
        grid: int,
        mask_config: MaskConfig,
        source_size: tuple[int, int],
        crop_scale: tuple[float, float],
        horizontal_flip_probability: float,
    ) -> None:
        self.grid = grid
        self.mask_config = mask_config
        self.source_size = source_size
        self.crop_scale = crop_scale
        self.horizontal_flip_probability = horizontal_flip_probability

    def __call__(self, rows: list[tuple[int, int, int]]) -> PreparedIJEPABatch:
        if not rows:
            raise ValueError("cannot collate an empty mask batch")
        indices = torch.tensor([row[1] for row in rows], dtype=torch.long)
        seed = rows[0][2]
        if any(row[2] != seed for row in rows):
            raise RuntimeError("a DataLoader batch crossed deterministic mask-seed boundaries")
        mask_generator = torch.Generator().manual_seed(seed)
        context_masks, target_masks = sample_masks(
            self.grid,
            self.grid,
            self.mask_config,
            mask_generator,
            batch_size=len(rows),
        )
        transform_generator = torch.Generator().manual_seed(derive_seed(seed, "transforms"))
        theta, flip = _sample_resized_crop_theta(
            batch_size=len(rows),
            source_size=self.source_size,
            scale=self.crop_scale,
            horizontal_flip_probability=self.horizontal_flip_probability,
            generator=transform_generator,
        )
        return PreparedIJEPABatch(
            indices=indices,
            context_masks=context_masks,
            target_masks=target_masks,
            crop_theta=theta,
            horizontal_flip=flip,
        )


def apply_prepared_crop(
    images: torch.Tensor,
    *,
    theta: torch.Tensor,
    horizontal_flip: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Apply worker-prepared crop geometry as one batched GPU operation."""
    batch_size, channels = images.shape[:2]
    theta = theta.to(device=images.device, non_blocking=True)
    horizontal_flip = horizontal_flip.to(device=images.device, non_blocking=True)
    grid = F.affine_grid(
        theta,
        size=[batch_size, channels, output_size, output_size],
        align_corners=False,
    )
    crops = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return torch.where(horizontal_flip[:, None, None, None], crops.flip(-1), crops)


def _init_mask_worker(_: int) -> None:
    torch.set_num_threads(1)


class IndexMaskLoader:
    """Prefetch sampled indices and masks while the GPU trains the current batch."""

    def __init__(
        self,
        *,
        num_draws: int,
        batch_size: int,
        grid: int,
        mask_config: MaskConfig,
        source_size: tuple[int, int],
        crop_scale: tuple[float, float],
        horizontal_flip_probability: float,
        num_workers: int,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
    ) -> None:
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        self.dataset = _SharedEpochOrderDataset(
            num_draws=num_draws,
            batch_size=batch_size,
        )
        loader_kwargs: dict[str, object] = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "drop_last": False,
            "collate_fn": _MaskCollator(
                grid=grid,
                mask_config=mask_config,
                source_size=source_size,
                crop_scale=crop_scale,
                horizontal_flip_probability=horizontal_flip_probability,
            ),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "worker_init_fn": _init_mask_worker,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["multiprocessing_context"] = "spawn"
        self.loader: DataLoader[PreparedIJEPABatch] = DataLoader(**loader_kwargs)

    def iter_epoch(self, order: torch.Tensor, *, seed: int) -> Iterator[PreparedIJEPABatch]:
        self.dataset.set_epoch(order, seed=seed)
        return iter(self.loader)


def _pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return tensor.sub_(mean).div_(std)


def _random_resized_crop_pil(
    image: Image.Image,
    *,
    output_size: int,
    scale: tuple[float, float],
    horizontal_flip_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    source_w, source_h = image.size
    area = source_h * source_w
    top, left, crop_h, crop_w = 0, 0, source_h, source_w
    for _ in range(10):
        target_area = area * (
            scale[0] + torch.rand((), generator=generator).item() * (scale[1] - scale[0])
        )
        log_ratio = math.log(3 / 4) + torch.rand((), generator=generator).item() * (
            math.log(4 / 3) - math.log(3 / 4)
        )
        ratio = math.exp(log_ratio)
        proposed_w = round(math.sqrt(target_area * ratio))
        proposed_h = round(math.sqrt(target_area / ratio))
        if 0 < proposed_w <= source_w and 0 < proposed_h <= source_h:
            crop_w, crop_h = proposed_w, proposed_h
            top = int(
                torch.randint(
                    0,
                    source_h - crop_h + 1,
                    (),
                    generator=generator,
                ).item()
            )
            left = int(
                torch.randint(
                    0,
                    source_w - crop_w + 1,
                    (),
                    generator=generator,
                ).item()
            )
            break
    image = image.crop((left, top, left + crop_w, top + crop_h))
    image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)
    if torch.rand((), generator=generator).item() < horizontal_flip_probability:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return _pil_to_normalized_tensor(image)


class _FileEpochDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        *,
        num_draws: int,
        batch_size: int,
        output_size: int,
        crop_scale: tuple[float, float],
        horizontal_flip_probability: float,
    ) -> None:
        if num_draws <= 0:
            raise ValueError("num_draws must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.num_draws = num_draws
        self.batch_size = batch_size
        self.order = torch.empty(num_draws, dtype=torch.long).share_memory_()
        self.mask_seeds = torch.empty(
            math.ceil(num_draws / batch_size),
            dtype=torch.long,
        ).share_memory_()
        self.paths = paths
        self.output_size = output_size
        self.crop_scale = crop_scale
        self.horizontal_flip_probability = horizontal_flip_probability

    def set_epoch(self, order: torch.Tensor, *, seed: int) -> None:
        order = order.detach().cpu().to(torch.long)
        if order.shape != self.order.shape:
            raise ValueError(
                f"epoch order must have shape {tuple(self.order.shape)}, got {tuple(order.shape)}"
            )
        self.order.copy_(order)
        seeds = torch.tensor(
            [
                derive_seed(seed, "mask-batch", batch_index)
                for batch_index in range(self.mask_seeds.numel())
            ],
            dtype=torch.long,
        )
        self.mask_seeds.copy_(seeds)

    def __len__(self) -> int:
        return self.num_draws

    def __getitem__(self, position: int) -> tuple[int, int, int, torch.Tensor]:
        batch_index = position // self.batch_size
        sample_index = int(self.order[position])
        mask_seed = int(self.mask_seeds[batch_index])
        generator = torch.Generator().manual_seed(
            derive_seed(mask_seed, "file-transform", position)
        )
        with Image.open(self.paths[sample_index]) as source:
            image = _random_resized_crop_pil(
                source.convert("RGB"),
                output_size=self.output_size,
                scale=self.crop_scale,
                horizontal_flip_probability=self.horizontal_flip_probability,
                generator=generator,
            )
        return position, sample_index, mask_seed, image


class _FileMaskCollator:
    def __init__(self, *, grid: int, mask_config: MaskConfig) -> None:
        self.grid = grid
        self.mask_config = mask_config

    def __call__(self, rows: list[tuple[int, int, int, torch.Tensor]]) -> PreparedIJEPABatch:
        if not rows:
            raise ValueError("cannot collate an empty image batch")
        seed = rows[0][2]
        if any(row[2] != seed for row in rows):
            raise RuntimeError("a DataLoader batch crossed deterministic mask-seed boundaries")
        context_masks, target_masks = sample_masks(
            self.grid,
            self.grid,
            self.mask_config,
            torch.Generator().manual_seed(seed),
            batch_size=len(rows),
        )
        return PreparedIJEPABatch(
            indices=torch.tensor([row[1] for row in rows], dtype=torch.long),
            images=torch.stack([row[3] for row in rows]),
            context_masks=context_masks,
            target_masks=target_masks,
        )


class FileImageMaskLoader:
    """Upstream-style file decode, augmentation, collation, and mask prefetch."""

    def __init__(
        self,
        paths: list[Path],
        *,
        num_draws: int,
        batch_size: int,
        image_size: int,
        patch_size: int,
        mask_config: MaskConfig,
        crop_scale: tuple[float, float],
        horizontal_flip_probability: float,
        num_workers: int,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
    ) -> None:
        self.dataset = _FileEpochDataset(
            paths,
            num_draws=num_draws,
            batch_size=batch_size,
            output_size=image_size,
            crop_scale=crop_scale,
            horizontal_flip_probability=horizontal_flip_probability,
        )
        loader_kwargs: dict[str, object] = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "drop_last": False,
            "collate_fn": _FileMaskCollator(
                grid=image_size // patch_size,
                mask_config=mask_config,
            ),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "worker_init_fn": _init_mask_worker,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["multiprocessing_context"] = "spawn"
        self.loader: DataLoader[PreparedIJEPABatch] = DataLoader(**loader_kwargs)

    def iter_epoch(self, order: torch.Tensor, *, seed: int) -> Iterator[PreparedIJEPABatch]:
        self.dataset.set_epoch(order, seed=seed)
        return iter(self.loader)


__all__ = [
    "FileImageMaskLoader",
    "IndexMaskLoader",
    "PreparedIJEPABatch",
    "apply_prepared_crop",
]
