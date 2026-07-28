"""Barlow Twins auxiliary objective adapted to the project's image encoders."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError("off_diagonal expects a square matrix")
    return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


class BarlowTwinsProjector(nn.Module):
    """Official projector/normalization/loss with a configurable backbone width."""

    def __init__(
        self,
        input_dim: int,
        projector_dims: Sequence[int],
        *,
        redundancy_weight: float = 0.0051,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or not projector_dims or any(dim <= 0 for dim in projector_dims):
            raise ValueError("projector dimensions must be positive")
        if redundancy_weight < 0:
            raise ValueError("redundancy_weight must be non-negative")
        sizes = [input_dim, *projector_dims]
        layers: list[nn.Module] = []
        for index in range(len(sizes) - 2):
            layers.extend(
                (
                    nn.Linear(sizes[index], sizes[index + 1], bias=False),
                    nn.BatchNorm1d(sizes[index + 1]),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.normalization = nn.BatchNorm1d(sizes[-1], affine=False)
        self.redundancy_weight = redundancy_weight

    def forward(
        self,
        first_representation: torch.Tensor,
        second_representation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if first_representation.shape != second_representation.shape:
            raise ValueError("Barlow views must have matching representation shapes")
        if first_representation.ndim != 2 or first_representation.shape[0] < 2:
            raise ValueError("Barlow representations must have shape [batch>=2, dim]")
        first = self.projector(first_representation)
        second = self.projector(second_representation)
        cross_correlation = self.normalization(first).T @ self.normalization(second)
        cross_correlation = cross_correlation / first.shape[0]
        on_diagonal = torch.diagonal(cross_correlation).add(-1).square().sum()
        off_diagonal_loss = off_diagonal(cross_correlation).square().sum()
        loss = on_diagonal + self.redundancy_weight * off_diagonal_loss
        return loss, {
            "official_barlow/loss": float(loss.detach().cpu().item()),
            "official_barlow/on_diagonal": float(on_diagonal.detach().cpu().item()),
            "official_barlow/off_diagonal": float(off_diagonal_loss.detach().cpu().item()),
            "official_barlow/diag_mean": float(
                torch.diagonal(cross_correlation).detach().mean().cpu().item()
            ),
        }


def _random_resized_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    area = width * height
    log_min_ratio, log_max_ratio = math.log(3 / 4), math.log(4 / 3)
    for _ in range(10):
        target_area = random.uniform(0.08, 1.0) * area
        aspect_ratio = math.exp(random.uniform(log_min_ratio, log_max_ratio))
        crop_width = round(math.sqrt(target_area * aspect_ratio))
        crop_height = round(math.sqrt(target_area / aspect_ratio))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            return image.crop((left, top, left + crop_width, top + crop_height)).resize(
                (size, size), Image.Resampling.BICUBIC
            )
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size)).resize(
        (size, size), Image.Resampling.BICUBIC
    )


def _adjust_hue(image: Image.Image, factor: float) -> Image.Image:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + round(factor * 255)) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _color_jitter(image: Image.Image) -> Image.Image:
    operations = [
        lambda value: ImageEnhance.Brightness(value).enhance(random.uniform(0.6, 1.4)),
        lambda value: ImageEnhance.Contrast(value).enhance(random.uniform(0.6, 1.4)),
        lambda value: ImageEnhance.Color(value).enhance(random.uniform(0.8, 1.2)),
        lambda value: _adjust_hue(value, random.uniform(-0.1, 0.1)),
    ]
    random.shuffle(operations)
    for operation in operations:
        image = operation(image)
    return image


class OfficialBarlowTransform:
    """The two-view augmentation recipe from facebookresearch/barlowtwins."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def _view(self, image: Image.Image, *, blur_probability: float, solarize: bool) -> torch.Tensor:
        view = _random_resized_crop(image, self.image_size)
        if random.random() < 0.5:
            view = ImageOps.mirror(view)
        if random.random() < 0.8:
            view = _color_jitter(view)
        if random.random() < 0.2:
            view = ImageOps.grayscale(view).convert("RGB")
        if random.random() < blur_probability:
            view = view.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
        if solarize and random.random() < 0.2:
            view = ImageOps.solarize(view)
        array = np.asarray(view, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._view(image, blur_probability=1.0, solarize=False),
            self._view(image, blur_probability=0.1, solarize=True),
        )


class BarlowImageDataset(Dataset):
    def __init__(self, paths: Sequence[str | Path], *, image_size: int) -> None:
        if not paths:
            raise ValueError("BarlowImageDataset requires at least one image")
        self.paths = tuple(Path(path) for path in paths)
        self.transform = OfficialBarlowTransform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def pooled_encoder_representation(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    tokens = encoder(images)
    if tokens.ndim != 3:
        raise RuntimeError("image encoder must return [batch, tokens, dimensions]")
    return tokens.mean(dim=1)
