"""Deterministic Moving-MNIST video generation with downstream motion labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

MovingMNISTSplit = Literal[
    "pretrain_train",
    "pretrain_validation",
    "probe_train",
    "probe_validation",
    "probe_evaluation",
    "test",
]

_SPLIT_CODE: dict[str, int] = {
    "pretrain_train": 11,
    "pretrain_validation": 17,
    "probe_train": 23,
    "probe_validation": 29,
    "probe_evaluation": 31,
    "test": 37,
}

# Non-overlapping MNIST image pools. Trajectories are generated independently.
_MNIST_POOLS: dict[str, tuple[bool, int, int]] = {
    "pretrain_train": (True, 0, 50_000),
    "pretrain_validation": (True, 50_000, 55_000),
    "probe_train": (True, 55_000, 58_000),
    "probe_validation": (True, 58_000, 59_000),
    "probe_evaluation": (True, 59_000, 60_000),
    "test": (False, 0, 10_000),
}


@dataclass(frozen=True)
class MovingMNISTSpec:
    """Configuration for one-object deterministic Moving-MNIST videos."""

    num_samples: int
    total_frames: int = 20
    context_frames: int = 10
    canvas_size: int = 64
    digit_size: int = 28
    min_speed: float = 2.0
    max_speed: float = 5.0
    num_directions: int = 8
    seed: int = 0

    def validate(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not 1 <= self.context_frames < self.total_frames:
            raise ValueError("context_frames must be in [1, total_frames)")
        if self.canvas_size < self.digit_size:
            raise ValueError("canvas_size must be at least digit_size")
        if not 0.0 < self.min_speed <= self.max_speed:
            raise ValueError("expected 0 < min_speed <= max_speed")
        if self.num_directions < 4:
            raise ValueError("num_directions must be at least 4")


@dataclass(frozen=True)
class TrajectoryParameters:
    digit_bank_index: int
    direction_index: int
    speed: float
    x: float
    y: float
    vx: float
    vy: float


def _load_mnist_bank(
    root: Path,
    split: MovingMNISTSplit,
    *,
    download: bool,
) -> tuple[Tensor, Tensor]:
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
        raise RuntimeError(
            "MovingMNISTDataset requires torchvision. Run `uv add torchvision`."
        ) from exc

    train, start, stop = _MNIST_POOLS[split]
    dataset = MNIST(root=str(root), train=train, download=download)
    images = dataset.data[start:stop].contiguous()
    labels = dataset.targets[start:stop].contiguous()
    if len(images) == 0:
        raise RuntimeError(f"MNIST pool for split {split!r} is empty")
    return images, labels


def make_fake_digit_bank(samples_per_class: int = 2, size: int = 28) -> tuple[Tensor, Tensor]:
    """Create a tiny deterministic digit-like bank for unit tests and offline smoke tests."""

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    images: list[Tensor] = []
    labels: list[int] = []
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for label in range(10):
        for variant in range(samples_per_class):
            center_x = 4 + ((3 * label + variant) % max(size - 8, 1))
            center_y = 4 + ((5 * label + 2 * variant) % max(size - 8, 1))
            radius = 2 + label % 4
            blob = ((xx - center_x).square() + (yy - center_y).square() <= radius**2)
            stripe = ((xx + label + variant) % (3 + label % 3) == 0) & (yy > size // 3)
            image = (blob | stripe).to(torch.uint8) * 255
            images.append(image)
            labels.append(label)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


class MovingMNISTDataset(Dataset[dict[str, Tensor]]):
    """Generate stable Moving-MNIST samples from real, split-disjoint MNIST images.

    Each index always maps to the same digit and trajectory. This is important for
    per-sample surprise tracking: a cached score refers to one immutable video.
    """

    def __init__(
        self,
        root: str | Path,
        split: MovingMNISTSplit,
        spec: MovingMNISTSpec,
        *,
        download: bool = True,
        digit_images: Tensor | None = None,
        digit_labels: Tensor | None = None,
    ) -> None:
        spec.validate()
        if split not in _SPLIT_CODE:
            raise ValueError(f"unknown split: {split}")
        self.root = Path(root)
        self.split = split
        self.spec = spec

        if (digit_images is None) != (digit_labels is None):
            raise ValueError("digit_images and digit_labels must be provided together")
        if digit_images is None:
            digit_images, digit_labels = _load_mnist_bank(
                self.root,
                split,
                download=download,
            )
        assert digit_labels is not None
        if digit_images.ndim not in {3, 4}:
            raise ValueError("digit_images must have shape [N,H,W] or [N,1,H,W]")
        if digit_images.ndim == 4:
            if digit_images.shape[1] != 1:
                raise ValueError("digit_images channel dimension must equal 1")
            digit_images = digit_images[:, 0]
        if len(digit_images) != len(digit_labels):
            raise ValueError("digit image and label counts do not match")
        self.digit_images = digit_images.to(torch.uint8).contiguous()
        self.digit_labels = digit_labels.to(torch.long).contiguous()

    def __len__(self) -> int:
        return self.spec.num_samples

    def _rng(self, index: int) -> np.random.Generator:
        if not 0 <= index < len(self):
            raise IndexError(index)
        sequence = np.random.SeedSequence(
            [self.spec.seed, _SPLIT_CODE[self.split], int(index)]
        )
        return np.random.default_rng(sequence)

    def trajectory_parameters(self, index: int) -> TrajectoryParameters:
        rng = self._rng(index)
        bank_index = int(rng.integers(0, len(self.digit_images)))
        direction_index = int(rng.integers(0, self.spec.num_directions))
        speed = float(rng.uniform(self.spec.min_speed, self.spec.max_speed))
        angle = 2.0 * np.pi * direction_index / self.spec.num_directions
        vx = speed * float(np.cos(angle))
        vy = speed * float(np.sin(angle))
        max_position = float(self.spec.canvas_size - self.spec.digit_size)
        x = float(rng.uniform(0.0, max_position))
        y = float(rng.uniform(0.0, max_position))
        return TrajectoryParameters(
            digit_bank_index=bank_index,
            direction_index=direction_index,
            speed=speed,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
        )

    def _resize_digit(self, digit: Tensor) -> Tensor:
        digit = digit.to(torch.float32).div_(255.0)[None, None]
        if digit.shape[-1] != self.spec.digit_size or digit.shape[-2] != self.spec.digit_size:
            digit = F.interpolate(
                digit,
                size=(self.spec.digit_size, self.spec.digit_size),
                mode="bilinear",
                align_corners=False,
            )
        return digit[0, 0]

    @staticmethod
    def _direction_from_velocity(vx: float, vy: float, num_directions: int) -> int:
        angle = float(np.arctan2(vy, vx)) % (2.0 * np.pi)
        scaled = angle * num_directions / (2.0 * np.pi)
        return int(np.floor(scaled + 0.5)) % num_directions

    @staticmethod
    def _reflect(position: float, velocity: float, maximum: float) -> tuple[float, float, bool]:
        proposed = position + velocity
        bounced = False
        if proposed < 0.0:
            proposed = -proposed
            velocity = -velocity
            bounced = True
        elif proposed > maximum:
            proposed = 2.0 * maximum - proposed
            velocity = -velocity
            bounced = True
        proposed = min(max(proposed, 0.0), maximum)
        return proposed, velocity, bounced

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        params = self.trajectory_parameters(index)
        digit = self._resize_digit(self.digit_images[params.digit_bank_index])
        digit_label = self.digit_labels[params.digit_bank_index]

        total_frames = self.spec.total_frames
        canvas_size = self.spec.canvas_size
        digit_size = self.spec.digit_size
        maximum = float(canvas_size - digit_size)

        video = torch.zeros(total_frames, 1, canvas_size, canvas_size, dtype=torch.float32)
        positions = torch.empty(total_frames, 2, dtype=torch.float32)
        velocities = torch.empty(total_frames, 2, dtype=torch.float32)
        transition_bounces = torch.zeros(total_frames - 1, dtype=torch.bool)

        x, y, vx, vy = params.x, params.y, params.vx, params.vy
        for frame_index in range(total_frames):
            left = int(round(x))
            top = int(round(y))
            patch = video[frame_index, 0, top : top + digit_size, left : left + digit_size]
            patch.copy_(torch.maximum(patch, digit))
            positions[frame_index] = torch.tensor(
                [x + digit_size / 2.0, y + digit_size / 2.0],
                dtype=torch.float32,
            )
            velocities[frame_index] = torch.tensor([vx, vy], dtype=torch.float32)
            if frame_index + 1 < total_frames:
                x, vx, bounce_x = self._reflect(x, vx, maximum)
                y, vy, bounce_y = self._reflect(y, vy, maximum)
                transition_bounces[frame_index] = bounce_x or bounce_y

        context_end = self.spec.context_frames - 1
        boundary_velocity = velocities[context_end]
        direction = self._direction_from_velocity(
            float(boundary_velocity[0]),
            float(boundary_velocity[1]),
            self.spec.num_directions,
        )
        target_transition_start = max(self.spec.context_frames - 1, 0)
        bounce_target = transition_bounces[target_transition_start:].any()
        speed = boundary_velocity.square().sum().sqrt()

        return {
            "sample_id": torch.tensor(index, dtype=torch.long),
            "video": video,
            "context": video[: self.spec.context_frames],
            "target": video[self.spec.context_frames :],
            "positions": positions,
            "future_positions": positions[self.spec.context_frames :],
            "boundary_velocity": boundary_velocity,
            "direction_label": torch.tensor(direction, dtype=torch.long),
            "speed": speed,
            "bounce_target": bounce_target.to(torch.long),
            "digit_label": digit_label.clone(),
        }
