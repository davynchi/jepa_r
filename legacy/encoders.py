"""Image encoders used only by the archived patch-based spatial pipeline."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from jepa.models.encoders import CNNEncoder, TanhPredictor

LegacyArchitecture = Literal["cnn", "resnet"]


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


class ResidualBlock(nn.Module):
    """Small residual block for patch-sized RGB inputs."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class ResNetPatchEncoder(nn.Module):
    """A compact ResNet-style encoder for flattened RGB image patches."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        channels: int = 3,
        width: int = 64,
    ) -> None:
        super().__init__()
        _require_positive("input_dim", input_dim)
        _require_positive("latent_dim", latent_dim)
        side_float = math.sqrt(input_dim / channels)
        side = round(side_float)
        if channels * side * side != input_dim:
            raise ValueError(
                f"input_dim ({input_dim}) is not channels ({channels}) x a square "
                "image side; ResNetPatchEncoder only supports square RGB-style inputs"
            )
        self.channels = channels
        self.side = side
        self.stem = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, width),
            ResidualBlock(width, width),
            ResidualBlock(width, width * 2, stride=2),
            ResidualBlock(width * 2, width * 2),
            ResidualBlock(width * 2, width * 4, stride=2),
            ResidualBlock(width * 4, width * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 4, latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_shape = inputs.shape[:-1]
        images = inputs.reshape(-1, self.channels, self.side, self.side)
        features = self.stem(images)
        features = self.blocks(features)
        features = self.pool(features).flatten(1)
        latents = self.fc(features)
        return latents.reshape(*batch_shape, -1)


def build_legacy_model_pair(
    architecture: LegacyArchitecture,
    *,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int = 64,
    hidden_layers: int = 1,
) -> tuple[nn.Module, nn.Module]:
    if architecture == "cnn":
        return (
            CNNEncoder(input_dim, latent_dim),
            TanhPredictor(latent_dim, hidden_dim, hidden_layers),
        )
    if architecture == "resnet":
        return (
            ResNetPatchEncoder(input_dim, latent_dim),
            TanhPredictor(latent_dim, hidden_dim, hidden_layers),
        )
    raise ValueError(f"unknown legacy architecture: {architecture!r}")
