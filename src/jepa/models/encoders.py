"""Linear, smooth nonlinear, and image encoder/predictor families."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

Architecture = Literal["linear", "nonlinear", "cnn", "resnet"]


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


class LinearEncoder(nn.Module):
    """A strictly affine encoder."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        _require_positive("input_dim", input_dim)
        _require_positive("latent_dim", latent_dim)
        self.projection = nn.Linear(input_dim, latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class LinearPredictor(nn.Module):
    """A strictly affine latent predictor."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        _require_positive("latent_dim", latent_dim)
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.projection(latents)


class _TanhMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        _require_positive("input_dim", input_dim)
        _require_positive("output_dim", output_dim)
        _require_positive("hidden_dim", hidden_dim)
        _require_positive("hidden_layers", hidden_layers)

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.Tanh()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TanhEncoder(_TanhMLP):
    """An encoder with one or more smooth tanh hidden layers."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__(input_dim, latent_dim, hidden_dim, hidden_layers)


class TanhPredictor(_TanhMLP):
    """A latent predictor matched to the nonlinear encoder family."""

    def __init__(self, latent_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__(latent_dim, latent_dim, hidden_dim, hidden_layers)


class CNNEncoder(nn.Module):
    """A small strided-conv encoder for square RGB images passed in flattened.

    Takes the same flat ``[..., input_dim]`` tensor as every other encoder in
    this module (so it drops into the vector/image worlds' shared duck-typed
    training loop unmodified) and internally reshapes to
    ``[channels, side, side]`` before convolving. ``side`` is inferred from
    ``input_dim`` and ``channels``, so this only accepts perfect-square image
    inputs (e.g. Shapes3D's 3x64x64 = 12288).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        channels: int = 3,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        _require_positive("input_dim", input_dim)
        _require_positive("latent_dim", latent_dim)
        side_float = math.sqrt(input_dim / channels)
        side = round(side_float)
        if channels * side * side != input_dim:
            raise ValueError(
                f"input_dim ({input_dim}) is not channels ({channels}) x a square "
                "image side; CNNEncoder only supports square RGB-style inputs"
            )
        if side % 8 != 0:
            raise ValueError(f"image side ({side}) must be divisible by 8 for this CNN encoder")
        self.channels = channels
        self.side = side
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        reduced_side = side // 8
        self.fc = nn.Linear(hidden_channels * 2 * reduced_side * reduced_side, latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_shape = inputs.shape[:-1]
        images = inputs.reshape(-1, self.channels, self.side, self.side)
        features = self.conv(images).reshape(images.shape[0], -1)
        latents = self.fc(features)
        return latents.reshape(*batch_shape, -1)


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
    """A compact ResNet-style encoder for flattened RGB image patches.

    The spatial I-JEPA code passes one flattened patch at a time, with shape
    ``[..., input_dim]``. This encoder reshapes each vector back to a small RGB
    image, applies residual convolutional blocks, global-average-pools, and
    projects to the patch latent.
    """

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


def build_model_pair(
    architecture: Architecture,
    *,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int = 64,
    hidden_layers: int = 1,
) -> tuple[nn.Module, nn.Module]:
    """Build a matched encoder and predictor without hidden policy behavior."""
    if architecture == "linear":
        return LinearEncoder(input_dim, latent_dim), LinearPredictor(latent_dim)
    if architecture == "nonlinear":
        return (
            TanhEncoder(input_dim, latent_dim, hidden_dim, hidden_layers),
            TanhPredictor(latent_dim, hidden_dim, hidden_layers),
        )
    if architecture == "cnn":
        # The predictor operates purely in latent space (latent_dim -> latent_dim),
        # so it's the same tanh-MLP family as "nonlinear" -- only the encoder needs
        # to know about image structure.
        return (
            CNNEncoder(input_dim, latent_dim),
            TanhPredictor(latent_dim, hidden_dim, hidden_layers),
        )
    if architecture == "resnet":
        return (
            ResNetPatchEncoder(input_dim, latent_dim),
            TanhPredictor(latent_dim, hidden_dim, hidden_layers),
        )
    raise ValueError(f"unknown architecture: {architecture!r}")
