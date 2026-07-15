"""Linear and smooth nonlinear encoder/predictor families."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

Architecture = Literal["linear", "nonlinear"]


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
    raise ValueError(f"unknown architecture: {architecture!r}")
