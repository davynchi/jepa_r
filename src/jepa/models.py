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


def sincos_position_table(num_positions: int, dimension: int) -> torch.Tensor:
    """Build a fixed one-dimensional sin/cos table using standard frequencies."""
    _require_positive("num_positions", num_positions)
    _require_positive("dimension", dimension)
    if dimension % 2:
        raise ValueError("position dimension must be even")
    positions = torch.arange(num_positions, dtype=torch.float64).unsqueeze(1)
    frequencies = torch.arange(0, dimension, 2, dtype=torch.float64) / dimension
    angles = positions * torch.pow(10000.0, -frequencies).unsqueeze(0)
    table = torch.empty(num_positions, dimension, dtype=torch.float64)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table.float()


class MaskedPatchPredictor(nn.Module):
    """Vectorized target-query predictor for local-patch masked JEPA."""

    def __init__(
        self,
        architecture: Architecture,
        *,
        num_patches: int,
        latent_dim: int,
        position_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_patches", num_patches),
            ("latent_dim", latent_dim),
            ("position_dim", position_dim),
        ):
            _require_positive(name, value)
        self.num_patches = num_patches
        self.latent_dim = latent_dim
        self.position_dim = position_dim
        self.register_buffer(
            "position_table", sincos_position_table(num_patches, position_dim), persistent=True
        )
        input_dim = num_patches * latent_dim + num_patches + position_dim
        if architecture == "linear":
            self.network: nn.Module = nn.Linear(input_dim, latent_dim)
        elif architecture == "nonlinear":
            self.network = _TanhMLP(
                input_dim, latent_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers
            )
        else:
            raise ValueError(f"unknown architecture: {architecture!r}")

    def forward(
        self,
        context_grid: torch.Tensor,
        visibility: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        if context_grid.ndim != 3 or context_grid.shape[1:] != (
            self.num_patches,
            self.latent_dim,
        ):
            raise ValueError("context_grid has an invalid shape")
        if visibility.shape != context_grid.shape[:2]:
            raise ValueError("visibility has an invalid shape")
        if target_indices.ndim != 2 or target_indices.shape[0] != context_grid.shape[0]:
            raise ValueError("target_indices has an invalid shape")
        if target_indices.dtype != torch.long:
            raise ValueError("target_indices must use torch.long")
        batch_size, target_count = target_indices.shape
        shared = torch.cat((context_grid.flatten(1), visibility.to(context_grid.dtype)), dim=1)
        shared = shared.unsqueeze(1).expand(batch_size, target_count, shared.shape[1])
        positions = self.get_buffer("position_table")[target_indices]
        queries = torch.cat((shared, positions.to(context_grid.dtype)), dim=-1)
        return self.network(queries)


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


def build_masked_model_pair(
    architecture: Architecture,
    *,
    patch_input_dim: int,
    num_patches: int,
    latent_dim: int,
    position_dim: int,
    hidden_dim: int = 64,
    hidden_layers: int = 1,
) -> tuple[nn.Module, MaskedPatchPredictor]:
    """Build a local patch encoder and explicit position-conditioned predictor."""
    if architecture == "linear":
        encoder: nn.Module = LinearEncoder(patch_input_dim, latent_dim)
    elif architecture == "nonlinear":
        encoder = TanhEncoder(patch_input_dim, latent_dim, hidden_dim, hidden_layers)
    else:
        raise ValueError(f"unknown architecture: {architecture!r}")
    predictor = MaskedPatchPredictor(
        architecture,
        num_patches=num_patches,
        latent_dim=latent_dim,
        position_dim=position_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    )
    return encoder, predictor
