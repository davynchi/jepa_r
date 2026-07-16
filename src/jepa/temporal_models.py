"""Model families for the entity/context temporal experiment.

The "standard" JEPA reuses :func:`jepa.models.build_model_pair` unchanged: an
encoder maps a single observation to a latent, and a predictor maps the
context latent to a predicted target latent. The "hierarchical" model adds an
explicit ``z = (z_E, z_C)`` split with two prediction heads, per Section 5.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from jepa.models import Architecture, build_model_pair


@dataclass(slots=True)
class HierarchicalOutputs:
    latent: torch.Tensor
    entity_latent: torch.Tensor
    context_latent: torch.Tensor
    predicted_entity: torch.Tensor
    predicted_context: torch.Tensor


class HierarchicalEncoderPredictor(nn.Module):
    """A shared encoder with an explicit z_E/z_C split and two predictor heads."""

    def __init__(
        self,
        architecture: Architecture,
        *,
        input_dim: int,
        latent_dim: int,
        entity_latent_dim: int,
        context_latent_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        if entity_latent_dim + context_latent_dim != latent_dim:
            raise ValueError("entity_latent_dim + context_latent_dim must equal latent_dim")
        self.entity_latent_dim = entity_latent_dim
        self.context_latent_dim = context_latent_dim
        encoder, _ = build_model_pair(
            architecture,
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
        self.encoder = encoder
        # Predictor heads consume the full latent z_t, per the specification.
        # build_model_pair's predictor half assumes latent_dim -> latent_dim,
        # so the heads (latent_dim -> entity/context_latent_dim) are built
        # directly using the same linear/tanh-MLP family instead.
        self.entity_head = _build_head(
            architecture, latent_dim, entity_latent_dim, hidden_dim, hidden_layers
        )
        self.context_head = _build_head(
            architecture, latent_dim, context_latent_dim, hidden_dim, hidden_layers
        )

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def split(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return latent[..., : self.entity_latent_dim], latent[..., self.entity_latent_dim :]

    def forward(self, observation: torch.Tensor) -> HierarchicalOutputs:
        latent = self.encode(observation)
        entity_latent, context_latent = self.split(latent)
        return HierarchicalOutputs(
            latent=latent,
            entity_latent=entity_latent,
            context_latent=context_latent,
            predicted_entity=self.entity_head(latent),
            predicted_context=self.context_head(latent),
        )


def _build_head(
    architecture: Architecture,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
) -> nn.Module:
    if architecture == "linear":
        return nn.Linear(input_dim, output_dim)
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(current, hidden_dim), nn.Tanh()))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)
