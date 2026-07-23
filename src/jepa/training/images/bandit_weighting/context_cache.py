"""Online latent context cache for dataset-scale bandit policies."""

from __future__ import annotations

from typing import Any

import torch


class LatentContextCache:
    def __init__(
        self,
        num_samples: int,
        context_dim: int,
        *,
        ema_beta: float = 0.1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_samples <= 0 or context_dim <= 0:
            raise ValueError("num_samples and context_dim must be positive")
        if not 0 < ema_beta <= 1:
            raise ValueError("ema_beta must be in (0, 1]")
        self.num_samples = num_samples
        self.context_dim = context_dim
        self.ema_beta = ema_beta
        self.device = torch.device(device)
        self.dtype = dtype
        self.contexts = torch.zeros((num_samples, context_dim), device=self.device, dtype=dtype)
        self.observation_counts = torch.zeros(num_samples, device=self.device, dtype=torch.int64)
        self.last_seen_steps = torch.full((num_samples,), -1, device=self.device, dtype=torch.int64)

    @property
    def known_mask(self) -> torch.Tensor:
        return self.observation_counts > 0

    @property
    def coverage(self) -> float:
        return float(self.known_mask.to(torch.float64).mean().item())

    def update(self, indices: torch.Tensor, contexts: torch.Tensor, *, step: int) -> None:
        if step < 0:
            raise ValueError("step must be non-negative")
        sample_indices = indices.detach().to(device=self.device, dtype=torch.long).flatten()
        values = contexts.detach().to(device=self.device, dtype=self.dtype)
        if values.ndim != 2 or values.shape[1] != self.context_dim:
            raise ValueError(f"contexts must have shape [batch, {self.context_dim}]")
        if sample_indices.numel() == 0 or sample_indices.numel() != values.shape[0]:
            raise ValueError("indices and contexts must have the same non-zero batch size")
        if sample_indices.min() < 0 or sample_indices.max() >= self.num_samples:
            raise ValueError("cache indices are out of bounds")
        if not torch.isfinite(values).all():
            raise ValueError("contexts must be finite")

        unique, inverse, counts = torch.unique(
            sample_indices, sorted=True, return_inverse=True, return_counts=True
        )
        means = torch.zeros(
            (unique.numel(), self.context_dim), device=self.device, dtype=self.dtype
        )
        means.index_add_(0, inverse, values)
        means /= counts.to(self.dtype).unsqueeze(1)
        seen = self.observation_counts[unique] > 0
        updated = means.clone()
        updated[seen] = (1.0 - self.ema_beta) * self.contexts[unique[seen]] + self.ema_beta * means[
            seen
        ]
        self.contexts[unique] = updated
        self.observation_counts[unique] += counts
        self.last_seen_steps[unique] = step

    def age(self, *, step: int) -> torch.Tensor:
        if step < 0:
            raise ValueError("step must be non-negative")
        return torch.where(
            self.known_mask,
            step - self.last_seen_steps,
            torch.full_like(self.last_seen_steps, step + 1),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "context_dim": self.context_dim,
            "ema_beta": self.ema_beta,
            "contexts": self.contexts.detach().cpu(),
            "observation_counts": self.observation_counts.detach().cpu(),
            "last_seen_steps": self.last_seen_steps.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["num_samples"]) != self.num_samples:
            raise ValueError("cache sample count does not match checkpoint")
        if int(state["context_dim"]) != self.context_dim:
            raise ValueError("cache context dimension does not match checkpoint")
        if float(state["ema_beta"]) != self.ema_beta:
            raise ValueError("cache EMA beta does not match checkpoint")
        contexts = torch.as_tensor(state["contexts"])
        counts = torch.as_tensor(state["observation_counts"])
        last_seen = torch.as_tensor(state["last_seen_steps"])
        if contexts.shape != self.contexts.shape:
            raise ValueError("checkpoint contexts shape does not match cache")
        if counts.shape != self.observation_counts.shape:
            raise ValueError("checkpoint observation count shape does not match cache")
        if last_seen.shape != self.last_seen_steps.shape:
            raise ValueError("checkpoint last-seen shape does not match cache")
        self.contexts.copy_(contexts.to(device=self.device, dtype=self.dtype))
        self.observation_counts.copy_(
            counts.to(device=self.device, dtype=self.observation_counts.dtype)
        )
        self.last_seen_steps.copy_(
            last_seen.to(device=self.device, dtype=self.last_seen_steps.dtype)
        )
