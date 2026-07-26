"""Per-sample surprise state and label-free sampling rules for video JEPA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

SamplingStrategy = Literal[
    "uniform_shuffle",
    "uniform_replacement",
    "soft_surprise",
    "warmup_learnable_surprise",
    "progress_quarantine_surprise",
]
ScoreUpdateMode = Literal["oracle_epochwise", "online_cached"]

ADAPTIVE_STRATEGIES: set[str] = {
    "soft_surprise",
    "warmup_learnable_surprise",
    "progress_quarantine_surprise",
}


@dataclass(frozen=True)
class VideoSamplingConfig:
    strategy: SamplingStrategy = "uniform_shuffle"
    score_update_mode: ScoreUpdateMode = "oracle_epochwise"
    warmup_epochs: int = 5
    uniform_mix: float = 0.4
    loss_ema_beta: float = 0.8
    progress_ema_beta: float = 0.8
    surprise_alpha: float = 1.0
    band_center: float = 0.60
    band_width: float = 0.18
    progress_boost: float = 2.0
    progress_margin: float = 0.0
    quarantine_percentile: float = 0.97
    min_weight: float = 0.25
    max_weight: float = 4.0

    def validate(self) -> None:
        valid = {
            "uniform_shuffle",
            "uniform_replacement",
            "soft_surprise",
            "warmup_learnable_surprise",
            "progress_quarantine_surprise",
        }
        if self.strategy not in valid:
            raise ValueError(f"unknown sampling strategy: {self.strategy}")
        if self.score_update_mode not in {"oracle_epochwise", "online_cached"}:
            raise ValueError(f"unknown score update mode: {self.score_update_mode}")
        for name, value in {
            "uniform_mix": self.uniform_mix,
            "loss_ema_beta": self.loss_ema_beta,
            "progress_ema_beta": self.progress_ema_beta,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.band_width <= 0.0:
            raise ValueError("band_width must be positive")
        if not 0.0 <= self.band_center <= 1.0:
            raise ValueError("band_center must be in [0,1]")
        if not 0.0 <= self.quarantine_percentile <= 1.0:
            raise ValueError("quarantine_percentile must be in [0,1]")
        if not 0.0 < self.min_weight <= self.max_weight:
            raise ValueError("expected 0 < min_weight <= max_weight")

    @property
    def is_adaptive(self) -> bool:
        return self.strategy in ADAPTIVE_STRATEGIES


def _aggregate_duplicate_updates(indices: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    indices = indices.detach().to(device="cpu", dtype=torch.long).flatten()
    values = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    if len(indices) != len(values):
        raise ValueError("indices and values must have equal length")
    if len(indices) == 0:
        return indices, values
    unique, inverse = torch.unique(indices, sorted=True, return_inverse=True)
    sums = torch.zeros(len(unique), dtype=torch.float64)
    counts = torch.zeros(len(unique), dtype=torch.float64)
    sums.scatter_add_(0, inverse, values)
    counts.scatter_add_(0, inverse, torch.ones_like(values))
    return unique, sums / counts.clamp_min(1.0)


def percentile_ranks(values: Tensor) -> Tensor:
    """Stable empirical percentiles in [0,1], preserving average ranks for ties."""

    values = values.to(torch.float64)
    if values.numel() <= 1:
        return torch.full_like(values, 0.5)
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) / (len(values) - 1)
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


class AdaptiveSampleState:
    """CPU-resident state whose entries correspond to immutable dataset indices."""

    def __init__(self, num_samples: int, config: VideoSamplingConfig) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        config.validate()
        self.num_samples = num_samples
        self.config = config
        self.loss_ema = torch.full((num_samples,), float("nan"), dtype=torch.float64)
        self.progress_ema = torch.zeros(num_samples, dtype=torch.float64)
        self.visits = torch.zeros(num_samples, dtype=torch.long)

    def state_dict(self) -> dict[str, Tensor | int]:
        return {
            "num_samples": self.num_samples,
            "loss_ema": self.loss_ema.clone(),
            "progress_ema": self.progress_ema.clone(),
            "visits": self.visits.clone(),
        }

    def load_state_dict(self, state: dict[str, Tensor | int]) -> None:
        if int(state["num_samples"]) != self.num_samples:
            raise ValueError("sampler state size mismatch")
        self.loss_ema.copy_(torch.as_tensor(state["loss_ema"], dtype=torch.float64))
        self.progress_ema.copy_(torch.as_tensor(state["progress_ema"], dtype=torch.float64))
        self.visits.copy_(torch.as_tensor(state["visits"], dtype=torch.long))

    def update(self, indices: Tensor, losses: Tensor) -> None:
        indices, losses = _aggregate_duplicate_updates(indices, losses)
        if len(indices) == 0:
            return
        if int(indices.min()) < 0 or int(indices.max()) >= self.num_samples:
            raise IndexError("sample index outside sampler state")

        seen = self.visits[indices] > 0
        old = self.loss_ema[indices]
        new = losses.clone()
        new[seen] = (
            self.config.loss_ema_beta * old[seen]
            + (1.0 - self.config.loss_ema_beta) * losses[seen]
        )

        relative_progress = torch.zeros_like(new)
        relative_progress[seen] = torch.clamp(
            (old[seen] - new[seen]) / old[seen].abs().clamp_min(1e-8),
            min=0.0,
        )
        old_progress = self.progress_ema[indices]
        updated_progress = (
            self.config.progress_ema_beta * old_progress
            + (1.0 - self.config.progress_ema_beta) * relative_progress
        )
        updated_progress[~seen] = 0.0

        self.loss_ema[indices] = new
        self.progress_ema[indices] = updated_progress
        self.visits[indices] += 1

    def _seen_percentiles(self) -> Tensor:
        seen = self.visits > 0
        result = torch.full((self.num_samples,), 0.5, dtype=torch.float64)
        if seen.any():
            result[seen] = percentile_ranks(self.loss_ema[seen])
        return result

    def _progress_scale(self) -> float:
        positive = self.progress_ema[self.progress_ema > 0.0]
        if len(positive) == 0:
            return 1.0
        return max(float(positive.median()), 1e-8)

    def weights(self, epoch: int) -> Tensor:
        strategy = self.config.strategy
        if strategy in {"uniform_shuffle", "uniform_replacement"}:
            return torch.ones(self.num_samples, dtype=torch.double)
        if epoch <= self.config.warmup_epochs or not (self.visits > 0).any():
            return torch.ones(self.num_samples, dtype=torch.double)

        seen = self.visits > 0
        percentiles = self._seen_percentiles()
        weights = torch.ones(self.num_samples, dtype=torch.float64)

        if strategy == "soft_surprise":
            median = self.loss_ema[seen].median().abs().clamp_min(1e-8)
            relative = torch.clamp(self.loss_ema[seen] / median, min=0.0, max=8.0)
            weights[seen] = 1.0 + self.config.surprise_alpha * relative
        else:
            q = percentiles[seen]
            band = torch.exp(-0.5 * ((q - self.config.band_center) / self.config.band_width) ** 2)
            if strategy == "warmup_learnable_surprise":
                scale = self._progress_scale()
                progress = torch.clamp(self.progress_ema[seen] / scale, min=0.0, max=4.0)
                weights[seen] = band * (1.0 + self.config.progress_boost * progress)
            elif strategy == "progress_quarantine_surprise":
                scale = self._progress_scale()
                progress = torch.clamp(self.progress_ema[seen] / scale, min=0.0, max=4.0)
                candidate = band * (1.0 + self.config.progress_boost * progress)
                quarantine = (q >= self.config.quarantine_percentile) & (
                    self.progress_ema[seen] <= self.config.progress_margin
                )
                candidate[quarantine] = self.config.min_weight
                weights[seen] = candidate
            else:  # pragma: no cover - config validation makes this unreachable
                raise ValueError(strategy)

        weights = weights.clamp(self.config.min_weight, self.config.max_weight)
        weights = weights / weights.mean().clamp_min(1e-12)
        weights = (
            self.config.uniform_mix * torch.ones_like(weights)
            + (1.0 - self.config.uniform_mix) * weights
        )
        return weights.to(torch.double)

    def diagnostics(self, weights: Tensor) -> dict[str, float]:
        weights = weights.detach().to(torch.float64).flatten()
        probabilities = weights / weights.sum().clamp_min(1e-12)
        ess = 1.0 / probabilities.square().sum().clamp_min(1e-12)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        normalized_entropy = entropy / torch.log(torch.tensor(float(self.num_samples)))
        seen = self.visits > 0
        return {
            "sampler_ess_fraction": float(ess / self.num_samples),
            "sampler_entropy_fraction": float(normalized_entropy),
            "sampler_weight_min": float(weights.min()),
            "sampler_weight_max": float(weights.max()),
            "score_seen_fraction": float(seen.to(torch.float64).mean()),
            "cached_loss_mean": float(self.loss_ema[seen].mean()) if seen.any() else float("nan"),
            "cached_progress_mean": (
                float(self.progress_ema[seen].mean()) if seen.any() else float("nan")
            ),
        }


def selection_diagnostics(
    selected_indices: Tensor,
    *,
    dataset_size: int,
    selected_bounce: Tensor | None = None,
    selected_speed: Tensor | None = None,
) -> dict[str, float]:
    indices = selected_indices.detach().to(torch.long).flatten().cpu()
    if len(indices) == 0:
        return {
            "dataset_coverage": 0.0,
            "repeat_fraction": 0.0,
            "selected_bounce_fraction": float("nan"),
            "selected_speed_mean": float("nan"),
        }
    unique = torch.unique(indices)
    result = {
        "dataset_coverage": float(len(unique) / dataset_size),
        "repeat_fraction": float(1.0 - len(unique) / len(indices)),
        "selected_bounce_fraction": float("nan"),
        "selected_speed_mean": float("nan"),
    }
    if selected_bounce is not None:
        result["selected_bounce_fraction"] = float(
            selected_bounce.detach().to(torch.float64).mean()
        )
    if selected_speed is not None:
        result["selected_speed_mean"] = float(
            selected_speed.detach().to(torch.float64).mean()
        )
    return result
