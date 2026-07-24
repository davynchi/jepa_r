"""Run logging utilities for spatial I-JEPA experiments."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

TensorBoardSummaryWriter: Any
try:
    from torch.utils.tensorboard import SummaryWriter as TensorBoardSummaryWriter
except ModuleNotFoundError:  # pragma: no cover - depends on optional local install.
    TensorBoardSummaryWriter = None


_TENSORBOARD_SCALARS = {
    "train/loss",
    "train/lr",
    "train/weight_decay",
    "train/ema_momentum",
    "train/epoch_loss",
    "train/epoch_seconds",
    "train/elapsed_seconds",
    "eval/test_loss",
    "repr/effective_rank",
    "repr/trace_covariance",
    "repr/mean_latent_norm",
    "repr/eig_mass_top_1",
    "repr/eig_mass_top_4",
    "repr/eig_mass_top_8",
    "weighting/score_mean",
    "weighting/score_std",
    "weighting/prob_entropy_normalized",
    "weighting/prob_max",
    "weighting/effective_sample_size",
    "weighting/sampling_temperature",
    "weighting/target_effective_sample_size",
    "weighting/top_10pct_mass",
    "ras/richness_value",
    "ras/richness_logdet",
    "ras/richness_trace",
    "ras/richness_pr",
    "ras/richness_trace_penalty",
    "ras/richness_top_eigenvalue",
    "ras/predictive_spectral_energy",
    "ras/predictive_spectral_effective_rank",
    "ras/predictive_spectral_rank_01",
    "ras/predictive_spectral_sigma_max",
    "ras/predictive_spectral_sigma_mean",
    "ras/grad_richness_norm",
    "ras/positive_fraction",
    "ras/negative_fraction",
    "ras/score_granularity_batch",
    "ras/score_groups",
    "ras/alignment_cosine",
    "ras/loss_gradient_norm_mean",
    "ras/loss_gradient_norm_max",
    "coord/importance_min",
    "coord/importance_max",
    "coord/importance_std",
    "coord/covariance_top_eigenvalue",
    "coord/covariance_trace",
    "coord/transform_singular_min",
    "coord/transform_singular_max",
    "coord/dynamics_cold_start",
    "coord/dynamics_delta_mean",
    "coord/grad_coordinate_norm",
    "coord/score_positive_fraction",
    "bandit/raw_reward",
    "bandit/normalized_reward",
    "bandit/predicted_reward",
    "bandit/prediction_error",
    "bandit/reward_mean",
    "bandit/reward_std",
    "bandit/prediction_rmse",
    "bandit/cache_coverage",
    "bandit/num_updates",
    "bandit/posterior_mean_norm",
    "bandit/posterior_trace",
    "bandit/precision_condition",
    "diag/test_loss",
    "diag/entity_accuracy",
    "diag/class_accuracy",
    "diag/class_top5_accuracy",
    "diag/context_accuracy_mean",
    "diag/raw_q_entity",
    "diag/white_q_entity",
    "diag/mi_ratio",
}

_TENSORBOARD_HISTOGRAMS = {
    "hist/eigenvalues",
    "hist/weighting_scores",
    "hist/weighting_probabilities",
    "hist/diag_eigenvalues",
}


class SpatialRunLogger:
    """Write durable JSONL metrics and optional TensorBoard summaries."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        enable_tensorboard: bool = True,
        extra_tensorboard_scalars: Iterable[str] = (),
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.tensorboard_scalars = _TENSORBOARD_SCALARS | set(extra_tensorboard_scalars)
        self.writer: Any | None = None
        if enable_tensorboard and TensorBoardSummaryWriter is not None:
            self.writer = TensorBoardSummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

    def write_config(self, payload: Mapping[str, Any]) -> None:
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")

    def log(
        self,
        *,
        step: int,
        epoch: int,
        event: str,
        scalars: Mapping[str, float | int | bool | None],
        histograms: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        safe_scalars: dict[str, float | int | bool | None] = {}
        for name, value in scalars.items():
            if (
                value is not None
                and not isinstance(value, bool)
                and not math.isfinite(float(value))
            ):
                safe_scalars[name] = None
            else:
                safe_scalars[name] = value
        row: dict[str, Any] = {
            "time": time.time(),
            "step": step,
            "epoch": epoch,
            "event": event,
            "scalars": safe_scalars,
        }
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

        if self.writer is None:
            return
        for name, value in safe_scalars.items():
            if name not in self.tensorboard_scalars:
                continue
            if value is None or isinstance(value, bool):
                continue
            self.writer.add_scalar(name, float(value), step)
        for name, values in (histograms or {}).items():
            if name not in _TENSORBOARD_HISTOGRAMS:
                continue
            self.writer.add_histogram(name, values.detach().float().cpu(), step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def eigenvalue_scalars(eigenvalues: Any, *, top_k: int = 10) -> dict[str, float]:
    values = [float(v) for v in eigenvalues[:top_k]]
    scalars = {f"repr/top_eig_{index + 1:02d}": value for index, value in enumerate(values)}
    total = float(sum(float(v) for v in eigenvalues))
    if total > 0:
        for k in (1, 4, 8):
            scalars[f"repr/eig_mass_top_{k}"] = float(
                sum(float(v) for v in eigenvalues[:k]) / total
            )
    return scalars
