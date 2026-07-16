"""Representation diagnostics and closed-form linear probes."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from jepa.config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class MetricValue:
    value: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.reason is None):
            raise ValueError("a metric must contain either a value or a null reason")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("metric values must be finite")


@dataclass(frozen=True, slots=True)
class RepresentationMetrics:
    latent_std_mean: MetricValue
    effective_rank: MetricValue
    effective_rank_normalized: MetricValue
    top_eigen_fraction: MetricValue
    vector_norm_mean: MetricValue
    eigenvalues: tuple[float, ...] | None
    collapsed: bool
    collapse_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RidgeProbe:
    weights: torch.Tensor
    intercept: torch.Tensor
    feature_mean: torch.Tensor
    target_mean: torch.Tensor
    target_train_sum_squares: float
    alpha: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        matrix = _as_finite_matrix("features", features)
        if matrix.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"features have width {matrix.shape[1]}, expected {self.weights.shape[0]}"
            )
        predictions = matrix @ self.weights + self.intercept
        if not torch.isfinite(predictions).all():
            raise ValueError("ridge predictions contain non-finite values")
        return predictions


@dataclass(slots=True)
class WeightedMean:
    """Sample/element-weighted scalar aggregation for epoch metrics."""

    weighted_total: float = 0.0
    total_weight: int = 0

    def update(self, value: float, weight: int) -> None:
        if not math.isfinite(value):
            raise ValueError("weighted mean values must be finite")
        if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
            raise ValueError("weighted mean weights must be positive integers")
        updated_total = self.weighted_total + value * weight
        if not math.isfinite(updated_total):
            raise ValueError("weighted mean accumulation overflowed")
        self.weighted_total = updated_total
        self.total_weight += weight

    def compute(self) -> MetricValue:
        if self.total_weight == 0:
            return MetricValue(None, "no_samples")
        return MetricValue(self.weighted_total / self.total_weight)


def _cpu_float64(values: torch.Tensor) -> torch.Tensor:
    """Move first, then cast: MPS cannot perform a direct float64 conversion."""
    return values.detach().to(device="cpu").to(dtype=torch.float64)


def _as_finite_matrix(name: str, values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, dimensions]")
    if values.shape[1] == 0:
        raise ValueError(f"{name} must have at least one dimension")
    matrix = _cpu_float64(values)
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _null_representation_metrics(reason: str, vector_norm: MetricValue) -> RepresentationMetrics:
    missing = MetricValue(None, reason)
    return RepresentationMetrics(
        latent_std_mean=missing,
        effective_rank=missing,
        effective_rank_normalized=missing,
        top_eigen_fraction=missing,
        vector_norm_mean=vector_norm,
        eigenvalues=None,
        collapsed=True,
        collapse_reasons=(reason,),
    )


def _validate_evaluation_config(config: EvaluationConfig) -> None:
    values = {
        "covariance_epsilon": config.covariance_epsilon,
        "collapse_std_threshold": config.collapse_std_threshold,
        "collapse_rank_threshold": config.collapse_rank_threshold,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if config.covariance_epsilon == 0:
        raise ValueError("covariance_epsilon must be positive")
    if config.collapse_rank_threshold > 1:
        raise ValueError("collapse_rank_threshold must be <= 1")


def compute_representation_metrics(
    representations: torch.Tensor,
    config: EvaluationConfig | None = None,
) -> RepresentationMetrics:
    """Compute full-matrix covariance diagnostics in CPU float64."""
    config = config if config is not None else EvaluationConfig()
    _validate_evaluation_config(config)
    if representations.ndim != 2:
        raise ValueError("representations must have shape [samples, dimensions]")
    if representations.shape[1] == 0:
        raise ValueError("representations must have at least one dimension")
    matrix = _cpu_float64(representations)
    if not torch.isfinite(matrix).all():
        missing = MetricValue(None, "non_finite")
        return _null_representation_metrics("non_finite", missing)
    if matrix.shape[0] == 0:
        return _null_representation_metrics("no_samples", MetricValue(None, "no_samples"))

    vector_norm_value = torch.linalg.vector_norm(matrix, dim=1).mean().item()
    if not math.isfinite(vector_norm_value):
        missing = MetricValue(None, "non_finite")
        return _null_representation_metrics("non_finite", missing)
    vector_norm = MetricValue(vector_norm_value)
    if matrix.shape[0] < 2:
        return _null_representation_metrics("insufficient_samples", vector_norm)

    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (matrix.shape[0] - 1)
    covariance = (covariance + covariance.T) / 2
    variances = torch.diagonal(covariance).clamp_min(0)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    if not torch.isfinite(variances).all() or not torch.isfinite(eigenvalues).all():
        return _null_representation_metrics("non_finite", vector_norm)

    eigenvalue_sum = eigenvalues.sum()
    if not torch.isfinite(eigenvalue_sum):
        return _null_representation_metrics("non_finite", vector_norm)
    if eigenvalue_sum <= config.covariance_epsilon:
        return _null_representation_metrics("zero_spectrum", vector_norm)

    probabilities = eigenvalues / eigenvalue_sum
    positive_probabilities = probabilities[probabilities > 0]
    entropy = -(positive_probabilities * positive_probabilities.log()).sum()
    effective_rank = entropy.exp().item()
    normalized_rank = effective_rank / matrix.shape[1]
    latent_std_mean = variances.sqrt().mean().item()
    top_eigen_fraction = probabilities.max().item()
    if not all(
        math.isfinite(value)
        for value in (effective_rank, normalized_rank, latent_std_mean, top_eigen_fraction)
    ):
        return _null_representation_metrics("non_finite", vector_norm)

    collapse_reasons: list[str] = []
    if latent_std_mean < config.collapse_std_threshold:
        collapse_reasons.append("low_latent_std")
    if normalized_rank < config.collapse_rank_threshold:
        collapse_reasons.append("low_effective_rank")

    return RepresentationMetrics(
        latent_std_mean=MetricValue(latent_std_mean),
        effective_rank=MetricValue(effective_rank),
        effective_rank_normalized=MetricValue(normalized_rank),
        top_eigen_fraction=MetricValue(top_eigen_fraction),
        vector_norm_mean=vector_norm,
        eigenvalues=tuple(eigenvalues.tolist()),
        collapsed=bool(collapse_reasons),
        collapse_reasons=tuple(collapse_reasons),
    )


def global_gradient_norm(
    parameters: Iterable[nn.Parameter],
    *,
    missing_reason: str = "missing_gradient",
) -> MetricValue:
    """Compute a global L2 norm without mutating or clipping gradients."""
    squared_norm = torch.zeros((), dtype=torch.float64)
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        gradient = _cpu_float64(parameter.grad)
        if not torch.isfinite(gradient).all():
            return MetricValue(None, "non_finite")
        squared_norm += gradient.square().sum()
        if not torch.isfinite(squared_norm):
            return MetricValue(None, "non_finite")
    if not found:
        return MetricValue(None, missing_reason)
    return MetricValue(squared_norm.sqrt().item())


def fit_ridge_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
) -> RidgeProbe:
    """Fit centered multi-output ridge with an unregularized intercept."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be finite and positive")
    feature_matrix = _as_finite_matrix("features", features)
    target_matrix = _as_finite_matrix("targets", targets)
    if feature_matrix.shape[0] == 0:
        raise ValueError("ridge fitting requires at least one sample")
    if feature_matrix.shape[0] != target_matrix.shape[0]:
        raise ValueError("features and targets must contain the same number of samples")

    feature_mean = feature_matrix.mean(dim=0)
    target_mean = target_matrix.mean(dim=0)
    centered_features = feature_matrix - feature_mean
    centered_targets = target_matrix - target_mean
    gram = centered_features.T @ centered_features
    regularized = gram + alpha * torch.eye(gram.shape[0], dtype=torch.float64)
    right_hand_side = centered_features.T @ centered_targets
    if not torch.isfinite(regularized).all() or not torch.isfinite(right_hand_side).all():
        raise ValueError("ridge normal equations contain non-finite values")
    factor = torch.linalg.cholesky(regularized)
    weights = torch.cholesky_solve(right_hand_side, factor)
    intercept = target_mean - feature_mean @ weights
    target_train_sum_squares = centered_targets.square().sum().item()
    return RidgeProbe(
        weights=weights,
        intercept=intercept,
        feature_mean=feature_mean,
        target_mean=target_mean,
        target_train_sum_squares=target_train_sum_squares,
        alpha=alpha,
    )


def ridge_r2_score(
    probe: RidgeProbe,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> MetricValue:
    """Evaluate aggregate multi-output R² around the frozen train target mean."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("R² epsilon must be finite and positive")
    target_matrix = _as_finite_matrix("targets", targets)
    predictions = probe.predict(features)
    if predictions.shape != target_matrix.shape:
        raise ValueError(
            f"prediction shape {tuple(predictions.shape)} does not match "
            f"target shape {tuple(target_matrix.shape)}"
        )
    if probe.target_train_sum_squares <= epsilon:
        return MetricValue(None, "constant_target")
    denominator = (target_matrix - probe.target_mean).square().sum()
    if not torch.isfinite(denominator):
        return MetricValue(None, "non_finite")
    if denominator <= epsilon:
        return MetricValue(None, "constant_target")
    numerator = (target_matrix - predictions).square().sum()
    if not torch.isfinite(numerator):
        return MetricValue(None, "non_finite")
    return MetricValue((1.0 - numerator / denominator).item())
