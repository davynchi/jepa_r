"""Stable view-factorization diagnostics without arbitrary spectral bands."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _matrix(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError("features must be a rank-2 tensor")
    result = value.detach().cpu().to(torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError("features contain non-finite values")
    return result


def _covariance(centered: torch.Tensor) -> torch.Tensor:
    return centered.T @ centered / max(centered.shape[0] - 1, 1)


def _effective_rank(values: torch.Tensor, epsilon: float) -> float:
    values = values.clamp_min(0)
    total = values.sum()
    if float(total) <= epsilon:
        return 0.0
    probabilities = values / total
    entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()
    return float(entropy.exp())


@dataclass(frozen=True)
class ViewFactorization:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    signal_covariance: torch.Tensor
    perturbation_covariance: torch.Tensor
    total_covariance: torch.Tensor
    whitening_floor: float


def fit_view_factorization(
    views: torch.Tensor,
    *,
    relative_floor: float = 1e-5,
    absolute_floor: float = 1e-10,
) -> ViewFactorization:
    """Decompose representation energy into image signal and mask perturbation.

    ``views`` has shape ``[num_views, num_images, latent_dim]``. Eigenvalues are
    generalized signal-to-total ratios and therefore lie in ``[0, 1]`` up to
    numerical error. This avoids dividing the spectrum into unstable equal bands.
    """
    if views.ndim != 3 or views.shape[0] < 2 or views.shape[1] < 2:
        raise ValueError("views must have shape [V, N, D] with V,N >= 2")
    matrix = views.detach().cpu().to(torch.float64)
    if not torch.isfinite(matrix).all():
        raise ValueError("views contain non-finite values")

    image_means = matrix.mean(dim=0)
    centered_means = image_means - image_means.mean(dim=0)
    signal = _covariance(centered_means)

    num_views, num_images, _ = matrix.shape
    residuals = matrix - image_means.unsqueeze(0)
    residuals = residuals.reshape(-1, matrix.shape[-1])
    perturbation = residuals.T @ residuals / max(num_images * (num_views - 1), 1)
    signal = (signal + signal.T) / 2
    perturbation = (perturbation + perturbation.T) / 2
    total = signal + perturbation

    total_values, total_vectors = torch.linalg.eigh(total)
    scale = max(float(total_values.max()), absolute_floor)
    floor = max(relative_floor * scale, absolute_floor)
    inverse_sqrt = total_vectors @ torch.diag(total_values.clamp_min(floor).rsqrt()) @ total_vectors.T
    whitened_signal = inverse_sqrt @ signal @ inverse_sqrt
    whitened_signal = (whitened_signal + whitened_signal.T) / 2
    values, vectors = torch.linalg.eigh(whitened_signal)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp(0, 1)
    # Columns map whitened coordinates back to the original latent coordinates.
    basis = inverse_sqrt @ vectors[:, order]
    return ViewFactorization(values, basis, signal, perturbation, total, floor)


def factorization_metrics(
    factorization: ViewFactorization,
    *,
    invariant_threshold: float = 0.9,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    values = factorization.eigenvalues
    perturbation_ratios = 1 - values
    signal_trace = float(torch.trace(factorization.signal_covariance))
    perturbation_trace = float(torch.trace(factorization.perturbation_covariance))
    return {
        "invariance_mean": float(values.mean()),
        "invariance_min": float(values.min()),
        "invariance_effective_rank": _effective_rank(values, epsilon),
        "invariant_dimension_090": float((values >= invariant_threshold).sum()),
        "perturbation_mean": float(perturbation_ratios.mean()),
        "perturbation_effective_rank": _effective_rank(perturbation_ratios, epsilon),
        "signal_to_perturbation_trace": signal_trace / max(perturbation_trace, epsilon),
        "signal_trace": signal_trace,
        "perturbation_trace": perturbation_trace,
        "latent_effective_rank": _effective_rank(
            torch.linalg.eigvalsh(factorization.total_covariance), epsilon
        ),
        "condition_number_regularized": float(
            torch.linalg.eigvalsh(factorization.total_covariance).max()
            / factorization.whitening_floor
        ),
    }


def supervised_alignment_metrics(
    image_features: torch.Tensor,
    labels: torch.Tensor,
    factorization: ViewFactorization,
    *,
    maximum_rank: int = 32,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Audit whether mask-invariant directions carry class information.

    Labels are deliberately confined to this audit and never used to fit the
    label-free factorization.
    """
    features = _matrix(image_features)
    labels = labels.detach().cpu().to(torch.long)
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must align with image_features")
    centered = features - features.mean(dim=0)
    total = _covariance(centered)
    between = torch.zeros_like(total)
    for label in labels.unique(sorted=True):
        selected = centered[labels == label]
        if selected.numel() == 0:
            continue
        mean = selected.mean(dim=0)
        between += selected.shape[0] * torch.outer(mean, mean)
    between /= max(features.shape[0] - 1, 1)

    total_values, total_vectors = torch.linalg.eigh(total)
    floor = max(float(total_values.max()) * 1e-5, 1e-10)
    inverse_sqrt = total_vectors @ torch.diag(total_values.clamp_min(floor).rsqrt()) @ total_vectors.T
    fisher = inverse_sqrt @ between @ inverse_sqrt
    fisher = (fisher + fisher.T) / 2
    class_values, class_vectors = torch.linalg.eigh(fisher)
    order = torch.argsort(class_values, descending=True)
    class_values = class_values[order].clamp_min(0)
    class_rank_limit = max(int(labels.unique().numel()) - 1, 0)
    rank = min(
        maximum_rank,
        class_rank_limit,
        int((class_values > epsilon).sum()),
        features.shape[1],
    )
    if rank == 0:
        return {
            "class_invariant_overlap": 0.0,
            "class_weighted_invariance": 0.0,
            "class_signal_effective_rank": 0.0,
            "audit_rank": 0.0,
        }

    class_basis = inverse_sqrt @ class_vectors[:, order[:rank]]
    class_basis, _ = torch.linalg.qr(class_basis, mode="reduced")
    invariant_rank = max(
        rank,
        int((factorization.eigenvalues >= 0.9).sum()),
    )
    invariant_basis = factorization.eigenvectors[:, :invariant_rank]
    invariant_basis, _ = torch.linalg.qr(invariant_basis, mode="reduced")
    overlap = torch.linalg.matrix_norm(class_basis.T @ invariant_basis).square() / rank

    # Measure mask invariance directly along class-discriminative directions.
    numerator = torch.diagonal(class_basis.T @ factorization.signal_covariance @ class_basis)
    denominator = torch.diagonal(class_basis.T @ factorization.total_covariance @ class_basis)
    ratios = numerator / denominator.clamp_min(epsilon)
    weights = class_values[:rank]
    weighted = (ratios * weights).sum() / weights.sum().clamp_min(epsilon)
    return {
        "class_invariant_overlap": float(overlap.clamp(0, 1)),
        "class_weighted_invariance": float(weighted.clamp(0, 1)),
        "class_signal_effective_rank": _effective_rank(weights, epsilon),
        "audit_rank": float(rank),
    }


def principal_subspace_similarity(
    current: ViewFactorization,
    previous: ViewFactorization,
    *,
    rank: int,
) -> float:
    rank = min(rank, current.eigenvectors.shape[1], previous.eigenvectors.shape[1])
    if rank <= 0:
        raise ValueError("rank must be positive")
    current_basis, _ = torch.linalg.qr(current.eigenvectors[:, :rank], mode="reduced")
    previous_basis, _ = torch.linalg.qr(previous.eigenvectors[:, :rank], mode="reduced")
    return float(torch.linalg.matrix_norm(current_basis.T @ previous_basis).square() / rank)


def lda_spectrum_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    relative_floor: float = 1e-5,
    absolute_floor: float = 1e-10,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Return scale-controlled LDA diagnostics for a labelled embedding sample."""
    matrix = _matrix(features)
    labels = labels.detach().cpu().to(torch.long)
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("labels must align with features")
    centered = matrix - matrix.mean(dim=0)
    total = _covariance(centered)
    within = torch.zeros_like(total)
    between = torch.zeros_like(total)
    classes = labels.unique(sorted=True)
    for label in classes:
        selected = matrix[labels == label]
        class_centered = selected - selected.mean(dim=0)
        within += class_centered.T @ class_centered
        mean_offset = selected.mean(dim=0) - matrix.mean(dim=0)
        between += selected.shape[0] * torch.outer(mean_offset, mean_offset)
    denominator = max(matrix.shape[0] - 1, 1)
    within /= denominator
    between /= denominator

    total_values, total_vectors = torch.linalg.eigh(total)
    floor = max(float(total_values.max()) * relative_floor, absolute_floor)
    inverse_sqrt = total_vectors @ torch.diag(total_values.clamp_min(floor).rsqrt()) @ total_vectors.T
    discriminative = inverse_sqrt @ between @ inverse_sqrt
    discriminative = (discriminative + discriminative.T) / 2
    values = torch.linalg.eigvalsh(discriminative).flip(0).clamp(0, 1)
    class_rank = min(max(int(classes.numel()) - 1, 0), values.numel())
    signal = values[:class_rank]
    signal_sum = float(signal.sum())
    total_trace = float(torch.trace(total))
    between_trace = float(torch.trace(between))
    within_trace = float(torch.trace(within))
    top10 = min(10, class_rank)
    top32 = min(32, class_rank)
    return {
        "lda_discriminative_trace": signal_sum,
        "lda_effective_rank": _effective_rank(signal, epsilon),
        "lda_top1_fraction": float(signal[0] / max(signal_sum, epsilon))
        if class_rank else 0.0,
        "lda_top10_fraction": float(signal[:top10].sum() / max(signal_sum, epsilon))
        if class_rank else 0.0,
        "lda_top32_fraction": float(signal[:top32].sum() / max(signal_sum, epsilon))
        if class_rank else 0.0,
        "between_total_trace_ratio": between_trace / max(total_trace, epsilon),
        "between_within_trace_ratio": between_trace / max(within_trace, epsilon),
        "latent_effective_rank": _effective_rank(total_values.clamp_min(0), epsilon),
        "num_samples": float(matrix.shape[0]),
        "num_classes_present": float(classes.numel()),
        "regularized_condition_number": float(total_values.max() / floor),
    }


__all__ = [
    "ViewFactorization",
    "factorization_metrics",
    "fit_view_factorization",
    "lda_spectrum_metrics",
    "principal_subspace_similarity",
    "supervised_alignment_metrics",
]
