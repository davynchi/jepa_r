"""Post-hoc entity-subspace analysis (Sections 6-7 of the experiment spec).

The primary object of analysis is a *subspace*, not a single "entity neuron":
JEPA's objective is invariant to any orthogonal rotation of the latent space,
so individual coordinates are not identifiable. Everything here operates on
scatter matrices, projections, and probes that are rotation-aware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import eigh

from jepa.analysis.metrics import MetricValue, RidgeProbe, fit_ridge_probe


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().to(torch.float64).numpy()


@dataclass(frozen=True, slots=True)
class ScatterMatrices:
    within: np.ndarray
    between: np.ndarray
    num_entities: int


def compute_scatter_matrices(
    latents: torch.Tensor, entity: torch.Tensor, num_entities: int
) -> ScatterMatrices:
    """S_W = mean over entities of Cov(z | E=e); S_B = Cov over entity means."""
    z = _to_numpy(latents)
    e = entity.detach().cpu().numpy()
    dim = z.shape[1]
    within = np.zeros((dim, dim))
    active_means: list[np.ndarray] = []
    used = 0
    for entity_id in range(num_entities):
        mask = e == entity_id
        count = int(mask.sum())
        if count == 0:
            continue
        rows = z[mask]
        mean = rows.mean(axis=0)
        active_means.append(mean)
        if count > 1:
            centered = rows - mean
            within += (centered.T @ centered) / (count - 1)
        used += 1
    within = within / max(used, 1)
    if used > 1:
        means_matrix = np.stack(active_means, axis=0)
        centered_means = means_matrix - means_matrix.mean(axis=0)
        between = (centered_means.T @ centered_means) / (used - 1)
    else:
        between = np.zeros((dim, dim))
    return ScatterMatrices(within=within, between=between, num_entities=num_entities)


@dataclass(frozen=True, slots=True)
class GeneralizedEigenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def solve_generalized_eigenproblem(
    scatter: ScatterMatrices, *, epsilon: float
) -> GeneralizedEigenResult:
    """Solve S_B v = lambda (S_W + eps I) v with a stable symmetric eigensolver."""
    dim = scatter.within.shape[0]
    regularized_within = scatter.within + epsilon * np.eye(dim)
    symmetric_between = (scatter.between + scatter.between.T) / 2
    symmetric_within = (regularized_within + regularized_within.T) / 2
    eigenvalues, eigenvectors = eigh(symmetric_between, symmetric_within)
    order = np.argsort(eigenvalues)[::-1]
    return GeneralizedEigenResult(
        eigenvalues=eigenvalues[order], eigenvectors=eigenvectors[:, order]
    )


def entity_subspace_projection(eigenvectors: np.ndarray, k: int) -> np.ndarray:
    """P_E = V_k V_k^T, the orthogonal projector onto the top-k generalized eigenvectors."""
    top = eigenvectors[:, :k]
    # Generalized eigenvectors from eigh(A, B) are B-orthogonal, not Euclidean-orthogonal;
    # re-orthonormalize in the Euclidean sense so P is a genuine orthogonal projector.
    q, _ = np.linalg.qr(top)
    return q @ q.T


def project(latents: torch.Tensor, projection: np.ndarray) -> torch.Tensor:
    matrix = torch.as_tensor(projection, dtype=torch.float64)
    return (latents.to(torch.float64) @ matrix.T).to(latents.dtype)


@dataclass(frozen=True, slots=True)
class LinearClassifier:
    probe: RidgeProbe
    num_classes: int


def fit_entity_classifier(
    features: torch.Tensor, entity: torch.Tensor, num_classes: int, *, ridge: float
) -> LinearClassifier:
    """A linear classifier via ridge regression onto one-hot labels plus argmax."""
    one_hot = torch.nn.functional.one_hot(entity, num_classes).to(torch.float64)
    probe = fit_ridge_probe(features, one_hot, alpha=ridge)
    return LinearClassifier(probe=probe, num_classes=num_classes)


def classifier_accuracy(
    classifier: LinearClassifier, features: torch.Tensor, entity: torch.Tensor
) -> float:
    scores = classifier.probe.predict(features)
    predicted = scores.argmax(dim=-1)
    return (predicted == entity).to(torch.float64).mean().item()


def fit_context_regressor(
    features: torch.Tensor, context: torch.Tensor, *, ridge: float
) -> RidgeProbe:
    return fit_ridge_probe(features, context, alpha=ridge)


def context_r2_mean(
    probe: RidgeProbe, features: torch.Tensor, context: torch.Tensor, *, epsilon: float = 1e-8
) -> MetricValue:
    """Mean R^2 across context dimensions (each dimension scored independently)."""
    predictions = probe.predict(features).to(torch.float64)
    targets = context.to(torch.float64)
    scores: list[float] = []
    for dim in range(targets.shape[1]):
        target_dim = targets[:, dim]
        denom = (target_dim - target_dim.mean()).square().sum().item()
        if denom <= epsilon:
            continue
        numerator = (target_dim - predictions[:, dim]).square().sum().item()
        scores.append(1.0 - numerator / denom)
    if not scores:
        return MetricValue(None, "constant_target")
    return MetricValue(float(np.mean(scores)))


@dataclass(frozen=True, slots=True)
class CounterfactualInvariance:
    d_same: float
    d_diff: float
    q_entity: float


def counterfactual_invariance(
    projected_same_1: torch.Tensor,
    projected_same_2: torch.Tensor,
    projected_diff_1: torch.Tensor,
    projected_diff_2: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> CounterfactualInvariance:
    d_same = (projected_same_1 - projected_same_2).square().sum(dim=-1).mean().item()
    d_diff = (projected_diff_1 - projected_diff_2).square().sum(dim=-1).mean().item()
    return CounterfactualInvariance(
        d_same=d_same, d_diff=d_diff, q_entity=d_diff / (d_same + epsilon)
    )


def entity_selectivity_score(
    entity_accuracy_subspace: float,
    entity_accuracy_complement: float,
    context_r2_subspace: float | None,
    *,
    alpha: float = 1.0,
) -> float:
    leakage = alpha * max(0.0, context_r2_subspace) if context_r2_subspace is not None else 0.0
    return entity_accuracy_subspace - entity_accuracy_complement - leakage


@dataclass(frozen=True, slots=True)
class AutocorrelationResult:
    lags: tuple[int, ...]
    per_dimension: np.ndarray  # [dim, num_lags], normalized rho_j(tau)
    correlation_times: np.ndarray  # [dim]


def compute_autocorrelation(trajectories: torch.Tensor, *, max_lag: int) -> AutocorrelationResult:
    """Lagged autocorrelation per latent coordinate, averaged over trajectories.

    ``trajectories`` has shape [num_trajectories, T, dim].
    """
    z = _to_numpy(trajectories)
    n, length, dim = z.shape
    max_lag = min(max_lag, length - 1)
    centered = z - z.mean(axis=(0, 1), keepdims=True)
    variance = (centered**2).mean(axis=(0, 1)) + 1e-12
    curves = np.zeros((dim, max_lag + 1))
    for tau in range(max_lag + 1):
        if tau == 0:
            product = centered * centered
        else:
            product = centered[:, : length - tau, :] * centered[:, tau:, :]
        curves[:, tau] = product.mean(axis=(0, 1)) / variance
    correlation_times = np.ones(dim)
    threshold = 1.0 / math.e
    for j in range(dim):
        below = np.where(curves[j] <= threshold)[0]
        correlation_times[j] = float(below[0]) if below.size else float(max_lag)
    return AutocorrelationResult(
        lags=tuple(range(max_lag + 1)), per_dimension=curves, correlation_times=correlation_times
    )


def select_entity_subspace_dim(
    candidate_dims: tuple[int, ...],
    validation_entity_accuracy: dict[int, float],
) -> int:
    """Pick the k with the highest validation-only entity accuracy."""
    return max(candidate_dims, key=lambda k: validation_entity_accuracy[k])


@dataclass(frozen=True, slots=True)
class WhiteningTransform:
    """A train-set-only affine whitening map: z_white = (z - mean) @ inv_sqrt_cov^T."""

    mean: np.ndarray
    inv_sqrt_cov: np.ndarray


def fit_whitening(reference_latents: torch.Tensor, *, epsilon: float = 1e-6) -> WhiteningTransform:
    """Fit an isotropic (identity-covariance) whitening transform from train latents.

    Used to strip out a raw-scale/overall-norm confound before comparing
    D_same / D_diff across training conditions: two encoders can differ in
    D_same and D_diff purely because one produces larger-norm vectors, without
    differing in the *relative* separation between entities and contexts.
    """
    reference = _to_numpy(reference_latents)
    mean = reference.mean(axis=0, keepdims=True)
    centered = reference - mean
    cov = (centered.T @ centered) / max(reference.shape[0] - 1, 1)
    cov = (cov + cov.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    inv_sqrt = (
        eigenvectors @ np.diag(1.0 / np.sqrt(np.clip(eigenvalues, epsilon, None))) @ eigenvectors.T
    )
    return WhiteningTransform(mean=mean, inv_sqrt_cov=inv_sqrt)


def apply_whitening(transform: WhiteningTransform, latents: torch.Tensor) -> torch.Tensor:
    matrix = _to_numpy(latents)
    whitened = (matrix - transform.mean) @ transform.inv_sqrt_cov.T
    return torch.as_tensor(whitened, dtype=torch.float64)


@dataclass(frozen=True, slots=True)
class LatentSpectrum:
    """Full covariance diagnostics: not just effective rank, but the whole spectrum."""

    eigenvalues: np.ndarray  # descending, full spectrum
    trace_covariance: float
    per_dimension_std: np.ndarray
    effective_rank: float


def compute_latent_spectrum(latents: torch.Tensor) -> LatentSpectrum:
    matrix = _to_numpy(latents)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    cov = (centered.T @ centered) / max(matrix.shape[0] - 1, 1)
    cov = (cov + cov.T) / 2
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    trace = float(eigenvalues.sum())
    probabilities = eigenvalues / trace if trace > 0 else eigenvalues
    positive = probabilities[probabilities > 0]
    effective_rank = (
        float(np.exp(-(positive * np.log(positive + 1e-12)).sum())) if trace > 0 else 0.0
    )
    return LatentSpectrum(
        eigenvalues=eigenvalues,
        trace_covariance=trace,
        per_dimension_std=np.sqrt(np.clip(np.diagonal(cov), 0.0, None)),
        effective_rank=effective_rank,
    )


def mutual_information_entity(
    projected: torch.Tensor, entity: torch.Tensor, num_entities: int, *, num_bins: int = 20
) -> float:
    """Histogram-based I(projected; entity) in nats, no external MI-estimator
    dependency (sklearn isn't installed on the training server). ``projected``
    is a single continuous coordinate (e.g. the top generalized-eigenvector
    projection); quantile bin edges keep bins roughly equiprobable regardless
    of the coordinate's marginal shape, which a fixed-width histogram would
    not for a heavy-tailed or multimodal latent.

    Complements (not replaces) the counterfactual/probe metrics: MI catches
    nonlinear or non-monotonic dependence between a coordinate and the entity
    label that a linear probe or a simple variance ratio could miss.
    """
    values = _to_numpy(projected).reshape(-1)
    labels = entity.detach().cpu().numpy()
    n = values.shape[0]
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    quantiles = np.unique(quantiles)
    effective_bins = max(len(quantiles) - 1, 1)
    bin_index = np.clip(np.searchsorted(quantiles, values, side="right") - 1, 0, effective_bins - 1)
    joint_counts = np.zeros((effective_bins, num_entities))
    np.add.at(joint_counts, (bin_index, labels), 1.0)
    joint_p = joint_counts / n
    bin_p = joint_p.sum(axis=1, keepdims=True)
    entity_p = joint_p.sum(axis=0, keepdims=True)
    independent_p = bin_p @ entity_p
    nonzero = joint_p > 0
    return float(np.sum(joint_p[nonzero] * np.log(joint_p[nonzero] / independent_p[nonzero])))


def entity_label_entropy(entity: torch.Tensor, num_entities: int) -> float:
    """H(E) in nats, from empirical label frequencies -- the natural normalizer
    for :func:`mutual_information_entity` (I(z;E)/H(E) in [0, 1])."""
    labels = entity.detach().cpu().numpy()
    counts = np.bincount(labels, minlength=num_entities).astype(np.float64)
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    return float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))


def empirical_entity_predictability(entities: torch.Tensor, *, horizon: int) -> float:
    """P(E_{t+h} = E_t), measured directly from generated trajectories (not the
    theoretical Markov-chain formula) -- the quantity that actually matters for
    a horizon-h prediction objective, as opposed to the raw switch probability
    p_E alone. ``entities`` has shape [num_trajectories, T]."""
    if horizon >= entities.shape[1]:
        raise ValueError("horizon must be less than the trajectory length")
    source = entities[:, : entities.shape[1] - horizon]
    target = entities[:, horizon:]
    return float((source == target).float().mean().item())
