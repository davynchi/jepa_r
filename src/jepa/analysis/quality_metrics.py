"""Versioned Shapes3D representation-quality metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias, cast

import torch

from jepa.analysis.metrics import MetricValue, fit_ridge_probe
from jepa.analysis.spatial_decomposition import (
    Panel,
    SubspacePartition,
    assert_corresponding_partitions,
)


class MetricDirection(str, Enum):
    LOWER = "lower"
    HIGHER = "higher"
    UNKNOWN = "unknown"


class MetricRole(str, Enum):
    CANDIDATE = "candidate"
    IDENTITY = "identity"
    VALIDITY = "validity"
    SUPERVISED_SECONDARY = "supervised_secondary"


class CostTier(str, Enum):
    CHEAP = "cheap"
    EXPENSIVE = "expensive"


class CellState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class NullReason(str, Enum):
    NON_FINITE = "non_finite"
    ZERO_DENOMINATOR = "zero_denominator"
    DEGENERATE_COVARIANCE = "degenerate_covariance"
    DEGENERATE_PARTITION = "degenerate_partition"
    NO_PREVIOUS_CHECKPOINT = "no_previous_checkpoint"
    NO_SUBSPACE_MOVEMENT = "no_subspace_movement"
    ZERO_JACOBIAN = "zero_jacobian"
    ZERO_GRADIENT_ENERGY = "zero_gradient_energy"
    ZERO_PERTURBATION_ENERGY = "zero_perturbation_energy"
    INSUFFICIENT_NONZERO_PERTURBATIONS = "insufficient_nonzero_perturbations"
    INVALID_NEGATIVE_MI = "invalid_negative_mi"
    NOT_SCHEDULED = "not_scheduled"
    NOT_TESTED_BUDGET = "not_tested_budget"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    q_number: str
    direction: MetricDirection
    role: MetricRole
    family: str
    cost_tier: CostTier
    milestone: int
    primary_eligible: bool
    version: str = "1"


def _spec(
    name: str,
    q: str,
    direction: MetricDirection,
    role: MetricRole,
    family: str,
    tier: CostTier = CostTier.CHEAP,
    milestone: int = 1,
) -> MetricSpec:
    return MetricSpec(
        name,
        q,
        direction,
        role,
        family,
        tier,
        milestone,
        role is MetricRole.CANDIDATE,
    )


METRIC_SPECS: tuple[MetricSpec, ...] = (
    _spec(
        "q1_cross_covariance", "1", MetricDirection.LOWER, MetricRole.CANDIDATE, "cross-covariance"
    ),
    _spec(
        "q2_projector_interaction",
        "2",
        MetricDirection.LOWER,
        MetricRole.IDENTITY,
        "cross-covariance",
    ),
    _spec(
        "q3_projector_overlap",
        "3",
        MetricDirection.LOWER,
        MetricRole.VALIDITY,
        "partition-validity",
    ),
    _spec(
        "q4_partition_incompleteness",
        "4",
        MetricDirection.LOWER,
        MetricRole.VALIDITY,
        "partition-validity",
    ),
    _spec(
        "q5_subspace_similarity",
        "5",
        MetricDirection.HIGHER,
        MetricRole.IDENTITY,
        "temporal-stability",
    ),
    _spec(
        "q6_subspace_distance",
        "6",
        MetricDirection.LOWER,
        MetricRole.CANDIDATE,
        "temporal-stability",
    ),
    _spec(
        "q7_cross_jacobian_energy",
        "7",
        MetricDirection.LOWER,
        MetricRole.CANDIDATE,
        "jacobian-simplicity",
        CostTier.EXPENSIVE,
        2,
    ),
    _spec(
        "q8_virtual_sensitivity_magnitude",
        "8",
        MetricDirection.LOWER,
        MetricRole.CANDIDATE,
        "virtual-update",
        CostTier.EXPENSIVE,
        2,
    ),
    _spec(
        "q9_subspace_velocity",
        "9",
        MetricDirection.LOWER,
        MetricRole.CANDIDATE,
        "temporal-stability",
    ),
    _spec(
        "q10_weighted_shape_entropy", "10", MetricDirection.HIGHER, MetricRole.CANDIDATE, "entropy"
    ),
    _spec(
        "q11_cross_subspace_gaussian_mi",
        "11",
        MetricDirection.LOWER,
        MetricRole.CANDIDATE,
        "information-coupling",
    ),
    _spec(
        "q12_absolute_entropy_change", "12", MetricDirection.LOWER, MetricRole.CANDIDATE, "entropy"
    ),
    _spec(
        "q13_realized_surprise_locality",
        "13",
        MetricDirection.HIGHER,
        MetricRole.CANDIDATE,
        "factorization-locality",
    ),
    _spec(
        "q14_gradient_locality",
        "14",
        MetricDirection.HIGHER,
        MetricRole.CANDIDATE,
        "gradient-locality",
        CostTier.EXPENSIVE,
        2,
    ),
    _spec(
        "q15_virtual_interference",
        "15",
        MetricDirection.LOWER,
        MetricRole.CANDIDATE,
        "virtual-update",
        CostTier.EXPENSIVE,
        2,
    ),
    _spec(
        "q16_entity_consistency",
        "16",
        MetricDirection.LOWER,
        MetricRole.SUPERVISED_SECONDARY,
        "semantic-control",
    ),
    _spec(
        "q17_mask_transformation_residual",
        "17",
        MetricDirection.LOWER,
        MetricRole.CANDIDATE,
        "transformation",
    ),
    _spec(
        "q18_mask_perturbation_concentration",
        "18",
        MetricDirection.HIGHER,
        MetricRole.CANDIDATE,
        "transformation",
    ),
    _spec(
        "q19a_jacobian_effective_rank",
        "19a",
        MetricDirection.UNKNOWN,
        MetricRole.CANDIDATE,
        "jacobian-simplicity",
        CostTier.EXPENSIVE,
        2,
    ),
    _spec(
        "q19b_jacobian_density_tau_1em3",
        "19b",
        MetricDirection.LOWER,
        MetricRole.CANDIDATE,
        "jacobian-simplicity",
        CostTier.EXPENSIVE,
        2,
    ),
    _spec(
        "q20_reconstruction_nmse",
        "20",
        MetricDirection.LOWER,
        MetricRole.CANDIDATE,
        "decoder-simplicity",
    ),
)


def validate_metric_specs(specs: Sequence[MetricSpec] = METRIC_SPECS) -> None:
    identities: set[tuple[str, str]] = set()
    for spec in specs:
        identity = (spec.name, spec.version)
        if identity in identities:
            raise ValueError(f"duplicate metric identity: {identity}")
        identities.add(identity)
        if spec.primary_eligible and spec.role is not MetricRole.CANDIDATE:
            raise ValueError(f"non-candidate metric cannot be primary: {spec.name}")


validate_metric_specs()
METRIC_SPEC_BY_NAME = MappingProxyType({spec.name: spec for spec in METRIC_SPECS})

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple[JsonScalar, ...]


def _validate_diagnostic(value: object) -> JsonValue:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("diagnostic floats must be finite")
        return value
    if isinstance(value, tuple):
        validated = tuple(_validate_diagnostic(item) for item in value)
        if any(isinstance(item, tuple) for item in validated):
            raise TypeError("nested diagnostic tuples are not supported")
        return cast(tuple[JsonScalar, ...], validated)
    raise TypeError("diagnostics must contain immutable JSON-safe values")


@dataclass(frozen=True, slots=True)
class QualityMetricResult:
    metric_name: str
    metric_value: MetricValue
    panel: Panel | None = None
    band_slot: int | None = None
    diagnostics: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.metric_name not in METRIC_SPEC_BY_NAME:
            raise ValueError(f"unregistered metric: {self.metric_name}")
        if self.band_slot is not None and self.band_slot < 0:
            raise ValueError("band_slot must be non-negative")
        validated = {key: _validate_diagnostic(value) for key, value in self.diagnostics.items()}
        object.__setattr__(self, "diagnostics", MappingProxyType(validated))


def _finite_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TypeError(f"{name} must be a rank-2 tensor")
    matrix = value.detach().cpu().to(torch.float64)
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def covariance(features: torch.Tensor) -> torch.Tensor:
    matrix = _finite_matrix("features", features)
    if matrix.shape[0] < 2:
        raise ValueError("covariance requires at least two samples")
    centered = matrix - matrix.mean(dim=0)
    result = centered.T @ centered / (matrix.shape[0] - 1)
    return (result + result.T) / 2


def q1_cross_covariance(
    features: torch.Tensor, partition: SubspacePartition, *, epsilon: float = 1e-12
) -> float:
    cov = covariance(features)
    total = torch.zeros((), dtype=torch.float64)
    for left in partition.bands:
        for right in partition.bands:
            if left.slot != right.slot:
                total += (left.basis.T @ cov @ right.basis).square().sum()
    return float(torch.sqrt(total) / (torch.linalg.matrix_norm(cov) + epsilon))


def q2_projector_interaction(
    features: torch.Tensor, partition: SubspacePartition, *, epsilon: float = 1e-12
) -> float:
    cov = covariance(features)
    total = torch.zeros((), dtype=torch.float64)
    for left in partition.bands:
        for right in partition.bands:
            if left.slot != right.slot:
                total += torch.linalg.matrix_norm(left.projector @ cov @ right.projector).square()
    return float(torch.sqrt(total) / (torch.linalg.matrix_norm(cov) + epsilon))


def projector_overlap(bases: Sequence[torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    converted = [_finite_matrix("basis", basis) for basis in bases]
    for index, left in enumerate(converted):
        for other, right in enumerate(converted):
            if index != other:
                total += torch.linalg.matrix_norm(left.T @ right).square()
    return float(torch.sqrt(total))


def partition_incompleteness(projectors: Sequence[torch.Tensor], latent_dim: int) -> float:
    total = torch.zeros((latent_dim, latent_dim), dtype=torch.float64)
    for projector in projectors:
        matrix = _finite_matrix("projector", projector)
        if matrix.shape != total.shape:
            raise ValueError("projector shape mismatch")
        total += matrix
    return float(torch.linalg.matrix_norm(torch.eye(latent_dim) - total) / math.sqrt(latent_dim))


def q3_projector_overlap(partition: SubspacePartition) -> float:
    return projector_overlap([band.basis for band in partition.bands])


def q4_partition_incompleteness(partition: SubspacePartition) -> float:
    return partition_incompleteness(
        [band.projector for band in partition.bands], partition.latent_dim
    )


def q5_q6_subspace_stability(
    current: SubspacePartition, previous: SubspacePartition
) -> tuple[float, float, tuple[float, ...]]:
    assert_corresponding_partitions(current, previous)
    similarity = 0.0
    squared_distance = 0.0
    distances: list[float] = []
    for now, before in zip(current.bands, previous.bands, strict=True):
        weight = now.rank / current.latent_dim
        block_similarity = float(torch.linalg.matrix_norm(now.basis.T @ before.basis).square())
        block_similarity /= now.rank
        distance = float(
            torch.linalg.matrix_norm(now.projector - before.projector) / math.sqrt(2 * now.rank)
        )
        similarity += weight * block_similarity
        squared_distance += weight * distance * distance
        distances.append(distance)
    return similarity, math.sqrt(max(squared_distance, 0.0)), tuple(distances)


def q7_cross_jacobian_energy(
    jacobian: torch.Tensor, partition: SubspacePartition, *, epsilon: float = 1e-12
) -> MetricValue:
    matrix = _finite_matrix("jacobian", jacobian)
    energy = float(matrix.square().sum())
    if energy <= epsilon:
        return MetricValue(None, NullReason.ZERO_JACOBIAN.value)
    cross = torch.zeros((), dtype=torch.float64)
    for left in partition.bands:
        for right in partition.bands:
            if left.slot != right.slot:
                cross += torch.linalg.matrix_norm(
                    left.projector @ matrix @ right.projector
                ).square()
    return MetricValue(float(torch.sqrt(cross / energy)))


def q8_virtual_sensitivity(deltas: torch.Tensor) -> float:
    matrix = _finite_matrix("deltas", deltas)
    return float(matrix.sum(dim=1).mean())


def q9_subspace_velocity(q6: float, current_step: int, previous_step: int) -> float:
    return q6 / max(current_step - previous_step, 1)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    positive = probabilities[probabilities > 0]
    return -(positive * positive.log()).sum()


def q10_weighted_shape_entropy(
    features: torch.Tensor,
    partition: SubspacePartition,
    *,
    epsilon: float = 1e-8,
    trace_tolerance: float = 1e-12,
) -> tuple[MetricValue, tuple[float, ...]]:
    matrix = _finite_matrix("features", features)
    values: list[float] = []
    total = 0.0
    for band in partition.bands:
        cov = covariance(matrix @ band.basis)
        trace = float(torch.trace(cov))
        if trace <= trace_tolerance:
            return MetricValue(None, NullReason.DEGENERATE_COVARIANCE.value), ()
        shape = band.rank * cov / trace
        sign, logdet = torch.linalg.slogdet(shape + epsilon * torch.eye(band.rank))
        if sign <= 0:
            return MetricValue(None, NullReason.DEGENERATE_COVARIANCE.value), ()
        value = float(logdet / (2 * band.rank))
        values.append(value)
        total += (band.rank / partition.latent_dim) * value
    return MetricValue(total), tuple(values)


def q11_cross_subspace_gaussian_mi(
    features: torch.Tensor,
    partition: SubspacePartition,
    *,
    epsilon: float = 1e-8,
    negative_tolerance: float = 1e-8,
) -> tuple[MetricValue, bool]:
    matrix = _finite_matrix("features", features)
    values: list[float] = []
    clamped = False
    for left_index, left in enumerate(partition.bands):
        for right in partition.bands[left_index + 1 :]:
            coordinates = torch.cat((matrix @ left.basis, matrix @ right.basis), dim=1)
            joint = covariance(coordinates) + epsilon * torch.eye(left.rank + right.rank)
            left_cov = joint[: left.rank, : left.rank]
            right_cov = joint[left.rank :, left.rank :]
            signs = [torch.linalg.slogdet(item) for item in (left_cov, right_cov, joint)]
            if any(float(sign) <= 0 for sign, _ in signs):
                return MetricValue(None, NullReason.DEGENERATE_COVARIANCE.value), clamped
            value = 0.5 * float(signs[0][1] + signs[1][1] - signs[2][1])
            if value < -negative_tolerance:
                return MetricValue(None, NullReason.INVALID_NEGATIVE_MI.value), clamped
            if value < 0:
                value = 0.0
                clamped = True
            values.append(value / min(left.rank, right.rank))
    return MetricValue(sum(values) / len(values)), clamped


def q12_entropy_change(current: float, previous: float) -> tuple[float, float]:
    signed = current - previous
    return abs(signed), signed


def q13_surprise_locality(distances: Sequence[float], *, tolerance: float = 1e-12) -> MetricValue:
    values = torch.tensor(tuple(distances), dtype=torch.float64)
    total = float(values.sum())
    if total <= tolerance:
        return MetricValue(None, NullReason.NO_SUBSPACE_MOVEMENT.value)
    probabilities = values / total
    return MetricValue(float(1 - _entropy(probabilities) / math.log(len(distances))))


def q14_gradient_locality(
    gradients: torch.Tensor,
    partition: SubspacePartition,
    *,
    tolerance: float = 1e-12,
) -> tuple[MetricValue, tuple[float, ...], float | None]:
    if gradients.ndim < 2 or gradients.shape[-1] != partition.latent_dim:
        raise ValueError("gradients must end in the partition latent dimension")
    matrix = gradients.detach().cpu().to(torch.float64).reshape(-1, partition.latent_dim)
    total = float(matrix.square().sum())
    if total <= tolerance:
        return MetricValue(None, NullReason.ZERO_GRADIENT_ENERGY.value), (), None
    shares = tuple(float((matrix @ band.basis).square().sum() / total) for band in partition.bands)
    probabilities = torch.tensor(shares, dtype=torch.float64)
    locality = float(1 - _entropy(probabilities) / math.log(len(shares)))
    return MetricValue(locality), shares, max(shares)


def q15_virtual_interference(
    baseline: torch.Tensor, updated: torch.Tensor, *, epsilon: float = 1e-12
) -> MetricValue:
    before = _finite_matrix("baseline", baseline)
    after = _finite_matrix("updated", updated)
    if before.shape != after.shape:
        raise ValueError("baseline and updated representations must have equal shape")
    denominator = float(before.square().mean())
    if denominator <= epsilon:
        return MetricValue(None, NullReason.ZERO_DENOMINATOR.value)
    return MetricValue(float((after - before).square().mean()) / (denominator + epsilon))


def q16_entity_consistency(
    first: torch.Tensor, second: torch.Tensor, *, epsilon: float = 1e-12
) -> MetricValue:
    left = _finite_matrix("first", first)
    right = _finite_matrix("second", second)
    denominator = float(left.square().sum(dim=1).mean())
    if denominator <= epsilon:
        return MetricValue(None, NullReason.ZERO_DENOMINATOR.value)
    return MetricValue(float((left - right).square().sum(dim=1).mean()) / denominator)


def q17_transformation_residual(
    train_source: torch.Tensor,
    train_target: torch.Tensor,
    validation_source: torch.Tensor,
    validation_target: torch.Tensor,
    *,
    alpha: float = 1e-6,
    epsilon: float = 1e-12,
) -> MetricValue:
    probe = fit_ridge_probe(train_source, train_target, alpha=alpha)
    target = _finite_matrix("validation_target", validation_target)
    prediction = probe.predict(validation_source)
    denominator = float((target - target.mean(dim=0)).square().sum())
    if denominator <= epsilon:
        return MetricValue(None, NullReason.ZERO_DENOMINATOR.value)
    return MetricValue(float((prediction - target).square().sum()) / denominator)


def q18_perturbation_concentration(
    first: torch.Tensor,
    second: torch.Tensor,
    partition: SubspacePartition,
    *,
    tolerance: float = 1e-12,
    minimum_eligible_fraction: float = 0.9,
) -> tuple[MetricValue, int, int]:
    delta = _finite_matrix("first", first) - _finite_matrix("second", second)
    values: list[float] = []
    for row in delta:
        energies = torch.tensor(
            [float((row @ band.basis).square().sum()) for band in partition.bands],
            dtype=torch.float64,
        )
        total = float(energies.sum())
        if total <= tolerance:
            continue
        values.append(float(1 - _entropy(energies / total) / math.log(len(partition.bands))))
    if len(values) / max(delta.shape[0], 1) < minimum_eligible_fraction:
        return (
            MetricValue(None, NullReason.INSUFFICIENT_NONZERO_PERTURBATIONS.value),
            len(values),
            delta.shape[0],
        )
    return MetricValue(sum(values) / len(values)), len(values), delta.shape[0]


def q19_jacobian_simplicity(
    jacobian: torch.Tensor, *, threshold: float = 1e-3, tolerance: float = 1e-12
) -> tuple[MetricValue, MetricValue]:
    matrix = _finite_matrix("jacobian", jacobian)
    singular_values = torch.linalg.svdvals(matrix)
    energy = float(singular_values.square().sum())
    maximum = float(matrix.abs().max())
    if energy <= tolerance or maximum <= tolerance:
        missing = MetricValue(None, NullReason.ZERO_JACOBIAN.value)
        return missing, missing
    probabilities = singular_values.square() / energy
    effective_rank = float(torch.exp(_entropy(probabilities)) / min(matrix.shape))
    density = float((matrix.abs() > threshold * maximum).to(torch.float64).mean())
    return MetricValue(effective_rank), MetricValue(density)


def q20_reconstruction_nmse(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    alpha: float = 1e-6,
    epsilon: float = 1e-12,
) -> MetricValue:
    probe = fit_ridge_probe(train_features, train_targets, alpha=alpha)
    targets = _finite_matrix("validation_targets", validation_targets)
    denominator = float((targets - targets.mean(dim=0)).square().sum())
    if denominator <= epsilon:
        return MetricValue(None, NullReason.ZERO_DENOMINATOR.value)
    prediction = probe.predict(validation_features)
    return MetricValue(float((prediction - targets).square().sum()) / denominator)


__all__ = [
    "CellState",
    "CostTier",
    "METRIC_SPECS",
    "METRIC_SPEC_BY_NAME",
    "MetricDirection",
    "MetricRole",
    "MetricSpec",
    "NullReason",
    "QualityMetricResult",
    "covariance",
    "partition_incompleteness",
    "projector_overlap",
    "q1_cross_covariance",
    "q2_projector_interaction",
    "q3_projector_overlap",
    "q4_partition_incompleteness",
    "q5_q6_subspace_stability",
    "q7_cross_jacobian_energy",
    "q8_virtual_sensitivity",
    "q9_subspace_velocity",
    "q10_weighted_shape_entropy",
    "q11_cross_subspace_gaussian_mi",
    "q12_entropy_change",
    "q13_surprise_locality",
    "q14_gradient_locality",
    "q15_virtual_interference",
    "q16_entity_consistency",
    "q17_transformation_residual",
    "q18_perturbation_concentration",
    "q19_jacobian_simplicity",
    "q20_reconstruction_nmse",
    "validate_metric_specs",
]
