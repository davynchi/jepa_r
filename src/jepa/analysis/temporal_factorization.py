"""Temporal diagnostics for transformation-derived latent subspaces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    result = value.detach().cpu().to(torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0)
    return centered.T @ centered / max(features.shape[0] - 1, 1)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    positive = probabilities[probabilities > 0]
    return -(positive * positive.log()).sum()


@dataclass(frozen=True)
class ProcrustesFit:
    rotation: torch.Tensor
    train_normalized_residual: float
    validation_normalized_residual: float


@dataclass(frozen=True)
class MatchedSubspaceMetrics:
    assignment: tuple[int, ...]
    similarity: float
    grassmann_distance: float
    per_block_similarity: tuple[float, ...]
    per_block_distance: tuple[float, ...]
    principal_angles_degrees: tuple[tuple[float, ...], ...]
    principal_angle_mean_degrees: float
    principal_angle_max_degrees: float
    principal_angle_rms_sine: float


def fit_orthogonal_procrustes(
    train_source: torch.Tensor,
    train_target: torch.Tensor,
    validation_source: torch.Tensor,
    validation_target: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> ProcrustesFit:
    """Fit a rotation after independent centering and score it held out."""
    source = _matrix("train_source", train_source)
    target = _matrix("train_target", train_target)
    validation_source = _matrix("validation_source", validation_source)
    validation_target = _matrix("validation_target", validation_target)
    if source.shape != target.shape:
        raise ValueError("training representations must have equal shape")
    if validation_source.shape != validation_target.shape:
        raise ValueError("validation representations must have equal shape")
    if source.shape[1] != validation_source.shape[1]:
        raise ValueError("training and validation dimensions must match")

    source_centered = source - source.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    left, _, right_h = torch.linalg.svd(
        source_centered.T @ target_centered,
        full_matrices=False,
    )
    rotation = left @ right_h

    def residual(left_features: torch.Tensor, right_features: torch.Tensor) -> float:
        left_centered = left_features - left_features.mean(dim=0)
        right_centered = right_features - right_features.mean(dim=0)
        numerator = torch.linalg.matrix_norm(left_centered @ rotation - right_centered)
        denominator = torch.linalg.matrix_norm(right_centered).clamp_min(epsilon)
        return float(numerator / denominator)

    return ProcrustesFit(
        rotation=rotation,
        train_normalized_residual=residual(source, target),
        validation_normalized_residual=residual(
            validation_source,
            validation_target,
        ),
    )


def apply_affine_alignment(
    features: torch.Tensor,
    *,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    matrix = _matrix("features", features)
    return (matrix - source_mean) @ rotation + target_mean


def linear_cka(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Centered linear CKA computed without materializing sample Gram matrices."""
    left = _matrix("first", first)
    right = _matrix("second", second)
    if left.shape[0] != right.shape[0]:
        raise ValueError("representations must have equal sample counts")
    left -= left.mean(dim=0)
    right -= right.mean(dim=0)
    cross = torch.linalg.matrix_norm(left.T @ right).square()
    denominator = (
        torch.linalg.matrix_norm(left.T @ left)
        * torch.linalg.matrix_norm(right.T @ right)
    ).clamp_min(epsilon)
    return float((cross / denominator).clamp(0, 1))


def split_basis(basis: torch.Tensor, num_blocks: int) -> tuple[torch.Tensor, ...]:
    matrix = _matrix("basis", basis)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("basis must be square")
    if num_blocks <= 1 or matrix.shape[1] % num_blocks:
        raise ValueError("num_blocks must exceed one and divide the dimension")
    block_size = matrix.shape[1] // num_blocks
    return tuple(
        matrix[:, start : start + block_size]
        for start in range(0, matrix.shape[1], block_size)
    )


def _maximum_assignment(scores: torch.Tensor) -> tuple[int, ...]:
    """Exact maximum-weight assignment using O(K 2^K) dynamic programming."""
    matrix = _matrix("scores", scores)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("assignment scores must be square")
    size = matrix.shape[0]
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in range(size):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (score, assignment) in states.items():
            for column in range(size):
                bit = 1 << column
                if mask & bit:
                    continue
                candidate = (score + float(matrix[row, column]), assignment + (column,))
                new_mask = mask | bit
                if new_mask not in updated or candidate[0] > updated[new_mask][0]:
                    updated[new_mask] = candidate
        states = updated
    return states[(1 << size) - 1][1]


def match_subspaces(
    current_basis: torch.Tensor,
    previous_basis: torch.Tensor,
    *,
    num_blocks: int,
) -> MatchedSubspaceMetrics:
    """Optimally match unordered equal-rank blocks and report principal angles."""
    current = split_basis(current_basis, num_blocks)
    previous = split_basis(previous_basis, num_blocks)
    rank = current[0].shape[1]
    scores = torch.empty((num_blocks, num_blocks), dtype=torch.float64)
    for current_index, current_block in enumerate(current):
        for previous_index, previous_block in enumerate(previous):
            scores[current_index, previous_index] = (
                torch.linalg.matrix_norm(current_block.T @ previous_block).square()
                / rank
            )
    assignment = _maximum_assignment(scores)
    similarities = []
    distances = []
    all_angles: list[tuple[float, ...]] = []
    all_sines = []
    for current_index, previous_index in enumerate(assignment):
        singular_values = torch.linalg.svdvals(
            current[current_index].T @ previous[previous_index]
        ).clamp(0, 1)
        angles = torch.acos(singular_values)
        similarities.append(float(singular_values.square().mean()))
        distances.append(float(angles.sin().square().mean().sqrt()))
        all_sines.extend(float(value) for value in angles.sin())
        all_angles.append(tuple(float(value * 180 / math.pi) for value in angles))
    similarity = sum(similarities) / len(similarities)
    grassmann = math.sqrt(
        sum(distance * distance for distance in distances) / len(distances)
    )
    flat_angles = [angle for block in all_angles for angle in block]
    return MatchedSubspaceMetrics(
        assignment=assignment,
        similarity=similarity,
        grassmann_distance=grassmann,
        per_block_similarity=tuple(similarities),
        per_block_distance=tuple(distances),
        principal_angles_degrees=tuple(all_angles),
        principal_angle_mean_degrees=sum(flat_angles) / len(flat_angles),
        principal_angle_max_degrees=max(flat_angles),
        principal_angle_rms_sine=math.sqrt(
            sum(value * value for value in all_sines) / len(all_sines)
        ),
    )


def block_gaussian_statistics(
    features: torch.Tensor,
    basis: torch.Tensor,
    *,
    num_blocks: int,
    relative_floor: float = 1e-6,
) -> dict[str, object]:
    """Gaussian entropy, spectral shape, and effective rank per latent block."""
    matrix = _matrix("features", features)
    blocks = split_basis(basis, num_blocks)
    rows = []
    for block in blocks:
        covariance = _covariance(matrix @ block)
        eigenvalues = torch.linalg.eigvalsh(
            (covariance + covariance.T) / 2
        ).clamp_min(0)
        floor = max(float(eigenvalues.max()) * relative_floor, 1e-12)
        regularized = eigenvalues.clamp_min(floor)
        probabilities = regularized / regularized.sum()
        effective_rank = float(_entropy(probabilities).exp())
        rank = block.shape[1]
        entropy_per_dimension = 0.5 * (
            math.log(2 * math.pi * math.e) + float(regularized.log().mean())
        )
        normalized_shape = rank * regularized / regularized.sum()
        rows.append(
            {
                "gaussian_entropy_per_dimension": entropy_per_dimension,
                "effective_rank": effective_rank,
                "normalized_effective_rank": effective_rank / rank,
                "shape_logdet_per_dimension": 0.5
                * float(normalized_shape.log().mean()),
                "trace_per_dimension": float(regularized.mean()),
            }
        )
    keys = tuple(rows[0])
    return {
        "blocks": rows,
        "weighted_mean": {
            key: sum(float(row[key]) for row in rows) / len(rows) for key in keys
        },
    }


def cross_block_gaussian_mi(
    features: torch.Tensor,
    basis: torch.Tensor,
    *,
    num_blocks: int,
    relative_floor: float = 1e-6,
) -> dict[str, object]:
    """Pairwise Gaussian mutual information between orthogonal blocks."""
    matrix = _matrix("features", features)
    blocks = split_basis(basis, num_blocks)

    pairs = []
    for left_index, left in enumerate(blocks):
        left_features = matrix @ left
        for right_index in range(left_index + 1, len(blocks)):
            right_features = matrix @ blocks[right_index]
            joint = torch.cat((left_features, right_features), dim=1)
            joint_covariance = _covariance(joint)
            joint_covariance = (joint_covariance + joint_covariance.T) / 2
            maximum = float(
                torch.linalg.eigvalsh(joint_covariance).clamp_min(0).max()
            )
            ridge = max(maximum * relative_floor, 1e-12)
            left_rank = left.shape[1]
            left_covariance = joint_covariance[:left_rank, :left_rank]
            right_covariance = joint_covariance[left_rank:, left_rank:]

            def regularized_logdet(
                covariance: torch.Tensor,
                *,
                regularization: float = ridge,
            ) -> float:
                identity = torch.eye(covariance.shape[0], dtype=covariance.dtype)
                sign, value = torch.linalg.slogdet(
                    covariance + regularization * identity
                )
                if float(sign) <= 0:
                    raise ValueError("regularized covariance is not positive definite")
                return float(value)

            value = 0.5 * (
                regularized_logdet(left_covariance)
                + regularized_logdet(right_covariance)
                - regularized_logdet(joint_covariance)
            )
            normalized = max(value, 0.0) / min(left.shape[1], blocks[right_index].shape[1])
            pairs.append(
                {
                    "left": left_index,
                    "right": right_index,
                    "value": max(value, 0.0),
                    "normalized": normalized,
                }
            )
    return {
        "pairwise": pairs,
        "normalized_mean": sum(float(row["normalized"]) for row in pairs)
        / len(pairs),
        "normalized_max": max(float(row["normalized"]) for row in pairs),
    }


def perturbation_concentration(
    source: torch.Tensor,
    target: torch.Tensor,
    basis: torch.Tensor,
    *,
    num_blocks: int,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Measure how many discovered blocks support each augmentation delta."""
    left = _matrix("source", source)
    right = _matrix("target", target)
    if left.shape != right.shape:
        raise ValueError("source and target must have equal shape")
    blocks = split_basis(basis, num_blocks)
    delta = right - left
    energies = torch.stack(
        [(delta @ block).square().sum(dim=1) for block in blocks],
        dim=1,
    )
    totals = energies.sum(dim=1)
    eligible = totals > tolerance
    if not bool(eligible.any()):
        raise ValueError("all augmentation deltas have zero energy")
    probabilities = energies[eligible] / totals[eligible, None]
    entropies = torch.stack([_entropy(row) for row in probabilities])
    effective_support = entropies.exp()
    concentration = 1 - entropies / math.log(num_blocks)
    return {
        "mean_concentration": float(concentration.mean()),
        "mean_effective_support": float(effective_support.mean()),
        "normalized_effective_support": float(
            effective_support.mean() / num_blocks
        ),
        "mean_energy_shares": [
            float(value) for value in probabilities.mean(dim=0)
        ],
        "eligible_fraction": float(eligible.to(torch.float64).mean()),
    }


__all__ = [
    "MatchedSubspaceMetrics",
    "ProcrustesFit",
    "apply_affine_alignment",
    "block_gaussian_statistics",
    "cross_block_gaussian_mi",
    "fit_orthogonal_procrustes",
    "linear_cka",
    "match_subspaces",
    "perturbation_concentration",
    "split_basis",
]
