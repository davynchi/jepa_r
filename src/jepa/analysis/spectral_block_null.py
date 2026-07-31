"""Spectral-null controls for joint block structure of linear operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from jepa.analysis.joint_transform_factorization import (
    JointBlockFit,
    WhiteningProjection,
    fit_joint_block_diagonalization,
)


@dataclass(frozen=True)
class SpectralNullResult:
    real_factorization: float
    null_mean: float
    null_std: float
    delta: float
    z_score: float
    null_factorizations: tuple[float, ...]
    real_basis: torch.Tensor


@dataclass(frozen=True)
class ProjectorAgreement:
    mean_overlap: float
    minimum_overlap: float
    matched_overlaps: tuple[float, ...]
    assignment: tuple[int, ...]


def fit_variance_whitening_projection(
    features: torch.Tensor,
    *,
    minimum_variance_fraction: float = 0.95,
    dimension_multiple: int = 1,
    relative_floor: float = 1e-6,
) -> WhiteningProjection:
    """Fit per-checkpoint PCA whitening at the smallest admissible dimension."""
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must have shape [samples, dimensions]")
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    if not 0 < minimum_variance_fraction <= 1:
        raise ValueError("minimum_variance_fraction must be in (0, 1]")
    if dimension_multiple <= 0:
        raise ValueError("dimension_multiple must be positive")

    matrix = features.to(torch.float64)
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    covariance = centered.T @ centered / (matrix.shape[0] - 1)
    values, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    total = values.sum().clamp_min(1e-12)
    cumulative = values.cumsum(dim=0) / total
    required = (
        int(
            torch.searchsorted(
                cumulative,
                cumulative.new_tensor(minimum_variance_fraction),
            ).item()
        )
        + 1
    )
    dimension = min(
        ((required + dimension_multiple - 1) // dimension_multiple) * dimension_multiple,
        matrix.shape[1],
    )
    if dimension % dimension_multiple:
        raise ValueError("latent dimension is not divisible by dimension_multiple")
    selected = values[:dimension]
    floor = max(float(values[0]) * relative_floor, 1e-12)
    projection = vectors[:, :dimension] * selected.clamp_min(floor).rsqrt()
    retained = float(selected.sum() / total)
    return WhiteningProjection(
        mean=mean,
        projection=projection,
        retained_variance_fraction=retained,
    )


def haar_orthogonal(
    dimension: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample a Haar-distributed matrix from O(d) using sign-corrected QR."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    sample = torch.randn(
        dimension,
        dimension,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    basis, triangular = torch.linalg.qr(sample)
    signs = torch.sign(torch.diagonal(triangular))
    signs[signs == 0] = 1
    basis = basis * signs.unsqueeze(0)
    return basis.to(device=device, dtype=dtype)


def independently_rotate_operators(
    operators: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Destroy shared eigenspaces while preserving each operator's spectrum."""
    if operators.ndim != 3 or operators.shape[1] != operators.shape[2]:
        raise ValueError("operators must have shape [transforms, d, d]")
    generator = torch.Generator().manual_seed(seed)
    rotated = []
    for operator in operators:
        basis = haar_orthogonal(
            operator.shape[0],
            generator=generator,
            device=operator.device,
            dtype=operator.dtype,
        )
        rotated.append(basis.T @ operator @ basis)
    return torch.stack(rotated)


def optimized_factorization(
    operators: torch.Tensor,
    *,
    num_blocks: int,
    restarts: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[float, JointBlockFit]:
    fit = fit_joint_block_diagonalization(
        operators,
        operators,
        num_blocks=num_blocks,
        restarts=restarts,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
    )
    return fit.train_factorization, fit


def _block_factorization_scores(
    operator_sets: torch.Tensor,
    bases: torch.Tensor,
    *,
    num_blocks: int,
) -> torch.Tensor:
    """Return F_K for operator sets [S,A,D,D] and bases [S,R,D,D]."""
    dimension = operator_sets.shape[-1]
    block_size = dimension // num_blocks
    block_ids = torch.arange(dimension, device=operator_sets.device) // block_size
    mask = (block_ids[:, None] != block_ids[None, :]).to(operator_sets.dtype)
    rotated = bases.transpose(-1, -2).unsqueeze(2) @ operator_sets.unsqueeze(1) @ bases.unsqueeze(2)
    numerator = (rotated * mask).square().sum(dim=(2, 3, 4))
    denominator = operator_sets.square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
    return 1.0 - numerator / denominator[:, None]


def optimized_factorizations_batched(
    operator_sets: torch.Tensor,
    *,
    num_blocks: int,
    restarts: int,
    steps: int,
    learning_rate: float,
    seeds: list[int],
    batch_size: int,
) -> list[float]:
    """Optimize independent joint block objectives in GPU-friendly batches."""
    if operator_sets.ndim != 4 or operator_sets.shape[-2] != operator_sets.shape[-1]:
        raise ValueError("operator_sets must have shape [sets, transforms, d, d]")
    if len(seeds) != operator_sets.shape[0]:
        raise ValueError("one optimization seed is required per operator set")
    if operator_sets.shape[-1] % num_blocks:
        raise ValueError("num_blocks must divide the operator dimension")
    if restarts <= 0 or steps <= 0 or learning_rate <= 0 or batch_size <= 0:
        raise ValueError("optimization settings must be positive")

    dimension = operator_sets.shape[-1]
    identity = torch.eye(
        dimension,
        device=operator_sets.device,
        dtype=operator_sets.dtype,
    )
    all_scores: list[float] = []
    for start in range(0, operator_sets.shape[0], batch_size):
        stop = min(start + batch_size, operator_sets.shape[0])
        operators = operator_sets[start:stop]
        initial = []
        for sample_seed in seeds[start:stop]:
            for restart in range(restarts):
                initial.append(
                    haar_orthogonal(
                        dimension,
                        generator=torch.Generator().manual_seed(sample_seed + restart),
                        device=operators.device,
                        dtype=operators.dtype,
                    )
                )
        initial_bases = torch.stack(initial).reshape(
            stop - start,
            restarts,
            dimension,
            dimension,
        )
        coordinates = torch.nn.Parameter(torch.zeros_like(initial_bases))
        optimizer = torch.optim.Adam((coordinates,), lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            skew = coordinates - coordinates.transpose(-1, -2)
            bases = torch.linalg.solve(identity + skew, identity - skew) @ initial_bases
            scores = _block_factorization_scores(
                operators,
                bases,
                num_blocks=num_blocks,
            )
            (1.0 - scores).sum().backward()
            optimizer.step()
        with torch.no_grad():
            skew = coordinates - coordinates.transpose(-1, -2)
            bases = torch.linalg.solve(identity + skew, identity - skew) @ initial_bases
            scores = _block_factorization_scores(
                operators,
                bases,
                num_blocks=num_blocks,
            )
        all_scores.extend(float(value) for value in scores.max(dim=1).values.cpu())
    return all_scores


def block_projectors(basis: torch.Tensor, *, num_blocks: int) -> torch.Tensor:
    """Convert an orthogonal basis into its unordered block projectors."""
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("basis must be a square matrix")
    if num_blocks <= 1 or basis.shape[0] % num_blocks:
        raise ValueError("num_blocks must exceed one and divide the dimension")
    block_size = basis.shape[0] // num_blocks
    blocks = basis.T.reshape(num_blocks, block_size, basis.shape[0])
    return blocks.transpose(-1, -2) @ blocks


def _optimal_assignment(overlaps: torch.Tensor) -> tuple[int, ...]:
    """Maximize a small square assignment problem with bit-mask dynamic programming."""
    size = overlaps.shape[0]
    scores: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in range(size):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (score, assignment) in scores.items():
            for column in range(size):
                if mask & (1 << column):
                    continue
                new_mask = mask | (1 << column)
                candidate = (score + float(overlaps[row, column]), assignment + (column,))
                if new_mask not in updated or candidate[0] > updated[new_mask][0]:
                    updated[new_mask] = candidate
        scores = updated
    return scores[(1 << size) - 1][1]


def optimal_projector_agreement(
    first_basis: torch.Tensor,
    second_basis: torch.Tensor,
    *,
    num_blocks: int,
) -> ProjectorAgreement:
    """Compare unordered block subspaces, ignoring within-block rotations."""
    if first_basis.shape != second_basis.shape:
        raise ValueError("bases must have equal shape")
    first = block_projectors(first_basis, num_blocks=num_blocks)
    second = block_projectors(second_basis, num_blocks=num_blocks).to(
        device=first.device,
        dtype=first.dtype,
    )
    block_size = first_basis.shape[0] // num_blocks
    overlaps = torch.einsum("aij,bji->ab", first, second) / block_size
    assignment = _optimal_assignment(overlaps)
    matched = tuple(float(overlaps[row, column]) for row, column in enumerate(assignment))
    return ProjectorAgreement(
        mean_overlap=sum(matched) / len(matched),
        minimum_overlap=min(matched),
        matched_overlaps=matched,
        assignment=assignment,
    )


def spectral_null_test(
    operators: torch.Tensor,
    *,
    num_blocks: int,
    null_samples: int,
    restarts: int,
    steps: int,
    learning_rate: float,
    seed: int,
    null_batch_size: int = 10,
) -> SpectralNullResult:
    """Compare optimized real block structure with independent Haar rotations."""
    if null_samples < 2:
        raise ValueError("null_samples must be at least two")
    real, real_fit = optimized_factorization(
        operators,
        num_blocks=num_blocks,
        restarts=restarts,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
    )
    null_sets = []
    for sample_index in range(null_samples):
        null_sets.append(
            independently_rotate_operators(
                operators,
                seed=seed + 10_000 + sample_index,
            )
        )
    null_values = optimized_factorizations_batched(
        torch.stack(null_sets),
        num_blocks=num_blocks,
        restarts=restarts,
        steps=steps,
        learning_rate=learning_rate,
        seeds=[seed + 20_000 + index for index in range(null_samples)],
        batch_size=null_batch_size,
    )
    null_tensor = torch.tensor(null_values, dtype=torch.float64)
    null_mean = float(null_tensor.mean())
    null_std = float(null_tensor.std(unbiased=True))
    delta = real - null_mean
    z_score = delta / null_std if null_std > 0 else float("nan")
    return SpectralNullResult(
        real_factorization=real,
        null_mean=null_mean,
        null_std=null_std,
        delta=delta,
        z_score=z_score,
        null_factorizations=tuple(null_values),
        real_basis=real_fit.basis,
    )


__all__ = [
    "SpectralNullResult",
    "ProjectorAgreement",
    "block_projectors",
    "fit_variance_whitening_projection",
    "haar_orthogonal",
    "independently_rotate_operators",
    "optimized_factorization",
    "optimized_factorizations_batched",
    "optimal_projector_agreement",
    "spectral_null_test",
]
