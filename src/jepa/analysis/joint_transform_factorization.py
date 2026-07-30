"""Joint block factorization of latent transformation operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _finite_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


@dataclass(frozen=True)
class WhiteningProjection:
    mean: torch.Tensor
    projection: torch.Tensor
    retained_variance_fraction: float


@dataclass(frozen=True)
class LinearOperatorFit:
    operator: torch.Tensor
    normalized_mse: float
    r_squared: float
    energy_per_dimension: float


@dataclass(frozen=True)
class JointBlockFit:
    basis: torch.Tensor
    train_factorization: float
    validation_factorization: float
    random_validation_factorization: float
    restart_validation_factorizations: tuple[float, ...]


def fit_whitening_projection(
    features: torch.Tensor,
    *,
    dimension: int,
    relative_floor: float = 1e-6,
) -> WhiteningProjection:
    """Fit a PCA whitening map on reference representations."""
    matrix = _finite_matrix("features", features).to(torch.float64)
    if matrix.shape[0] < 2:
        raise ValueError("at least two features are required")
    if not 0 < dimension <= matrix.shape[1]:
        raise ValueError("dimension must be in [1, latent_dim]")
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    covariance = centered.T @ centered / (matrix.shape[0] - 1)
    values, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    selected = values[:dimension]
    floor = max(float(values[0]) * relative_floor, 1e-12)
    projection = vectors[:, :dimension] * selected.clamp_min(floor).rsqrt()
    retained = float(selected.sum() / values.sum().clamp_min(1e-12))
    return WhiteningProjection(mean, projection, retained)


def apply_whitening(
    features: torch.Tensor,
    whitening: WhiteningProjection,
) -> torch.Tensor:
    matrix = _finite_matrix("features", features).to(
        device=whitening.projection.device,
        dtype=whitening.projection.dtype,
    )
    return (matrix - whitening.mean) @ whitening.projection


def fit_linear_operator(
    train_source: torch.Tensor,
    train_target: torch.Tensor,
    validation_source: torch.Tensor,
    validation_target: torch.Tensor,
    *,
    ridge: float = 1e-3,
) -> LinearOperatorFit:
    """Fit ``target ~= source @ operator`` and score it out of sample."""
    source = _finite_matrix("train_source", train_source)
    target = _finite_matrix("train_target", train_target)
    validation_source = _finite_matrix("validation_source", validation_source)
    validation_target = _finite_matrix("validation_target", validation_target)
    if source.shape != target.shape:
        raise ValueError("training source and target must have equal shape")
    if validation_source.shape != validation_target.shape:
        raise ValueError("validation source and target must have equal shape")
    if source.shape[1] != validation_source.shape[1]:
        raise ValueError("training and validation dimensions must match")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    identity = torch.eye(source.shape[1], device=source.device, dtype=source.dtype)
    gram = source.T @ source
    operator = torch.linalg.solve(
        gram + ridge * source.shape[0] * identity,
        source.T @ target,
    )
    prediction = validation_source @ operator
    squared_error = (prediction - validation_target).square().sum()
    centered_target = validation_target - validation_target.mean(dim=0)
    denominator = centered_target.square().sum().clamp_min(1e-12)
    normalized_mse = float(squared_error / denominator)
    return LinearOperatorFit(
        operator=operator,
        normalized_mse=normalized_mse,
        r_squared=1.0 - normalized_mse,
        energy_per_dimension=float(operator.square().sum() / operator.shape[0]),
    )


def _off_block_mask(
    dimension: int,
    num_blocks: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_blocks <= 1 or dimension % num_blocks:
        raise ValueError("num_blocks must exceed one and divide the dimension")
    block_size = dimension // num_blocks
    block_ids = torch.arange(dimension, device=device) // block_size
    return (block_ids[:, None] != block_ids[None, :]).to(dtype)


def block_defect(
    operators: torch.Tensor,
    basis: torch.Tensor,
    *,
    num_blocks: int,
) -> torch.Tensor:
    """Return normalized off-block energy in a shared orthogonal basis."""
    if operators.ndim != 3 or operators.shape[1] != operators.shape[2]:
        raise ValueError("operators must have shape [A, D, D]")
    if basis.shape != operators.shape[1:]:
        raise ValueError("basis shape must match an operator")
    mask = _off_block_mask(
        operators.shape[-1],
        num_blocks,
        device=operators.device,
        dtype=operators.dtype,
    )
    rotated = basis.T.unsqueeze(0) @ operators @ basis.unsqueeze(0)
    numerator = (rotated * mask).square().sum()
    denominator = operators.square().sum().clamp_min(1e-12)
    return numerator / denominator


def remove_isotropic_component(operators: torch.Tensor) -> torch.Tensor:
    """Remove the scalar-identity part that is block diagonal in every basis."""
    if operators.ndim != 3 or operators.shape[1] != operators.shape[2]:
        raise ValueError("operators must have shape [A, D, D]")
    dimension = operators.shape[-1]
    identity = torch.eye(
        dimension,
        device=operators.device,
        dtype=operators.dtype,
    )
    scales = torch.diagonal(operators, dim1=-2, dim2=-1).sum(dim=-1) / dimension
    return operators - scales[:, None, None] * identity


def _random_orthogonal(
    dimension: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sample = torch.randn(
        dimension,
        dimension,
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    )
    basis, _ = torch.linalg.qr(sample)
    return basis.to(device=device, dtype=dtype)


def fit_joint_block_diagonalization(
    train_operators: torch.Tensor,
    validation_operators: torch.Tensor,
    *,
    num_blocks: int,
    restarts: int = 4,
    steps: int = 600,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> JointBlockFit:
    """Fit one orthogonal basis for all operators and evaluate it held out."""
    train = train_operators
    validation = validation_operators.to(device=train.device, dtype=train.dtype)
    if train.ndim != 3 or train.shape[1] != train.shape[2]:
        raise ValueError("train_operators must have shape [A, D, D]")
    if validation.shape != train.shape:
        raise ValueError("training and validation operators must have equal shape")
    if restarts <= 0 or steps <= 0 or learning_rate <= 0:
        raise ValueError("restarts, steps, and learning_rate must be positive")
    _off_block_mask(
        train.shape[-1],
        num_blocks,
        device=train.device,
        dtype=train.dtype,
    )

    validation_scores: list[float] = []
    candidates: list[tuple[float, float, torch.Tensor]] = []
    random_scores: list[float] = []
    identity = torch.eye(train.shape[-1], device=train.device, dtype=train.dtype)
    for restart in range(restarts):
        generator = torch.Generator().manual_seed(seed + restart)
        initial = _random_orthogonal(
            train.shape[-1],
            generator=generator,
            device=train.device,
            dtype=train.dtype,
        )
        random_scores.append(
            1.0
            - float(block_defect(validation, initial, num_blocks=num_blocks))
        )
        coordinates = torch.nn.Parameter(torch.zeros_like(initial))
        optimizer = torch.optim.Adam((coordinates,), lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            skew = coordinates - coordinates.T
            rotation = torch.linalg.solve(identity + skew, identity - skew)
            basis = rotation @ initial
            loss = block_defect(train, basis, num_blocks=num_blocks)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            skew = coordinates - coordinates.T
            basis = torch.linalg.solve(identity + skew, identity - skew) @ initial
            train_score = 1.0 - float(
                block_defect(train, basis, num_blocks=num_blocks)
            )
            validation_score = 1.0 - float(
                block_defect(validation, basis, num_blocks=num_blocks)
            )
        validation_scores.append(validation_score)
        candidates.append((train_score, validation_score, basis.detach().clone()))

    # Selection uses the training objective only; validation remains a true audit.
    train_score, validation_score, best_basis = max(candidates, key=lambda row: row[0])
    return JointBlockFit(
        basis=best_basis,
        train_factorization=train_score,
        validation_factorization=validation_score,
        random_validation_factorization=sum(random_scores) / len(random_scores),
        restart_validation_factorizations=tuple(validation_scores),
    )


__all__ = [
    "JointBlockFit",
    "LinearOperatorFit",
    "WhiteningProjection",
    "apply_whitening",
    "block_defect",
    "fit_joint_block_diagonalization",
    "fit_linear_operator",
    "fit_whitening_projection",
    "remove_isotropic_component",
]
