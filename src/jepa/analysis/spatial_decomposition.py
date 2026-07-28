"""Deterministic, validated spatial latent-space decompositions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from jepa.analysis.subspace import compute_scatter_matrices, solve_generalized_eigenproblem


class Panel(StrEnum):
    LABEL_FREE = "label_free"
    SUPERVISED_LDA = "supervised_lda"


class EigenBoundaryTieError(ValueError):
    pass


def _matrix(name: str, value: torch.Tensor, *, ndim: int = 2) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise TypeError(f"{name} must be a rank-{ndim} torch.Tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _sign_fix(basis: torch.Tensor) -> torch.Tensor:
    fixed = basis.clone()
    for column in range(fixed.shape[1]):
        vector = fixed[:, column]
        pivot = int(vector.abs().argmax())
        if vector[pivot] < 0:
            fixed[:, column] = -vector
    return fixed


def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return _sign_fix(q)


def _complement_basis(basis: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(basis, mode="complete")
    return _sign_fix(q[:, basis.shape[1] :])


def _band_ranks(latent_dim: int, bands: int) -> tuple[int, ...]:
    if bands <= 0 or bands > latent_dim:
        raise ValueError("bands must be in [1, latent_dim]")
    quotient, remainder = divmod(latent_dim, bands)
    return tuple(quotient + (slot < remainder) for slot in range(bands))


@dataclass(frozen=True, slots=True)
class SubspaceBand:
    slot: int
    rank: int
    basis: torch.Tensor
    projector: torch.Tensor
    provenance: str

    def __post_init__(self) -> None:
        basis = _matrix("basis", self.basis)
        projector = _matrix("projector", self.projector)
        if self.slot < 0 or self.rank <= 0:
            raise ValueError("slot must be non-negative and rank must be positive")
        if basis.shape[1] != self.rank:
            raise ValueError("rank must equal basis width")
        if projector.shape != (basis.shape[0], basis.shape[0]):
            raise ValueError("projector shape does not match basis latent dimension")
        identity = torch.eye(self.rank, dtype=torch.float64)
        if not torch.allclose(basis.T @ basis, identity, atol=1e-8, rtol=0):
            raise ValueError("basis is not orthonormal")
        if not torch.allclose(projector, basis @ basis.T, atol=1e-8, rtol=0):
            raise ValueError("projector does not equal basis @ basis.T")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "projector", projector)


@dataclass(frozen=True, slots=True)
class SubspacePartition:
    panel: Panel
    latent_dim: int
    checkpoint_id: str
    bands: tuple[SubspaceBand, ...]
    decomposition_version: str = "1"

    def __post_init__(self) -> None:
        if self.latent_dim <= 0 or not self.bands:
            raise ValueError("partition requires a positive latent dimension and bands")
        if tuple(band.slot for band in self.bands) != tuple(range(len(self.bands))):
            raise ValueError("band slots must be unique and contiguous")
        if sum(band.rank for band in self.bands) != self.latent_dim:
            raise ValueError("band ranks do not cover the latent dimension")
        for band in self.bands:
            if band.basis.shape[0] != self.latent_dim:
                raise ValueError("band latent dimension mismatch")
        projector_sum = sum(
            (band.projector for band in self.bands),
            torch.zeros((self.latent_dim, self.latent_dim), dtype=torch.float64),
        )
        identity = torch.eye(self.latent_dim, dtype=torch.float64)
        if not torch.allclose(projector_sum, identity, atol=1e-8, rtol=0):
            raise ValueError("partition is incomplete or bands overlap")


def _partition_from_basis(
    basis: torch.Tensor,
    *,
    ranks: tuple[int, ...],
    panel: Panel,
    checkpoint_id: str,
    provenances: tuple[str, ...],
) -> SubspacePartition:
    bands: list[SubspaceBand] = []
    start = 0
    for slot, (rank, provenance) in enumerate(zip(ranks, provenances, strict=True)):
        block = basis[:, start : start + rank]
        bands.append(SubspaceBand(slot, rank, block, block @ block.T, provenance))
        start += rank
    return SubspacePartition(panel, basis.shape[0], checkpoint_id, tuple(bands))


def _reject_tied_boundaries(
    eigenvalues: torch.Tensor, ranks: tuple[int, ...], *, relative_tolerance: float
) -> None:
    boundary = 0
    scale = max(float(eigenvalues.abs().max()), 1.0)
    for rank in ranks[:-1]:
        boundary += rank
        if abs(float(eigenvalues[boundary - 1] - eigenvalues[boundary])) <= (
            relative_tolerance * scale
        ):
            raise EigenBoundaryTieError(f"eigenvalue tie at band boundary {boundary}")


def fit_label_free_partition(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    *,
    checkpoint_id: str,
    bands: int = 4,
    epsilon: float = 1e-8,
    relative_tie_tolerance: float = 1e-6,
) -> SubspacePartition:
    """Fit the mask-invariance partition. The API intentionally has no labels."""
    a = _matrix("view_a", view_a)
    b = _matrix("view_b", view_b)
    if a.shape != b.shape or a.shape[0] < 2:
        raise ValueError("paired views must have equal shape and at least two rows")
    ranks = _band_ranks(a.shape[1], bands)
    delta = a - b
    mean_view = (a + b) / 2
    delta = delta - delta.mean(dim=0)
    mean_view = mean_view - mean_view.mean(dim=0)
    within = delta.T @ delta / (a.shape[0] - 1)
    signal = mean_view.T @ mean_view / (a.shape[0] - 1)
    within = within / max(float(torch.trace(within)), epsilon)
    signal = signal / max(float(torch.trace(signal)), epsilon)
    operator = ((signal - within) + (signal - within).T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    basis = _sign_fix(eigenvectors[:, order])
    _reject_tied_boundaries(eigenvalues, ranks, relative_tolerance=relative_tie_tolerance)
    provenances = tuple(f"mask_operator_spectral_band_{slot}" for slot in range(bands))
    return _partition_from_basis(
        basis,
        ranks=ranks,
        panel=Panel.LABEL_FREE,
        checkpoint_id=checkpoint_id,
        provenances=provenances,
    )


def fit_supervised_partition(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    checkpoint_id: str,
    num_classes: int = 4,
    epsilon: float = 1e-6,
    relative_tie_tolerance: float = 1e-6,
) -> SubspacePartition:
    matrix = _matrix("features", features)
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("labels must be a vector aligned with features")
    entity_rank = min(num_classes - 1, matrix.shape[1])
    scatter = compute_scatter_matrices(matrix, labels, num_classes)
    result = solve_generalized_eigenproblem(scatter, epsilon=epsilon)
    entity = _orthonormalize(torch.from_numpy(result.eigenvectors[:, :entity_rank]))
    complement = _complement_basis(entity)
    centered = matrix - matrix.mean(dim=0)
    covariance = centered.T @ centered / max(matrix.shape[0] - 1, 1)
    complement_covariance = complement.T @ covariance @ complement
    eigenvalues, eigenvectors = torch.linalg.eigh(
        (complement_covariance + complement_covariance.T) / 2
    )
    order = torch.argsort(eigenvalues, descending=True)
    complement_values = eigenvalues[order]
    complement_basis = _sign_fix(complement @ eigenvectors[:, order])
    complement_ranks = _band_ranks(complement.shape[1], 3)
    _reject_tied_boundaries(
        complement_values,
        complement_ranks,
        relative_tolerance=relative_tie_tolerance,
    )
    full_basis = torch.cat((entity, complement_basis), dim=1)
    ranks = (entity_rank, *complement_ranks)
    provenances = ("lda_entity", "complement_pca_0", "complement_pca_1", "complement_pca_2")
    return _partition_from_basis(
        full_basis,
        ranks=ranks,
        panel=Panel.SUPERVISED_LDA,
        checkpoint_id=checkpoint_id,
        provenances=provenances,
    )


def assert_corresponding_partitions(
    current: SubspacePartition, previous: SubspacePartition
) -> None:
    current_key = (
        current.panel,
        current.decomposition_version,
        tuple(band.rank for band in current.bands),
    )
    previous_key = (
        previous.panel,
        previous.decomposition_version,
        tuple(band.rank for band in previous.bands),
    )
    if current_key != previous_key:
        raise ValueError("partitions do not have corresponding slots")


__all__ = [
    "EigenBoundaryTieError",
    "Panel",
    "SubspaceBand",
    "SubspacePartition",
    "assert_corresponding_partitions",
    "fit_label_free_partition",
    "fit_supervised_partition",
]
