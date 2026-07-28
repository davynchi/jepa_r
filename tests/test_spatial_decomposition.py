from __future__ import annotations

import inspect
import math

import pytest
import torch

from jepa.analysis.quality_metrics import (
    partition_incompleteness,
    projector_overlap,
    q1_cross_covariance,
    q2_projector_interaction,
    q3_projector_overlap,
    q4_partition_incompleteness,
    q5_q6_subspace_stability,
)
from jepa.analysis.spatial_decomposition import (
    EigenBoundaryTieError,
    Panel,
    SubspaceBand,
    SubspacePartition,
    fit_label_free_partition,
)


def _coordinate_partition(dim: int = 4) -> SubspacePartition:
    basis = torch.eye(dim, dtype=torch.float64)
    bands = tuple(
        SubspaceBand(slot, 1, basis[:, slot : slot + 1], torch.diag(basis[:, slot]), "test")
        for slot in range(dim)
    )
    return SubspacePartition(Panel.LABEL_FREE, dim, "checkpoint", bands)


def test_q1_q2_and_complete_partition_identities() -> None:
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(200, 4, generator=generator, dtype=torch.float64)
    features[:, 1] += 0.7 * features[:, 0]
    partition = _coordinate_partition()
    assert q1_cross_covariance(features, partition) == pytest.approx(
        q2_projector_interaction(features, partition), abs=1e-12
    )
    assert q3_projector_overlap(partition) == pytest.approx(0.0, abs=1e-12)
    assert q4_partition_incompleteness(partition) == pytest.approx(0.0, abs=1e-12)


def test_overlap_and_incompleteness_have_known_positive_values() -> None:
    e0 = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    e1 = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    assert projector_overlap((e0, e0)) == pytest.approx(2**0.5)
    assert partition_incompleteness((e0 @ e0.T,), 2) == pytest.approx(2**-0.5)
    assert partition_incompleteness((e0 @ e0.T, e1 @ e1.T), 2) == 0.0


def test_partition_rejects_invalid_projector_and_coverage() -> None:
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="projector"):
        SubspaceBand(0, 1, basis, torch.eye(2, dtype=torch.float64), "bad")
    band = SubspaceBand(0, 1, basis, basis @ basis.T, "incomplete")
    with pytest.raises(ValueError, match="cover"):
        SubspacePartition(Panel.LABEL_FREE, 2, "checkpoint", (band,))


def test_label_free_api_has_no_label_or_factor_argument() -> None:
    parameters = inspect.signature(fit_label_free_partition).parameters
    assert "labels" not in parameters
    assert "factors" not in parameters


def test_label_free_partition_is_deterministic_and_orthogonal() -> None:
    generator = torch.Generator().manual_seed(7)
    signal = torch.randn(500, 4, generator=generator, dtype=torch.float64)
    scales = torch.tensor([4.0, 2.0, 1.0, 0.25], dtype=torch.float64)
    view_a = signal * scales + 0.02 * torch.randn(500, 4, generator=generator, dtype=torch.float64)
    view_b = signal * scales + 0.02 * torch.randn(500, 4, generator=generator, dtype=torch.float64)
    first = fit_label_free_partition(view_a, view_b, checkpoint_id="a")
    second = fit_label_free_partition(view_a, view_b, checkpoint_id="a")
    assert torch.equal(first.bands[0].basis, second.bands[0].basis)
    assert q3_projector_overlap(first) == pytest.approx(0.0, abs=1e-12)
    assert q4_partition_incompleteness(first) == pytest.approx(0.0, abs=1e-12)


def test_tied_band_boundary_fails_closed() -> None:
    rows = torch.eye(4, dtype=torch.float64).repeat(20, 1)
    with pytest.raises(EigenBoundaryTieError):
        fit_label_free_partition(rows, rows, checkpoint_id="tie")


def test_q5_q6_identity_and_rotation_invariance() -> None:
    previous = _coordinate_partition()
    angle = 0.3
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0, 0.0],
            [math.sin(angle), math.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    bands = tuple(
        SubspaceBand(
            slot,
            1,
            rotation[:, slot : slot + 1],
            rotation[:, slot : slot + 1] @ rotation[:, slot : slot + 1].T,
            "rotated",
        )
        for slot in range(4)
    )
    current = SubspacePartition(Panel.LABEL_FREE, 4, "next", bands)
    q5, q6, _ = q5_q6_subspace_stability(current, previous)
    assert q6 * q6 == pytest.approx(1 - q5, abs=1e-12)
