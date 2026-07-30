from __future__ import annotations

import math

import pytest
import torch

from jepa.analysis.temporal_factorization import (
    block_gaussian_statistics,
    cross_block_gaussian_mi,
    fit_orthogonal_procrustes,
    linear_cka,
    match_subspaces,
    perturbation_concentration,
)


def _permuted_block_basis() -> tuple[torch.Tensor, torch.Tensor]:
    previous = torch.eye(8, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [math.cos(0.4), -math.sin(0.4)],
            [math.sin(0.4), math.cos(0.4)],
        ],
        dtype=torch.float64,
    )
    current = previous.clone()
    current[:, :2] = previous[:, 4:6] @ rotation
    current[:, 2:4] = previous[:, :2]
    current[:, 4:6] = previous[:, 6:8]
    current[:, 6:8] = previous[:, 2:4]
    return current, previous


def test_matching_is_permutation_and_within_block_rotation_invariant() -> None:
    current, previous = _permuted_block_basis()

    metrics = match_subspaces(current, previous, num_blocks=4)

    assert metrics.assignment == (2, 0, 3, 1)
    assert metrics.similarity == pytest.approx(1.0, abs=1e-12)
    assert metrics.grassmann_distance == pytest.approx(0.0, abs=1e-8)
    assert metrics.principal_angle_max_degrees < 1e-5


def test_procrustes_and_cka_ignore_global_rotation() -> None:
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(600, 12, generator=generator, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(
        torch.randn(12, 12, generator=generator, dtype=torch.float64)
    )
    shifted = features @ rotation + 3

    fit = fit_orthogonal_procrustes(
        shifted[:400],
        features[:400],
        shifted[400:],
        features[400:],
    )

    assert fit.validation_normalized_residual < 1e-10
    assert linear_cka(features, shifted) == pytest.approx(1.0, abs=1e-12)


def test_block_gaussian_statistics_detect_isotropic_rank() -> None:
    generator = torch.Generator().manual_seed(8)
    features = torch.randn(20000, 8, generator=generator, dtype=torch.float64)

    metrics = block_gaussian_statistics(
        features,
        torch.eye(8),
        num_blocks=4,
    )

    for block in metrics["blocks"]:
        assert block["normalized_effective_rank"] > 0.999
        assert abs(block["shape_logdet_per_dimension"]) < 0.001


def test_gaussian_mi_distinguishes_independent_and_coupled_blocks() -> None:
    generator = torch.Generator().manual_seed(12)
    left = torch.randn(10000, 2, generator=generator, dtype=torch.float64)
    independent = torch.randn(10000, 2, generator=generator, dtype=torch.float64)
    coupled = left + 0.1 * torch.randn(
        10000,
        2,
        generator=generator,
        dtype=torch.float64,
    )

    independent_mi = cross_block_gaussian_mi(
        torch.cat((left, independent), dim=1),
        torch.eye(4),
        num_blocks=2,
    )
    coupled_mi = cross_block_gaussian_mi(
        torch.cat((left, coupled), dim=1),
        torch.eye(4),
        num_blocks=2,
    )

    assert independent_mi["normalized_mean"] < 0.001
    assert coupled_mi["normalized_mean"] > 1.0


def test_perturbation_concentration_detects_local_and_distributed_change() -> None:
    source = torch.zeros(100, 8, dtype=torch.float64)
    local = source.clone()
    local[:, :2] = 1
    distributed = torch.ones_like(source)

    local_metrics = perturbation_concentration(
        source,
        local,
        torch.eye(8),
        num_blocks=4,
    )
    distributed_metrics = perturbation_concentration(
        source,
        distributed,
        torch.eye(8),
        num_blocks=4,
    )

    assert local_metrics["mean_concentration"] == pytest.approx(1.0)
    assert local_metrics["mean_effective_support"] == pytest.approx(1.0)
    assert distributed_metrics["mean_concentration"] == pytest.approx(0.0)
    assert distributed_metrics["mean_effective_support"] == pytest.approx(4.0)
