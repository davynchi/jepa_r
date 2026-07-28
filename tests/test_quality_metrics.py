from __future__ import annotations

import math

import pytest
import torch

from jepa.analysis.metrics import MetricValue
from jepa.analysis.quality_metrics import (
    METRIC_SPECS,
    QualityMetricResult,
    q10_weighted_shape_entropy,
    q11_cross_subspace_gaussian_mi,
    q13_surprise_locality,
    q14_gradient_locality,
    q17_transformation_residual,
    q18_perturbation_concentration,
    q19_jacobian_simplicity,
    validate_metric_specs,
)
from jepa.analysis.spatial_decomposition import Panel, SubspaceBand, SubspacePartition


def _partition() -> SubspacePartition:
    identity = torch.eye(4, dtype=torch.float64)
    bands = tuple(
        SubspaceBand(i, 1, identity[:, i : i + 1], torch.diag(identity[:, i]), "test")
        for i in range(4)
    )
    return SubspacePartition(Panel.LABEL_FREE, 4, "checkpoint", bands)


def test_registry_is_unique_and_has_expected_primary_count() -> None:
    validate_metric_specs()
    assert len({(spec.name, spec.version) for spec in METRIC_SPECS}) == len(METRIC_SPECS)
    assert sum(spec.primary_eligible for spec in METRIC_SPECS) == 16


def test_quality_result_freezes_json_safe_diagnostics() -> None:
    result = QualityMetricResult(
        "q1_cross_covariance",
        MetricValue(0.5),
        diagnostics={"count": 2, "clamped": False, "values": (1.0, None)},
    )
    with pytest.raises(TypeError):
        result.diagnostics["other"] = 3  # type: ignore[index]
    with pytest.raises(TypeError, match="JSON-safe"):
        QualityMetricResult(
            "q1_cross_covariance", MetricValue(0.5), diagnostics={"tensor": torch.ones(1)}
        )
    with pytest.raises(ValueError, match="finite"):
        QualityMetricResult("q1_cross_covariance", MetricValue(0.5), diagnostics={"bad": math.nan})


def test_entropy_and_independent_gaussian_mi() -> None:
    generator = torch.Generator().manual_seed(8)
    features = torch.randn(5000, 4, generator=generator, dtype=torch.float64)
    entropy, bands = q10_weighted_shape_entropy(features, _partition())
    assert entropy.value == pytest.approx(0.0, abs=1e-7)
    assert bands == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-7)
    mi, _ = q11_cross_subspace_gaussian_mi(features, _partition())
    assert mi.value is not None
    assert 0 <= mi.value < 0.002


def test_locality_metrics_cover_zero_and_one_hot_energy() -> None:
    partition = _partition()
    zero = q13_surprise_locality((0.0, 0.0, 0.0, 0.0))
    assert zero.reason == "no_subspace_movement"
    concentrated = q13_surprise_locality((2.0, 0.0, 0.0, 0.0))
    assert concentrated.value == pytest.approx(1.0)
    gradients = torch.zeros(3, 4, dtype=torch.float64)
    gradients[:, 0] = 1
    q14, shares, maximum = q14_gradient_locality(gradients, partition)
    assert q14.value == pytest.approx(1.0)
    assert shares == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert maximum == 1.0


def test_q18_and_q19_known_support() -> None:
    partition = _partition()
    first = torch.zeros(10, 4, dtype=torch.float64)
    second = first.clone()
    second[:, 0] = 1
    q18, eligible, total = q18_perturbation_concentration(first, second, partition)
    assert q18.value == pytest.approx(1.0)
    assert (eligible, total) == (10, 10)
    diagonal = torch.diag(torch.tensor([4.0, 2.0, 0.0, 0.0], dtype=torch.float64))
    rank, density = q19_jacobian_simplicity(diagonal)
    probabilities = torch.tensor([0.8, 0.2], dtype=torch.float64)
    expected = float(torch.exp(-(probabilities * probabilities.log()).sum()) / 4)
    assert rank.value == pytest.approx(expected)
    assert density.value == pytest.approx(2 / 16)


def test_q17_recovers_shared_linear_mask_transformation() -> None:
    generator = torch.Generator().manual_seed(12)
    train = torch.randn(256, 4, generator=generator, dtype=torch.float64)
    validation = torch.randn(128, 4, generator=generator, dtype=torch.float64)
    transform = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.0, 0.8, 0.1, 0.0],
            [0.0, 0.0, 1.1, -0.2],
            [0.1, 0.0, 0.0, 0.9],
        ],
        dtype=torch.float64,
    )
    residual = q17_transformation_residual(
        train,
        train @ transform,
        validation,
        validation @ transform,
    )
    assert residual.value == pytest.approx(0.0, abs=1e-12)
