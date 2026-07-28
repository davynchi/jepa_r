from __future__ import annotations

import math

import pytest
import torch

from jepa.analysis.metrics import MetricValue
from jepa.analysis.quality_autograd import (
    VirtualStateMutationError,
    fused_q8_q15,
    virtual_context_step,
)
from jepa.analysis.quality_metrics import (
    METRIC_SPECS,
    QualityMetricResult,
    q10_weighted_shape_entropy,
    q11_cross_subspace_gaussian_mi,
    q13_surprise_locality,
    q14_gradient_locality,
    q18_perturbation_concentration,
    q19_jacobian_simplicity,
    validate_metric_specs,
)
from jepa.analysis.spatial_decomposition import Panel, SubspaceBand, SubspacePartition
from jepa.training.images.ijepa_spatial import build_spatial_ijepa_core


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


def test_fused_virtual_update_is_zero_at_zero_step_and_non_mutating() -> None:
    core = build_spatial_ijepa_core(
        "nonlinear", patch_dim=12, patch_latent_dim=4, num_patches=8, hidden_dim=8
    )
    core.context_encoder.eval()
    core.predictor.eval()
    core.target_encoder.eval()
    generator = torch.Generator().manual_seed(19)
    conditioning = torch.randn(2, 8, 12, generator=generator)
    refit_a = torch.randn(40, 5, 12, generator=generator)
    refit_b = torch.randn(40, 5, 12, generator=generator)
    replay = torch.randn(12, 8, 12, generator=generator)
    before = {name: value.clone() for name, value in core.context_encoder.state_dict().items()}
    result = fused_q8_q15(
        core,
        conditioning,
        torch.tensor([0, 1, 2, 3, 4]),
        torch.tensor([5, 6]),
        refit_a,
        refit_b,
        replay,
        checkpoint_id="test",
        eta=0.0,
        microbatch_size=7,
    )
    assert result.q8 == pytest.approx(0.0, abs=5e-8)
    assert result.q15 == pytest.approx(0.0, abs=1e-12)
    assert result.gradient_calls == len(conditioning)
    assert all(
        torch.equal(before[name], value)
        for name, value in core.context_encoder.state_dict().items()
    )


class _BufferMutatingEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.calls.add_(1)
        return self.linear(values)


def test_virtual_step_detects_buffer_mutation() -> None:
    core = build_spatial_ijepa_core(
        "nonlinear", patch_dim=3, patch_latent_dim=2, num_patches=4, hidden_dim=4
    )
    core.context_encoder = _BufferMutatingEncoder()
    core.target_encoder = torch.nn.Linear(3, 2)
    with pytest.raises(VirtualStateMutationError, match="virtual_state_mutation"):
        virtual_context_step(
            core,
            torch.randn(1, 4, 3),
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
        )
