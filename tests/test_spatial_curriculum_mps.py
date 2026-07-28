from __future__ import annotations

import pytest
import torch

from jepa.analysis.quality_metrics import q14_gradient_locality
from jepa.analysis.spatial_decomposition import Panel, SubspaceBand, SubspacePartition
from jepa.training.images.ijepa_spatial import MaskConfig, build_spatial_ijepa_core
from jepa.training.images.spatial_curriculum import (
    SpatialWeightingState,
    _coordinate_importance_from_covariance,
    _coordinate_importance_from_dynamics,
    _coordinate_importance_from_transformation,
    richness_from_patches,
    score_frames_by_coordinate_importance,
)


def test_coordinate_statistics_move_to_cpu_float64() -> None:
    latents = torch.randn(8, 4)
    weights, basis, _ = _coordinate_importance_from_covariance(latents, delta=1e-6)
    assert weights.device.type == basis.device.type == "cpu"
    assert weights.dtype == basis.dtype == torch.float64


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_richness_and_coordinate_scoring_do_not_create_float64_mps_tensors() -> None:
    device = torch.device("mps")
    core = build_spatial_ijepa_core(
        "nonlinear", patch_dim=3, patch_latent_dim=4, num_patches=16, hidden_dim=8
    )
    core.context_encoder.to(device)
    core.predictor.to(device)
    core.target_encoder.to(device)
    patches = torch.randn(6, 16, 3, device=device)

    richness, _ = richness_from_patches(
        core,
        patches[:4],
        functional="logdet",
        delta=1e-4,
        trace_target=1.0,
        trace_beta=0.01,
    )
    gradients = torch.autograd.grad(richness, tuple(core.context_encoder.parameters()))
    assert richness.device.type == "mps"
    assert richness.dtype == torch.float32
    assert all(gradient.device.type == "mps" for gradient in gradients)
    identity = torch.eye(4, dtype=torch.float64)
    partition = SubspacePartition(
        Panel.LABEL_FREE,
        4,
        "mps-test",
        tuple(
            SubspaceBand(i, 1, identity[:, i : i + 1], torch.diag(identity[:, i]), "test")
            for i in range(4)
        ),
    )
    locality, shares, _ = q14_gradient_locality(torch.randn(3, 4, device=device), partition)
    assert locality.value is not None
    assert sum(shares) == pytest.approx(1.0)

    for method in ("covariance", "transformation", "dynamics"):
        state = SpatialWeightingState(
            torch.zeros(6, dtype=torch.float64),
            torch.full((6,), 1 / 6, dtype=torch.float64),
            torch.zeros(6, dtype=torch.float64),
        )
        pairs = (patches[:4], patches[2:6]) if method == "transformation" else None
        scores, _, weights, previous = score_frames_by_coordinate_importance(
            core,
            patches,
            ref_indices=torch.arange(4),
            state=state,
            grid=4,
            mask_config=MaskConfig(),
            batch_size=3,
            seed=7,
            device=device,
            coordinate_importance=method,
            coordinate_ema_beta=0.1,
            coordinate_delta=1e-6,
            transform_pair_patches=pairs,
        )
        assert scores.device.type == weights.device.type == previous.device.type == "cpu"
        assert scores.dtype == torch.float64


def test_coordinate_helpers_accept_detached_mps_contract_on_cpu() -> None:
    source = torch.randn(8, 4)
    target = torch.randn(8, 4)
    weights, basis, _ = _coordinate_importance_from_transformation(source, target, delta=1e-6)
    assert weights.shape == (4,)
    assert basis.shape == (4, 4)
    state = SpatialWeightingState(
        torch.zeros(8, dtype=torch.float64),
        torch.full((8,), 1 / 8, dtype=torch.float64),
        torch.zeros(8, dtype=torch.float64),
    )
    dynamics, identity, metadata = _coordinate_importance_from_dynamics(
        source, state, ema_beta=0.1, delta=1e-6
    )
    assert dynamics.shape == (4,)
    assert torch.equal(identity, torch.eye(4, dtype=torch.float64))
    assert metadata["coord/dynamics_cold_start"] == 1.0
