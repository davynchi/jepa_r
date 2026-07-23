from __future__ import annotations

import pytest
import torch

from jepa.training.images import spatial_curriculum as current
from legacy import spatial_curriculum as legacy


@pytest.fixture(params=(current, legacy), ids=("current", "legacy"))
def curriculum(request):
    return request.param


def test_default_mapping_remains_zscore_softmax(curriculum) -> None:
    state = curriculum.init_spatial_weighting(3)
    config = curriculum.SpatialWeightingConfig(
        temperature=2.0,
        uniform_mix=0.0,
    )
    scores = torch.tensor([-1.0, 0.0, 2.0])
    updated = curriculum.update_spatial_weights(state, scores, config)

    normalized = (scores.double() - scores.double().mean()) / scores.double().std(unbiased=False)
    expected = torch.softmax(normalized / 2.0, dim=0)
    assert torch.allclose(updated.probabilities, expected)


def test_robust_mapping_clips_normalized_outlier(curriculum) -> None:
    state = curriculum.init_spatial_weighting(101)
    scores = torch.linspace(-1.0, 1.0, 101)
    scores[-1] = 10_000.0
    config = curriculum.SpatialWeightingConfig(
        score_normalization="robust",
        score_clip=3.0,
        uniform_mix=0.0,
    )
    updated = curriculum.update_spatial_weights(state, scores, config)

    assert updated.memory.abs().max() <= 3.0
    assert updated.probabilities.argmax() == 100


def test_adaptive_temperature_enforces_target_ess(curriculum) -> None:
    num_samples = 1000
    state = curriculum.init_spatial_weighting(num_samples)
    scores = torch.zeros(num_samples)
    scores[0] = 1000.0
    config = curriculum.SpatialWeightingConfig(
        uniform_mix=0.0,
        target_ess_fraction=0.2,
    )
    updated = curriculum.update_spatial_weights(state, scores, config)
    diagnostics = curriculum.weighting_diagnostics(updated)

    assert diagnostics["weighting/effective_sample_size"] >= 0.2 * num_samples - 1e-6
    assert diagnostics["weighting/sampling_temperature"] > config.temperature
    assert updated.probabilities.argmax() == 0


def test_cosine_ras_removes_gradient_magnitude(curriculum) -> None:
    richness = (torch.tensor([-3.0, -4.0]),)
    richness_norm = torch.tensor(5.0, dtype=torch.float64)
    small = (torch.tensor([0.3, 0.4]),)
    large = (torch.tensor([30.0, 40.0]),)

    small_score, _ = curriculum._ras_gradient_score(
        small,
        richness,
        alignment="cosine",
        richness_gradient_norm=richness_norm,
    )
    large_score, _ = curriculum._ras_gradient_score(
        large,
        richness,
        alignment="cosine",
        richness_gradient_norm=richness_norm,
    )

    assert torch.allclose(small_score, torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(large_score, torch.tensor(1.0, dtype=torch.float64))
