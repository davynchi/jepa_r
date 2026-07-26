from __future__ import annotations

import torch

from jepa.video_sampling import AdaptiveSampleState, VideoSamplingConfig, percentile_ranks


def test_percentile_ranks_handle_ties() -> None:
    ranks = percentile_ranks(torch.tensor([1.0, 1.0, 2.0, 3.0]))
    assert torch.allclose(ranks[:2], torch.tensor([1 / 6, 1 / 6], dtype=torch.float64))
    assert float(ranks[-1]) == 1.0


def test_progress_quarantine_weights_are_finite_and_nonuniform() -> None:
    config = VideoSamplingConfig(
        strategy="progress_quarantine_surprise",
        warmup_epochs=0,
        uniform_mix=0.2,
        loss_ema_beta=0.0,
        progress_ema_beta=0.0,
        quarantine_percentile=0.8,
    )
    state = AdaptiveSampleState(5, config)
    indices = torch.arange(5)
    state.update(indices, torch.tensor([0.1, 0.2, 0.4, 0.8, 2.0]))
    state.update(indices, torch.tensor([0.09, 0.15, 0.30, 0.79, 2.0]))
    weights = state.weights(epoch=1)
    assert torch.isfinite(weights).all()
    assert torch.isclose(weights.mean(), torch.tensor(1.0, dtype=torch.float64))
    assert float(weights.max()) > float(weights.min())
    # Highest-loss sample made no progress and should be quarantined.
    assert float(weights[-1]) < float(weights[2])


def test_uniform_strategy_stays_uniform() -> None:
    state = AdaptiveSampleState(7, VideoSamplingConfig(strategy="uniform_shuffle"))
    assert torch.equal(state.weights(epoch=10), torch.ones(7, dtype=torch.float64))
