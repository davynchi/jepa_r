from __future__ import annotations

import torch

from jepa.training.images.bandit_weighting import (
    DiscountedLinearThompsonSampler,
    DiscountedRewardNormalizer,
    LatentContextCache,
    LinearThompsonConfig,
    RichnessGradientSnapshot,
    batch_ras_from_parameter_gradients,
)


def test_linear_thompson_learns_reward_direction() -> None:
    sampler = DiscountedLinearThompsonSampler(
        LinearThompsonConfig(
            context_dim=2,
            discount=1.0,
            exploration_scale=0.0,
            fit_intercept=False,
        )
    )
    for _ in range(30):
        sampler.update_aggregate(torch.tensor([1.0, 0.0]), 2.0)
        sampler.update_aggregate(torch.tensor([0.0, 1.0]), -1.0)
    scores = sampler.mean_scores(torch.eye(2))
    assert scores[0] > 1.8
    assert scores[1] < -0.9


def test_probabilities_are_finite_normalized_and_mixed() -> None:
    sampler = DiscountedLinearThompsonSampler(
        LinearThompsonConfig(
            context_dim=2,
            exploration_scale=0.0,
            uniform_mix=0.2,
        )
    )
    probabilities = sampler.probabilities_from_scores(torch.tensor([-100.0, 0.0, 100.0]))
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert probabilities.min() >= 0.2 / 3


def test_batch_update_uses_mean_context() -> None:
    config = LinearThompsonConfig(
        context_dim=2,
        discount=1.0,
        exploration_scale=0.0,
        fit_intercept=False,
    )
    batch_sampler = DiscountedLinearThompsonSampler(config)
    aggregate_sampler = DiscountedLinearThompsonSampler(config)
    contexts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    batch_sampler.update_batch(contexts, 3.0)
    aggregate_sampler.update_aggregate(contexts.mean(dim=0), 3.0)
    assert torch.allclose(batch_sampler.precision, aggregate_sampler.precision)
    assert torch.allclose(batch_sampler.information, aggregate_sampler.information)


def test_sampler_state_round_trip() -> None:
    sampler = DiscountedLinearThompsonSampler(LinearThompsonConfig(context_dim=3, discount=0.9))
    sampler.update_batch(torch.randn(4, 3), 1.5)
    restored = DiscountedLinearThompsonSampler(sampler.config)
    restored.load_state_dict(sampler.state_dict())
    assert torch.equal(restored.precision, sampler.precision)
    assert torch.equal(restored.information, sampler.information)
    assert restored.num_updates == sampler.num_updates


def test_reward_normalizer_tracks_and_restores_state() -> None:
    normalizer = DiscountedRewardNormalizer(decay=0.9)
    assert normalizer.update(2.0) == 0.0
    assert normalizer.update(4.0) > 0
    restored = DiscountedRewardNormalizer(decay=0.9)
    restored.load_state_dict(normalizer.state_dict())
    assert restored.state_dict() == normalizer.state_dict()


def test_cache_updates_duplicates_with_one_ema_step() -> None:
    cache = LatentContextCache(4, 2, ema_beta=0.5)
    cache.update(
        torch.tensor([1, 1, 2]),
        torch.tensor([[1.0, 1.0], [3.0, 3.0], [4.0, 2.0]]),
        step=5,
    )
    assert torch.allclose(cache.contexts[1], torch.tensor([2.0, 2.0]))
    assert cache.observation_counts[1] == 2
    cache.update(torch.tensor([1]), torch.tensor([[4.0, 0.0]]), step=7)
    assert torch.allclose(cache.contexts[1], torch.tensor([3.0, 1.0]))
    assert cache.age(step=8)[1] == 1


def test_cache_coverage_and_unknown_age() -> None:
    cache = LatentContextCache(4, 2)
    cache.update(torch.tensor([0, 2]), torch.ones(2, 2), step=3)
    assert cache.coverage == 0.5
    assert cache.age(step=5).tolist() == [2, 6, 2, 6]


def test_cache_state_round_trip() -> None:
    cache = LatentContextCache(3, 2, ema_beta=0.25)
    cache.update(torch.tensor([1]), torch.tensor([[2.0, 3.0]]), step=4)
    restored = LatentContextCache(3, 2, ema_beta=0.25)
    restored.load_state_dict(cache.state_dict())
    assert torch.equal(restored.contexts, cache.contexts)
    assert torch.equal(restored.observation_counts, cache.observation_counts)
    assert torch.equal(restored.last_seen_steps, cache.last_seen_steps)


def test_batch_ras_reuses_existing_parameter_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    loss = (parameter.square()).sum()
    loss.backward()
    snapshot = RichnessGradientSnapshot(
        gradients=(torch.tensor([3.0, -1.0]),),
        metadata={},
        step=0,
    )
    reward = batch_ras_from_parameter_gradients((parameter,), snapshot)
    assert torch.allclose(reward, torch.tensor(-2.0))
