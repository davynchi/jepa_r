from __future__ import annotations

from dataclasses import replace

import torch

from jepa.temporal_config import EntityContextDataConfig
from jepa.temporal_data import (
    EntityContextTrajectoryDataset,
    build_counterfactual_pairs,
    build_dataset_splits,
    generate_observation_system,
    make_temporal_pairs,
)


def _small_config(**updates: object) -> EntityContextDataConfig:
    base = EntityContextDataConfig(
        num_entities=3,
        context_dim=4,
        trajectory_length=10,
        observation_dim=8,
        entity_switch_probability=0.2,
        context_rho=0.6,
        observation_noise_std=0.0,
        num_train_trajectories=40,
        num_val_trajectories=10,
        num_test_trajectories=10,
    )
    return replace(base, **updates)


def test_dataset_shapes_and_dtypes() -> None:
    config = _small_config()
    splits = build_dataset_splits(config)
    item = splits.train[0]
    assert item.observations.shape == (config.trajectory_length, config.observation_dim)
    assert item.entity.shape == (config.trajectory_length,)
    assert item.entity.dtype == torch.long
    assert item.context.shape == (config.trajectory_length, config.context_dim)
    assert item.observations.dtype == torch.float32
    assert item.trajectory_id == 0


def test_fixed_seed_reproduces_identical_trajectories() -> None:
    config = _small_config()
    first = EntityContextTrajectoryDataset(config, "train")
    second = EntityContextTrajectoryDataset(config, "train")
    assert torch.equal(first.entities, second.entities)
    assert torch.equal(first.contexts, second.contexts)
    assert torch.equal(first.observations, second.observations)


def test_splits_are_distinct() -> None:
    config = _small_config()
    splits = build_dataset_splits(config)
    assert not torch.equal(splits.train.observations[:10], splits.validation.observations[:10])
    assert not torch.equal(splits.validation.observations[:10], splits.test.observations[:10])


def test_observation_mapping_shared_across_splits() -> None:
    config = _small_config()
    splits = build_dataset_splits(config)
    assert torch.equal(splits.train.system.linear_map, splits.validation.system.linear_map)
    assert torch.equal(splits.validation.system.linear_map, splits.test.system.linear_map)


def test_nonlinear_observation_mapping_shared_across_splits() -> None:
    config = _small_config(observation_mode="nonlinear", nonlinear_hidden_dim=8)
    splits = build_dataset_splits(config)
    assert torch.equal(splits.train.system.hidden_map, splits.test.system.hidden_map)
    assert torch.equal(splits.train.system.output_map, splits.test.system.output_map)


def test_empirical_entity_switch_frequency_matches_configured_probability() -> None:
    config = _small_config(
        trajectory_length=200, num_train_trajectories=60, entity_switch_probability=0.1
    )
    dataset = EntityContextTrajectoryDataset(config, "train")
    switches = (dataset.entities[:, 1:] != dataset.entities[:, :-1]).float().mean().item()
    assert abs(switches - config.entity_switch_probability) < 0.03


def test_context_autocorrelation_matches_configured_rho() -> None:
    config = _small_config(trajectory_length=200, num_train_trajectories=80, context_rho=0.7)
    dataset = EntityContextTrajectoryDataset(config, "train")
    c = dataset.contexts
    numerator = (c[:, 1:] * c[:, :-1]).mean()
    denominator = (c * c).mean()
    empirical_rho = (numerator / denominator).item()
    assert abs(empirical_rho - config.context_rho) < 0.05


def test_temporal_pairs_come_from_the_same_trajectory() -> None:
    config = _small_config()
    dataset = EntityContextTrajectoryDataset(config, "train")
    pairs = make_temporal_pairs(dataset, horizon=2, pairing="temporal", seed=0)
    assert torch.equal(pairs.source_trajectory_id, pairs.target_trajectory_id)


def test_shuffled_pairs_come_from_different_trajectories() -> None:
    config = _small_config()
    dataset = EntityContextTrajectoryDataset(config, "train")
    pairs = make_temporal_pairs(dataset, horizon=2, pairing="shuffled", seed=0)
    assert torch.all(pairs.source_trajectory_id != pairs.target_trajectory_id)


def test_counterfactual_same_entity_pairs_hold_entity_fixed_and_vary_context() -> None:
    # Only entity is shared between x1/x2; the sampled contexts must differ.
    config = _small_config()
    system = generate_observation_system(config)
    pairs = build_counterfactual_pairs(config, system, num_pairs=50, seed=1)
    assert not torch.equal(pairs.same_entity_context_1, pairs.same_entity_context_2)
    assert not torch.equal(pairs.same_entity_x1, pairs.same_entity_x2)


def test_counterfactual_diff_entity_pairs_hold_context_fixed_and_vary_entity() -> None:
    # Only context is shared between x1/x2; the sampled entities must differ.
    config = _small_config()
    system = generate_observation_system(config)
    pairs = build_counterfactual_pairs(config, system, num_pairs=50, seed=1)
    entities = pairs.diff_entity_entities
    assert torch.all(entities[:, 0] != entities[:, 1])
    assert not torch.equal(pairs.diff_entity_x1, pairs.diff_entity_x2)
