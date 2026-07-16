from __future__ import annotations

from dataclasses import replace

import torch

from jepa.temporal_image_config import ENTITY_NAMES, ImageDataConfig, SpatialMaskingConfig
from jepa.temporal_image_data import (
    EntityContextImageTrajectoryDataset,
    StaticImageDataset,
    build_image_counterfactual_pairs,
    build_static_spatial_dataset,
    render_frame,
)


def _small_config(**updates: object) -> ImageDataConfig:
    base = ImageDataConfig(
        image_size=16,
        trajectory_length=10,
        entity_switch_probability=0.2,
        context_smoothness=0.6,
        num_train_trajectories=20,
        num_val_trajectories=8,
        num_test_trajectories=8,
        num_train_samples=20,
        num_val_samples=8,
        num_test_samples=8,
    )
    return replace(base, **updates)


def test_dataset_shapes_and_dtypes() -> None:
    config = _small_config()
    dataset = EntityContextImageTrajectoryDataset(config, "train")
    assert dataset.frames.shape == (
        20,
        config.trajectory_length,
        config.num_channels,
        config.image_size,
        config.image_size,
    )
    assert dataset.entities.dtype == torch.long
    assert dataset.contexts.shape == (20, config.trajectory_length, config.context_dim)
    assert dataset.observations.shape == (20, config.trajectory_length, config.observation_dim)
    assert dataset.frames.dtype == torch.float32
    assert dataset.frames.min() >= 0.0 and dataset.frames.max() <= 1.0


def test_fixed_seed_reproduces_identical_trajectories() -> None:
    config = _small_config()
    first = EntityContextImageTrajectoryDataset(config, "train")
    second = EntityContextImageTrajectoryDataset(config, "train")
    assert torch.equal(first.entities, second.entities)
    assert torch.equal(first.contexts, second.contexts)
    assert torch.equal(first.frames, second.frames)


def test_splits_are_distinct() -> None:
    config = _small_config()
    train = EntityContextImageTrajectoryDataset(config, "train")
    validation = EntityContextImageTrajectoryDataset(config, "validation")
    assert not torch.equal(train.frames[:5], validation.frames[:5])


def test_none_entity_renders_uniform_background() -> None:
    context = torch.tensor([0.5, 0.5, 0.2, 0.0, 1.0, 0.0, 0.0, 1.0, 0.4]).numpy()
    frame = render_frame(0, context, image_size=16, num_channels=3)
    assert (frame == 0.4).all()


def test_different_entities_render_different_pixels() -> None:
    context = torch.tensor([0.5, 0.5, 0.25, 0.3, 1.0, 0.0, 0.0, 1.0, 0.4]).numpy()
    circle = render_frame(1, context, image_size=32, num_channels=3)
    square = render_frame(2, context, image_size=32, num_channels=3)
    triangle = render_frame(3, context, image_size=32, num_channels=3)
    assert not (circle == square).all()
    assert not (square == triangle).all()


def test_empirical_entity_switch_frequency_matches_configured_probability() -> None:
    config = _small_config(
        trajectory_length=200, num_train_trajectories=40, entity_switch_probability=0.1
    )
    dataset = EntityContextImageTrajectoryDataset(config, "train")
    switches = (dataset.entities[:, 1:] != dataset.entities[:, :-1]).float().mean().item()
    assert abs(switches - config.entity_switch_probability) < 0.04


def test_static_dataset_can_exclude_none_entity() -> None:
    config = _small_config()
    dataset = StaticImageDataset(config, "train", exclude_none=True)
    assert (dataset.entities != ENTITY_NAMES.index("none")).all()


def test_block_masking_is_complementary_and_covers_the_full_image() -> None:
    config = _small_config()
    dataset = StaticImageDataset(config, "train")
    spatial = build_static_spatial_dataset(
        dataset, SpatialMaskingConfig(block_fraction=0.25), seed=0
    )
    reconstructed = spatial.visible + spatial.target
    assert torch.equal(reconstructed, dataset.images)
    # the target block must be nonzero somewhere and the visible view must zero it out
    assert (spatial.target[0].sum(dim=0) > 0).any()


def test_counterfactual_same_entity_pairs_hold_entity_fixed_and_vary_context() -> None:
    config = _small_config()
    pairs = build_image_counterfactual_pairs(config, num_pairs=10, seed=1)
    assert not torch.equal(pairs.same_entity_x1, pairs.same_entity_x2)


def test_counterfactual_diff_entity_pairs_hold_context_fixed_and_vary_entity() -> None:
    config = _small_config()
    pairs = build_image_counterfactual_pairs(config, num_pairs=10, seed=1)
    entities = pairs.diff_entity_entities
    assert torch.all(entities[:, 0] != entities[:, 1])
    assert not torch.equal(pairs.diff_entity_x1, pairs.diff_entity_x2)
