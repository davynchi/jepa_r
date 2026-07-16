from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from jepa.temporal_data import make_temporal_pairs
from jepa.temporal_shapes3d_config import Shapes3DDataConfig, Shapes3DExperimentConfig

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "3dshapes.h5"
_EXPECTED_SIZE = 267_573_662  # official 3dshapes.h5 size in bytes
_data_ready = DATA_PATH.exists() and DATA_PATH.stat().st_size == _EXPECTED_SIZE
requires_shapes3d = pytest.mark.skipif(
    not _data_ready, reason="data/3dshapes.h5 not downloaded or incomplete (see README)"
)


def _small_config(**updates: object) -> Shapes3DDataConfig:
    base = Shapes3DDataConfig(
        h5_path=str(DATA_PATH),
        trajectory_length=8,
        entity_switch_probability=0.2,
        num_train_trajectories=10,
        num_val_trajectories=5,
        num_test_trajectories=5,
        num_train_samples=10,
        num_val_samples=5,
        num_test_samples=5,
    )
    return replace(base, **updates)


@requires_shapes3d
def test_dataset_shapes_and_dtypes() -> None:
    from jepa.temporal_shapes3d_data import Shapes3DEntityContextTrajectoryDataset

    config = _small_config()
    dataset = Shapes3DEntityContextTrajectoryDataset(config, "train")
    assert dataset.frames.shape == (10, 8, 3, 64, 64)
    assert dataset.entities.dtype == torch.long
    assert dataset.entities.max() < config.num_entities
    assert dataset.contexts.shape == (10, 8, config.context_dim)
    assert dataset.frames.min() >= 0.0 and dataset.frames.max() <= 1.0


@requires_shapes3d
def test_fixed_seed_reproduces_identical_trajectories() -> None:
    from jepa.temporal_shapes3d_data import Shapes3DEntityContextTrajectoryDataset

    config = _small_config()
    first = Shapes3DEntityContextTrajectoryDataset(config, "train")
    second = Shapes3DEntityContextTrajectoryDataset(config, "train")
    assert torch.equal(first.entities, second.entities)
    assert torch.equal(first.frames, second.frames)


@requires_shapes3d
def test_empirical_entity_switch_frequency_matches_configured_probability() -> None:
    from jepa.temporal_shapes3d_data import Shapes3DEntityContextTrajectoryDataset

    config = _small_config(
        trajectory_length=200, num_train_trajectories=30, entity_switch_probability=0.1
    )
    dataset = Shapes3DEntityContextTrajectoryDataset(config, "train")
    switches = (dataset.entities[:, 1:] != dataset.entities[:, :-1]).float().mean().item()
    assert abs(switches - config.entity_switch_probability) < 0.04


@requires_shapes3d
def test_temporal_pairs_come_from_the_same_trajectory() -> None:
    from jepa.temporal_shapes3d_data import Shapes3DEntityContextTrajectoryDataset

    config = _small_config()
    dataset = Shapes3DEntityContextTrajectoryDataset(config, "train")
    pairs = make_temporal_pairs(dataset, horizon=2, pairing="temporal", seed=0)
    assert torch.equal(pairs.source_trajectory_id, pairs.target_trajectory_id)


@requires_shapes3d
def test_counterfactual_pairs_hold_the_right_factor_fixed() -> None:
    from jepa.temporal_shapes3d_data import Shapes3DSource, build_shapes3d_counterfactual_pairs

    config = _small_config()
    source = Shapes3DSource(config.h5_path)
    pairs = build_shapes3d_counterfactual_pairs(config, source, num_pairs=10, seed=1)
    assert not torch.equal(pairs.same_entity_x1, pairs.same_entity_x2)
    entities = pairs.diff_entity_entities
    assert torch.all(entities[:, 0] != entities[:, 1])


@requires_shapes3d
def test_block_masking_reused_from_procedural_image_world() -> None:
    from jepa.temporal_image_config import SpatialMaskingConfig
    from jepa.temporal_shapes3d_data import Shapes3DStaticImageDataset, build_static_spatial_dataset

    config = _small_config()
    dataset = Shapes3DStaticImageDataset(config, "train")
    spatial = build_static_spatial_dataset(
        dataset, SpatialMaskingConfig(block_fraction=0.25), seed=0
    )
    assert torch.equal(spatial.visible + spatial.target, dataset.images)


@requires_shapes3d
def test_one_optimization_step_completes(tmp_path) -> None:
    from jepa.temporal_shapes3d_training import train_shapes3d_experiment

    config = Shapes3DExperimentConfig()
    config = replace(
        config,
        data=_small_config(),
        model=replace(config.model, latent_dim=4, hidden_dim=8),
        training=replace(config.training, epochs=1, batch_size=4, device="cpu"),
        output=replace(config.output, root=str(tmp_path), overwrite=True),
    )
    result = train_shapes3d_experiment(config, seed_label=0)
    assert (result.run_dir / "checkpoint.pt").exists()
    assert result.metrics["test_loss"] == result.metrics["test_loss"]
