from __future__ import annotations

from dataclasses import replace

from jepa.temporal_image_config import ImageExperimentConfig
from jepa.temporal_image_training import train_temporal_image_experiment


def _tiny_config(tmp_path, **updates: object) -> ImageExperimentConfig:
    config = ImageExperimentConfig()
    config = replace(
        config,
        data=replace(
            config.data,
            image_size=12,
            trajectory_length=6,
            num_train_trajectories=8,
            num_val_trajectories=4,
            num_test_trajectories=4,
            num_train_samples=8,
            num_val_samples=4,
            num_test_samples=4,
        ),
        model=replace(config.model, latent_dim=6, hidden_dim=8),
        training=replace(config.training, epochs=1, batch_size=4, device="cpu"),
        output=replace(config.output, root=str(tmp_path), overwrite=True),
    )
    return replace(config, **updates)


def test_temporal_image_linear_model_completes_one_optimization_step(tmp_path) -> None:
    config = _tiny_config(tmp_path)
    result = train_temporal_image_experiment(config, seed_label=0)
    assert (result.run_dir / "checkpoint.pt").exists()
    assert result.metrics["test_loss"] == result.metrics["test_loss"]


def test_temporal_image_nonlinear_model_completes_one_optimization_step(tmp_path) -> None:
    config = _tiny_config(
        tmp_path,
        model=replace(
            ImageExperimentConfig().model, architecture="nonlinear", latent_dim=6, hidden_dim=8
        ),
    )
    result = train_temporal_image_experiment(config, seed_label=0)
    assert (result.run_dir / "checkpoint.pt").exists()


def test_temporal_image_hierarchical_runs_end_to_end(tmp_path) -> None:
    config = _tiny_config(tmp_path)
    config = replace(
        config,
        model=replace(
            config.model,
            kind="hierarchical",
            latent_dim=6,
            entity_latent_dim=2,
            context_latent_dim=4,
        ),
        hierarchical=replace(config.hierarchical, short_horizon=1, long_horizon=5),
    )
    result = train_temporal_image_experiment(config, seed_label=0)
    assert (result.run_dir / "checkpoint.pt").exists()


def test_temporal_image_random_baseline_runs_without_training(tmp_path) -> None:
    config = _tiny_config(
        tmp_path,
        model=replace(ImageExperimentConfig().model, kind="random", latent_dim=6, hidden_dim=8),
    )
    result = train_temporal_image_experiment(config, seed_label=0)
    assert result.metrics["test_loss"] == result.metrics["test_loss"]


def test_static_spatial_dataset_trains_end_to_end(tmp_path) -> None:
    config = _tiny_config(tmp_path, dataset_type="static_image_spatial")
    result = train_temporal_image_experiment(config, seed_label=0)
    assert (result.run_dir / "checkpoint.pt").exists()


def test_static_spatial_control_excludes_the_none_entity(tmp_path) -> None:
    from jepa.temporal_image_config import ENTITY_NAMES
    from jepa.temporal_image_data import StaticImageDataset

    config = _tiny_config(tmp_path, dataset_type="static_image_spatial_control")
    dataset = StaticImageDataset(config.data, "train", exclude_none=True)
    assert (dataset.entities != ENTITY_NAMES.index("none")).all()
    result = train_temporal_image_experiment(config, seed_label=0)
    assert (result.run_dir / "checkpoint.pt").exists()


def test_shuffled_target_pairing_uses_a_different_trajectory(tmp_path) -> None:
    from jepa.temporal_data import make_temporal_pairs
    from jepa.temporal_image_data import build_image_dataset_splits

    config = _tiny_config(tmp_path)
    splits = build_image_dataset_splits(config.data)
    pairs = make_temporal_pairs(splits.train, horizon=1, pairing="shuffled", seed=0)
    assert (pairs.source_trajectory_id != pairs.target_trajectory_id).all()
