from __future__ import annotations

from dataclasses import replace

import torch

from jepa.temporal_config import TemporalExperimentConfig
from jepa.temporal_training import (
    build_hierarchical_core,
    train_temporal_experiment,
)
from jepa.training import build_jepa_core


def _tiny_config(tmp_path, **model_updates: object) -> TemporalExperimentConfig:
    config = TemporalExperimentConfig()
    config = replace(
        config,
        data=replace(
            config.data,
            num_entities=3,
            context_dim=3,
            trajectory_length=8,
            observation_dim=6,
            num_train_trajectories=16,
            num_val_trajectories=8,
            num_test_trajectories=8,
        ),
        model=replace(config.model, latent_dim=6, hidden_dim=8, **model_updates),
        training=replace(
            config.training, epochs=1, batch_size=4, device="cpu", prediction_horizon=1
        ),
        output=replace(config.output, root=str(tmp_path), overwrite=True),
    )
    return config


def test_linear_model_completes_one_optimization_step(tmp_path) -> None:
    config = _tiny_config(tmp_path)
    core = build_jepa_core(
        "linear",
        input_dim=config.data.observation_dim,
        latent_dim=config.model.latent_dim,
        stop_gradient=True,
        ema_enabled=True,
    )
    before = [p.detach().clone() for p in core.context_encoder.parameters()]
    optimizer = torch.optim.Adam(core.context_encoder.parameters(), lr=0.1)
    context = torch.randn(4, config.data.observation_dim)
    target = torch.randn(4, config.data.observation_dim)
    from jepa.training import compute_loss

    optimizer.zero_grad()
    forward = compute_loss(core, context, target)
    forward.loss.backward()
    optimizer.step()
    after = list(core.context_encoder.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_nonlinear_model_completes_one_optimization_step(tmp_path) -> None:
    config = _tiny_config(tmp_path, architecture="nonlinear")
    result = train_temporal_experiment(config, seed_label=0)
    assert result.metrics["test_loss"] == result.metrics["test_loss"]  # finite, not NaN
    assert (result.run_dir / "checkpoint.pt").exists()


def test_standard_run_end_to_end(tmp_path) -> None:
    config = _tiny_config(tmp_path)
    result = train_temporal_experiment(config, seed_label=0)
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "config.yaml").exists()
    assert (result.run_dir / "history.json").exists()


def test_hierarchical_run_end_to_end(tmp_path) -> None:
    config = _tiny_config(tmp_path, kind="hierarchical", entity_latent_dim=2, context_latent_dim=4)
    config = replace(
        config, hierarchical=replace(config.hierarchical, short_horizon=1, long_horizon=5)
    )
    result = train_temporal_experiment(config, seed_label=0)
    assert (result.run_dir / "checkpoint.pt").exists()


def test_random_baseline_runs_without_training(tmp_path) -> None:
    config = _tiny_config(tmp_path, kind="random")
    result = train_temporal_experiment(config, seed_label=0)
    assert result.metrics["test_loss"] == result.metrics["test_loss"]


def test_entity_labels_are_not_passed_into_the_training_forward_call(tmp_path) -> None:
    from jepa.temporal_data import build_dataset_splits
    from jepa.temporal_training import _standard_epoch_loss

    config = _tiny_config(tmp_path)
    datasets = build_dataset_splits(config.data)
    core = build_jepa_core(
        config.model.architecture,
        input_dim=config.data.observation_dim,
        latent_dim=config.model.latent_dim,
        stop_gradient=True,
        ema_enabled=True,
    )
    optimizer = torch.optim.Adam(core.context_encoder.parameters(), lr=0.01)
    captured: list[torch.Tensor] = []
    handle = core.context_encoder.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    try:
        _standard_epoch_loss(
            core, datasets.train, config, torch.device("cpu"), 1, train=True, optimizer=optimizer
        )
    finally:
        handle.remove()
    assert captured, "encoder forward was never called"
    for tensor in captured:
        assert tensor.dtype.is_floating_point
        assert tensor.shape[-1] == config.data.observation_dim


def test_hierarchical_core_builds_target_encoder_per_policy() -> None:
    config = TemporalExperimentConfig()
    config = replace(config, model=replace(config.model, kind="hierarchical"))
    core = build_hierarchical_core(config)
    assert core.policy.separate_target is True
    assert core.target_encoder is not core.online.encoder
