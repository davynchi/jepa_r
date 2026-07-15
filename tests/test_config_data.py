from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from jepa.config import (
    DataConfig,
    ExperimentConfig,
    apply_paired_replicate,
    config_identity_hash,
    derive_paired_replicate,
    load_config,
    validate_config,
)
from jepa.data import (
    LatentDynamicsDataset,
    SyntheticSystem,
    build_dataset_splits,
    generate_system,
)


def _small_data_config(**updates: object) -> DataConfig:
    base = DataConfig(
        num_samples=6,
        validation_samples=4,
        test_samples=4,
        sequence_length=12,
        window_size=3,
        burn_in_steps=5,
        latent_state_dim=3,
        observation_dim=4,
    )
    return replace(base, **updates)


def test_base_yaml_matches_typed_defaults() -> None:
    assert load_config("configs/base.yaml") == ExperimentConfig()


def test_cli_style_overrides_take_precedence_and_parse_on_off(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"architecture": "nonlinear"}}))

    config = load_config(path, overrides={"architecture": "linear", "sg": "off", "ema": "on"})

    assert config.model.architecture == "linear"
    assert config.training.stop_gradient is False
    assert config.training.ema.enabled is True


@pytest.mark.parametrize(
    "content,match",
    [
        ({"mystery": 1}, "unknown configuration key: mystery"),
        ({"data": {"mystery": 1}}, "unknown configuration key: data.mystery"),
        ({"training": {"ema": False}}, "training.ema must be a mapping"),
    ],
)
def test_unknown_or_malformed_yaml_is_rejected(tmp_path, content, match: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(content))

    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_conflicting_alias_and_canonical_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting overrides"):
        load_config(overrides={"architecture": "linear", "model.architecture": "nonlinear"})


@pytest.mark.parametrize(
    "data,match",
    [
        (_small_data_config(sequence_length=5, window_size=3), "sequence_length"),
        (_small_data_config(observation_dim=2, latent_state_dim=3), "observation_dim"),
        (_small_data_config(transition_norm=1.0), "transition_norm"),
        (_small_data_config(process_noise=-0.1), "process_noise"),
        (_small_data_config(observation_noise=float("nan")), "observation_noise"),
        (_small_data_config(burn_in_steps=-1), "burn_in_steps"),
    ],
)
def test_invalid_data_contracts_are_rejected(data: DataConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_config(replace(ExperimentConfig(), data=data))


def test_config_identity_hash_is_stable_and_ignores_output_fields() -> None:
    base = ExperimentConfig()
    different_output = replace(base, output=replace(base.output, root="elsewhere", overwrite=True))
    different_learning_rate = replace(
        base, training=replace(base.training, learning_rate=base.training.learning_rate * 2)
    )

    assert config_identity_hash(base) == config_identity_hash(different_output)
    assert config_identity_hash(base) != config_identity_hash(different_learning_rate)


def test_paired_replicate_derivation_is_stable_and_axis_independent() -> None:
    first = derive_paired_replicate(7)
    second = derive_paired_replicate(7)
    other = derive_paired_replicate(8)

    assert first == second
    assert first != other
    assert (
        len(
            {
                first.system_seed,
                first.train_sample_seed,
                first.validation_sample_seed,
                first.test_sample_seed,
                first.training_seed,
            }
        )
        == 5
    )
    assert first.model_seed("linear") != first.model_seed("nonlinear")
    assert first.epoch_order_seed(0) != first.epoch_order_seed(1)


def test_apply_paired_replicate_does_not_depend_on_policy_axes() -> None:
    base = ExperimentConfig(data=_small_data_config())
    other_policy = replace(
        base,
        training=replace(
            base.training, stop_gradient=False, ema=replace(base.training.ema, enabled=False)
        ),
    )

    paired_base, seeds = apply_paired_replicate(base, 42)
    paired_other, other_seeds = apply_paired_replicate(other_policy, 42)

    assert seeds == other_seeds
    assert paired_base.data == paired_other.data
    assert paired_base.training.seed == paired_other.training.seed
    assert paired_base.training.stop_gradient is True
    assert paired_other.training.stop_gradient is False


def test_generated_system_is_a_contraction_with_orthonormal_observation_columns() -> None:
    config = _small_data_config(transition_norm=0.73)
    system = generate_system(config)

    torch.testing.assert_close(
        torch.linalg.matrix_norm(system.transition, ord=2),
        torch.tensor(0.73),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        system.observation.T @ system.observation,
        torch.eye(config.latent_state_dim),
        rtol=1e-5,
        atol=1e-6,
    )


def test_observation_qr_sign_is_canonicalized() -> None:
    config = _small_data_config()
    generator = torch.Generator().manual_seed(config.system_seed)
    torch.randn(
        config.latent_state_dim,
        config.latent_state_dim,
        generator=generator,
        dtype=torch.float64,
    )
    raw_observation = torch.randn(
        config.observation_dim,
        config.latent_state_dim,
        generator=generator,
        dtype=torch.float64,
    )
    q, upper = torch.linalg.qr(raw_observation, mode="reduced")
    signs = torch.where(
        torch.diagonal(upper) < 0,
        -torch.ones(config.latent_state_dim, dtype=torch.float64),
        torch.ones(config.latent_state_dim, dtype=torch.float64),
    )

    system = generate_system(config, dtype=torch.float64)

    torch.testing.assert_close(system.observation, q * signs.unsqueeze(0), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("system_kind", ["linear", "nonlinear"])
def test_dataset_shapes_finiteness_and_materialized_replay(system_kind: str) -> None:
    config = _small_data_config(system_kind=system_kind)
    dataset = LatentDynamicsDataset(config, "train")
    first = dataset[0]
    replay = dataset[0]

    assert len(dataset) == config.num_samples
    assert first.context.shape == (config.window_size, config.observation_dim)
    assert first.target.shape == (config.window_size, config.observation_dim)
    assert first.context_state.shape == (config.latent_state_dim,)
    assert first.target_state.shape == (config.latent_state_dim,)
    assert all(torch.isfinite(tensor).all() for tensor in first)
    for original, repeated in zip(first, replay, strict=True):
        torch.testing.assert_close(original, repeated, rtol=0.0, atol=0.0)


def test_named_samples_are_collated_by_pytorch_dataloader() -> None:
    config = _small_data_config()
    batch = next(iter(DataLoader(LatentDynamicsDataset(config, "train"), batch_size=2)))

    assert batch.context.shape == (2, config.window_size, config.observation_dim)
    assert batch.target.shape == (2, config.window_size, config.observation_dim)
    assert batch.context_state.shape == (2, config.latent_state_dim)
    assert batch.target_state.shape == (2, config.latent_state_dim)


def test_split_datasets_share_system_without_sharing_samples() -> None:
    config = _small_data_config()
    splits = build_dataset_splits(config)

    assert splits.train.system is splits.system
    assert splits.validation.system is splits.system
    assert splits.test.system is splits.system
    assert not torch.equal(splits.train.contexts[0], splits.validation.contexts[0])
    assert not torch.equal(splits.validation.contexts[0], splits.test.contexts[0])


def test_returned_states_match_window_endpoints_without_observation_noise() -> None:
    config = _small_data_config(observation_noise=0.0)
    dataset = LatentDynamicsDataset(config, "train")
    sample = dataset[0]

    torch.testing.assert_close(
        sample.context[-1], dataset.system.observation @ sample.context_state
    )
    torch.testing.assert_close(sample.target[-1], dataset.system.observation @ sample.target_state)


def test_custom_system_shape_is_validated_before_materialization() -> None:
    config = _small_data_config()
    invalid = SyntheticSystem(
        transition=torch.eye(config.latent_state_dim + 1),
        observation=torch.zeros(config.observation_dim, config.latent_state_dim),
    )

    with pytest.raises(ValueError, match="transition shape"):
        LatentDynamicsDataset(config, "train", system=invalid)


def test_same_configuration_regenerates_exact_samples() -> None:
    config = _small_data_config()
    first = build_dataset_splits(config)
    second = build_dataset_splits(config)

    torch.testing.assert_close(
        first.system.transition, second.system.transition, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        first.system.observation, second.system.observation, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(first.train.contexts, second.train.contexts, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.train.targets, second.train.targets, rtol=0.0, atol=0.0)


def test_different_replicates_change_system_and_samples() -> None:
    base = ExperimentConfig(data=_small_data_config())
    first_config, _ = apply_paired_replicate(base, 1)
    second_config, _ = apply_paired_replicate(base, 2)
    first = build_dataset_splits(first_config.data)
    second = build_dataset_splits(second_config.data)

    assert not torch.equal(first.system.transition, second.system.transition)
    assert not torch.equal(first.train.contexts, second.train.contexts)
