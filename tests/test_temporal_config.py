from __future__ import annotations

import pytest

from jepa.temporal_config import TemporalExperimentConfig, load_temporal_config


def test_base_yaml_loads_and_validates() -> None:
    config = load_temporal_config("configs/temporal_base.yaml")
    assert config.data.num_entities == 3
    assert config.model.latent_dim == 32


def test_unknown_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  mystery: 1\n")
    with pytest.raises(ValueError, match="unknown configuration key"):
        load_temporal_config(path)


def test_hierarchical_dims_must_sum_to_latent_dim() -> None:
    with pytest.raises(ValueError, match="entity_latent_dim"):
        load_temporal_config(
            overrides={
                "model.kind": "hierarchical",
                "model.latent_dim": "10",
                "model.entity_latent_dim": "8",
                "model.context_latent_dim": "8",
            }
        )


def test_overrides_apply() -> None:
    default = TemporalExperimentConfig()
    config = load_temporal_config(overrides={"data.entity_switch_probability": "0.2"})
    assert config.data.entity_switch_probability == pytest.approx(0.2)
    assert config.model == default.model
    assert config.training == default.training
