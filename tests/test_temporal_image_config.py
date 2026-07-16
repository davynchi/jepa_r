from __future__ import annotations

import pytest

from jepa.temporal_image_config import ImageExperimentConfig, load_image_config


def test_quick_preset_loads_and_validates() -> None:
    config = load_image_config("configs/temporal_hierarchy_image_quick.yaml")
    assert config.dataset_type == "temporal_image"
    assert config.data.num_entities == 4
    assert config.data.context_dim == 9


def test_static_presets_load() -> None:
    spatial = load_image_config("configs/spatial_semantics_image_quick.yaml")
    control = load_image_config("configs/spatial_semantics_image_quick_control.yaml")
    assert spatial.dataset_type == "static_image_spatial"
    assert control.dataset_type == "static_image_spatial_control"


def test_shuffled_preset_uses_shuffled_pairing() -> None:
    config = load_image_config("configs/temporal_hierarchy_image_quick_shuffled.yaml")
    assert config.training.target_pairing == "shuffled"


def test_unknown_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  mystery: 1\n")
    with pytest.raises(ValueError, match="unknown configuration key"):
        load_image_config(path)


def test_hierarchical_dims_must_sum_to_latent_dim() -> None:
    with pytest.raises(ValueError, match="entity_latent_dim"):
        load_image_config(
            overrides={
                "model.kind": "hierarchical",
                "model.latent_dim": "10",
                "model.entity_latent_dim": "8",
                "model.context_latent_dim": "8",
            }
        )


def test_default_config_is_valid() -> None:
    config = ImageExperimentConfig()
    assert config.data.image_size == 64
