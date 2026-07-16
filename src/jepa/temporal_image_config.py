"""Strict typed configuration for the rendered-image entity/context experiment.

A third, parallel configuration tree (see also :mod:`jepa.config` and
:mod:`jepa.temporal_config`). Reuses ``TemporalModelConfig``,
``TemporalTrainingConfig``, ``HierarchicalConfig``, ``TemporalEvaluationConfig``,
and ``OutputConfig`` verbatim -- the image world only needs a new hidden-factor
process (:class:`ImageDataConfig`) and a spatial-masking block (for the static
spatial-JEPA control), so nothing else is duplicated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from jepa.config import OutputConfig, _merge_strict
from jepa.config import _apply_overrides as _apply_overrides_generic
from jepa.temporal_config import (
    HierarchicalConfig,
    TemporalEvaluationConfig,
    TemporalModelConfig,
    TemporalTrainingConfig,
)

DatasetType = Literal["temporal_image", "static_image_spatial", "static_image_spatial_control"]
MaskingStrategy = Literal["block"]
ENTITY_NAMES: tuple[str, ...] = ("none", "circle", "square", "triangle")
CONTEXT_FIELD_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "scale",
    "rotation",
    "red",
    "green",
    "blue",
    "visibility",
    "background",
)


@dataclass(frozen=True, slots=True)
class ImageDataConfig:
    image_size: int = 64
    num_channels: int = 3
    trajectory_length: int = 16
    entity_switch_probability: float = 0.05
    context_smoothness: float = 0.85
    visibility_change_probability: float = 0.1
    background_change_probability: float = 0.05
    num_train_trajectories: int = 2000
    num_val_trajectories: int = 500
    num_test_trajectories: int = 500
    num_train_samples: int = 2000
    num_val_samples: int = 500
    num_test_samples: int = 500
    system_seed: int = 123
    train_sample_seed: int = 1000
    validation_sample_seed: int = 2000
    test_sample_seed: int = 3000
    counterfactual_seed: int = 4000

    @property
    def num_entities(self) -> int:
        return len(ENTITY_NAMES)

    @property
    def context_dim(self) -> int:
        return len(CONTEXT_FIELD_NAMES)

    @property
    def observation_dim(self) -> int:
        """Flattened pixel count -- lets image datasets reuse the vector-world
        pair builders and training helpers, which are written against a
        generic ``dataset.config.observation_dim`` / ``.trajectory_length``
        duck interface rather than :class:`EntityContextDataConfig` itself."""
        return self.num_channels * self.image_size * self.image_size


@dataclass(frozen=True, slots=True)
class SpatialMaskingConfig:
    masking_strategy: MaskingStrategy = "block"
    block_fraction: float = 0.25


@dataclass(frozen=True, slots=True)
class ImageExperimentConfig:
    dataset_type: DatasetType = "temporal_image"
    data: ImageDataConfig = ImageDataConfig()
    model: TemporalModelConfig = TemporalModelConfig()
    training: TemporalTrainingConfig = TemporalTrainingConfig()
    hierarchical: HierarchicalConfig = HierarchicalConfig()
    evaluation: TemporalEvaluationConfig = TemporalEvaluationConfig()
    spatial: SpatialMaskingConfig = SpatialMaskingConfig()
    output: OutputConfig = OutputConfig()


def _require_positive_int(path: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")


def validate_image_config(config: ImageExperimentConfig) -> None:
    data = config.data
    if config.dataset_type not in (
        "temporal_image",
        "static_image_spatial",
        "static_image_spatial_control",
    ):
        raise ValueError(f"dataset_type has invalid value {config.dataset_type!r}")
    for name in ("image_size", "num_channels", "trajectory_length"):
        _require_positive_int(f"data.{name}", getattr(data, name))
    for name in (
        "num_train_trajectories",
        "num_val_trajectories",
        "num_test_trajectories",
        "num_train_samples",
        "num_val_samples",
        "num_test_samples",
    ):
        _require_positive_int(f"data.{name}", getattr(data, name))
    if not 0.0 < data.entity_switch_probability < 1.0:
        raise ValueError("data.entity_switch_probability must satisfy 0 < value < 1")
    if not 0.0 < data.context_smoothness < 1.0:
        raise ValueError("data.context_smoothness must satisfy 0 < value < 1")
    if not 0.0 <= data.visibility_change_probability <= 1.0:
        raise ValueError("data.visibility_change_probability must be in [0, 1]")
    if not 0.0 <= data.background_change_probability <= 1.0:
        raise ValueError("data.background_change_probability must be in [0, 1]")
    if config.spatial.masking_strategy != "block":
        raise ValueError(
            f"spatial.masking_strategy has invalid value {config.spatial.masking_strategy!r}"
        )
    if not 0.0 < config.spatial.block_fraction < 1.0:
        raise ValueError("spatial.block_fraction must satisfy 0 < value < 1")
    if config.model.kind == "hierarchical":
        if (
            config.model.entity_latent_dim + config.model.context_latent_dim
            != config.model.latent_dim
        ):
            raise ValueError(
                "model.entity_latent_dim + model.context_latent_dim must equal model.latent_dim"
            )
    for dim in config.evaluation.entity_subspace_dims:
        if dim > config.model.latent_dim:
            raise ValueError(
                "evaluation.entity_subspace_dims values must not exceed model.latent_dim"
            )


def _build_config(raw: Mapping[str, Any]) -> ImageExperimentConfig:
    from jepa.temporal_config import EMAConfig

    data = ImageDataConfig(**raw["data"])
    model = TemporalModelConfig(**raw["model"])
    ema = EMAConfig(**raw["training"]["ema"])
    training_values = dict(raw["training"])
    training_values["ema"] = ema
    training_values["betas"] = tuple(training_values["betas"])
    training = TemporalTrainingConfig(**training_values)
    hierarchical = HierarchicalConfig(**raw["hierarchical"])
    evaluation_values = dict(raw["evaluation"])
    evaluation_values["entity_subspace_dims"] = tuple(evaluation_values["entity_subspace_dims"])
    evaluation = TemporalEvaluationConfig(**evaluation_values)
    spatial = SpatialMaskingConfig(**raw["spatial"])
    output = OutputConfig(**raw["output"])
    config = ImageExperimentConfig(
        dataset_type=raw["dataset_type"],
        data=data,
        model=model,
        training=training,
        hierarchical=hierarchical,
        evaluation=evaluation,
        spatial=spatial,
        output=output,
    )
    validate_image_config(config)
    return config


def load_image_config(
    path: str | Path | None = None, *, overrides: Mapping[str, Any] | None = None
) -> ImageExperimentConfig:
    raw = asdict(ImageExperimentConfig())
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text())
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise ValueError("configuration root must be a mapping")
        _merge_strict(raw, loaded)
    if overrides:
        _apply_overrides_generic(raw, overrides)
    return _build_config(raw)


def image_config_from_dict(values: Mapping[str, Any]) -> ImageExperimentConfig:
    raw = asdict(ImageExperimentConfig())
    _merge_strict(raw, values)
    return _build_config(raw)


def image_config_to_dict(config: ImageExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def image_config_identity_hash(config: ImageExperimentConfig) -> str:
    payload = image_config_to_dict(config)
    payload.pop("output")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "CONTEXT_FIELD_NAMES",
    "ENTITY_NAMES",
    "ImageDataConfig",
    "ImageExperimentConfig",
    "SpatialMaskingConfig",
    "image_config_from_dict",
    "image_config_identity_hash",
    "image_config_to_dict",
    "load_image_config",
    "validate_image_config",
]
