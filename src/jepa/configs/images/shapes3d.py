"""Config for the Shapes3D-backed entity/context world (real rendered images).

Every observation is an exact lookup into DeepMind's 3D Shapes dataset
(https://github.com/deepmind/3d-shapes, ``3dshapes.h5``: 480,000 images, every
combination of 6 factors). Because the source images only exist at fixed grid
points, the temporal process here is a **discrete random walk over factor
indices** -- this needs no snapping/interpolation and indexes directly into
the dataset with zero approximation error.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from jepa.configs.base import OutputConfig, _merge_strict
from jepa.configs.base import _apply_overrides as _apply_overrides_generic
from jepa.configs.timeseries import (
    HierarchicalConfig,
    TemporalEvaluationConfig,
    TemporalModelConfig,
    TemporalTrainingConfig,
)

# Official factor order and cardinality (github.com/deepmind/3d-shapes README).
SHAPES3D_FACTOR_NAMES: tuple[str, ...] = (
    "floor_hue",
    "wall_hue",
    "object_hue",
    "scale",
    "shape",
    "orientation",
)
SHAPES3D_FACTOR_SIZES: dict[str, int] = {
    "floor_hue": 10,
    "wall_hue": 10,
    "object_hue": 10,
    "scale": 8,
    "shape": 4,
    "orientation": 15,
}
# shape is the entity factor; the other five are context.
SHAPES3D_CONTEXT_FACTORS: tuple[str, ...] = tuple(
    name for name in SHAPES3D_FACTOR_NAMES if name != "shape"
)
# Official shape-index -> name mapping per the dataset's documentation.
SHAPES3D_ENTITY_NAMES: tuple[str, ...] = ("cube", "cylinder", "sphere", "capsule")


@dataclass(frozen=True, slots=True)
class Shapes3DDataConfig:
    h5_path: str = "data/3dshapes.h5"
    trajectory_length: int = 16
    entity_switch_probability: float = 0.05
    context_step_probability: float = 0.7
    context_step_max: int = 1
    num_train_trajectories: int = 2000
    num_val_trajectories: int = 500
    num_test_trajectories: int = 500
    num_train_samples: int = 2000
    num_val_samples: int = 500
    num_test_samples: int = 500
    train_sample_seed: int = 1000
    validation_sample_seed: int = 2000
    test_sample_seed: int = 3000
    counterfactual_seed: int = 4000

    @property
    def num_entities(self) -> int:
        return SHAPES3D_FACTOR_SIZES["shape"]

    @property
    def context_dim(self) -> int:
        return len(SHAPES3D_CONTEXT_FACTORS)

    @property
    def observation_dim(self) -> int:
        """Flattened 64x64x3 pixel count -- matches the vector world's duck
        interface so the same pair builders and training helpers run unmodified
        (see jepa.data.timeseries.entity_context / jepa.training.timeseries)."""
        return 3 * 64 * 64


@dataclass(frozen=True, slots=True)
class Shapes3DExperimentConfig:
    data: Shapes3DDataConfig = Shapes3DDataConfig()
    model: TemporalModelConfig = TemporalModelConfig()
    training: TemporalTrainingConfig = TemporalTrainingConfig()
    hierarchical: HierarchicalConfig = HierarchicalConfig()
    evaluation: TemporalEvaluationConfig = TemporalEvaluationConfig()
    output: OutputConfig = OutputConfig()


def _require_positive_int(path: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")


def validate_shapes3d_config(config: Shapes3DExperimentConfig) -> None:
    data = config.data
    _require_positive_int("data.trajectory_length", data.trajectory_length)
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
    if not 0.0 <= data.context_step_probability <= 1.0:
        raise ValueError("data.context_step_probability must be in [0, 1]")
    _require_positive_int("data.context_step_max", data.context_step_max)
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


def _build_config(raw: Mapping[str, Any]) -> Shapes3DExperimentConfig:
    from jepa.configs.timeseries import EMAConfig

    data = Shapes3DDataConfig(**raw["data"])
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
    output = OutputConfig(**raw["output"])
    config = Shapes3DExperimentConfig(
        data=data,
        model=model,
        training=training,
        hierarchical=hierarchical,
        evaluation=evaluation,
        output=output,
    )
    validate_shapes3d_config(config)
    return config


def load_shapes3d_config(
    path: str | Path | None = None, *, overrides: Mapping[str, Any] | None = None
) -> Shapes3DExperimentConfig:
    raw = asdict(Shapes3DExperimentConfig())
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


def shapes3d_config_from_dict(values: Mapping[str, Any]) -> Shapes3DExperimentConfig:
    raw = asdict(Shapes3DExperimentConfig())
    _merge_strict(raw, values)
    return _build_config(raw)


def shapes3d_config_to_dict(config: Shapes3DExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def shapes3d_config_identity_hash(config: Shapes3DExperimentConfig) -> str:
    payload = shapes3d_config_to_dict(config)
    payload.pop("output")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "SHAPES3D_CONTEXT_FACTORS",
    "SHAPES3D_ENTITY_NAMES",
    "SHAPES3D_FACTOR_NAMES",
    "SHAPES3D_FACTOR_SIZES",
    "Shapes3DDataConfig",
    "Shapes3DExperimentConfig",
    "load_shapes3d_config",
    "shapes3d_config_from_dict",
    "shapes3d_config_identity_hash",
    "shapes3d_config_to_dict",
    "validate_shapes3d_config",
]
