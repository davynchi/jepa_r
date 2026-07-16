"""Strict typed configuration for the entity/context temporal-persistence experiment.

This is a parallel configuration tree to :mod:`jepa.config`. It reuses the same
strict-merge / override / seed-derivation machinery but never touches
``ExperimentConfig``, so every existing config, run, and test keeps working
unmodified.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from jepa.config import EMAConfig, OutputConfig, _merge_strict, derive_seed
from jepa.config import _apply_overrides as _apply_overrides_generic
from jepa.models import Architecture

ObservationMode = Literal["linear", "nonlinear"]
TargetPairing = Literal["temporal", "shuffled", "shuffled_same_entity"]
ModelKind = Literal["standard", "hierarchical", "random"]


@dataclass(frozen=True, slots=True)
class EntityContextDataConfig:
    num_entities: int = 3
    context_dim: int = 6
    trajectory_length: int = 16
    observation_dim: int = 32
    entity_switch_probability: float = 0.05
    context_rho: float = 0.5
    observation_noise_std: float = 0.01
    observation_mode: ObservationMode = "linear"
    nonlinear_hidden_dim: int = 64
    num_train_trajectories: int = 2000
    num_val_trajectories: int = 500
    num_test_trajectories: int = 500
    system_seed: int = 123
    train_sample_seed: int = 1000
    validation_sample_seed: int = 2000
    test_sample_seed: int = 3000
    counterfactual_seed: int = 4000


@dataclass(frozen=True, slots=True)
class TemporalModelConfig:
    kind: ModelKind = "standard"
    architecture: Architecture = "linear"
    latent_dim: int = 32
    hidden_dim: int = 64
    hidden_layers: int = 1
    entity_latent_dim: int = 8
    context_latent_dim: int = 24


@dataclass(frozen=True, slots=True)
class TemporalTrainingConfig:
    seed: int = 0
    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1.0e-8
    batch_size: int = 128
    epochs: int = 30
    device: str = "auto"
    stop_gradient: bool = True
    ema: EMAConfig = EMAConfig()
    prediction_horizon: int = 1
    target_pairing: TargetPairing = "temporal"
    evaluation_every_epochs: int = 1
    checkpoint_every_epochs: int = 10


@dataclass(frozen=True, slots=True)
class HierarchicalConfig:
    short_horizon: int = 1
    long_horizon: int = 8
    lambda_entity: float = 1.0
    lambda_context: float = 1.0
    lambda_var: float = 1.0
    lambda_cross: float = 1.0
    variance_gamma: float = 1.0
    use_horizon_split: bool = True
    use_variance_reg: bool = True
    use_cross_cov_reg: bool = True
    epsilon: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class TemporalEvaluationConfig:
    entity_subspace_dims: tuple[int, ...] = (1, 2, 4, 8)
    covariance_epsilon: float = 1.0e-6
    probe_ridge: float = 1.0e-6
    autocorrelation_max_lag: int = 8
    counterfactual_pairs: int = 500
    selectivity_alpha: float = 1.0


@dataclass(frozen=True, slots=True)
class TemporalExperimentConfig:
    data: EntityContextDataConfig = EntityContextDataConfig()
    model: TemporalModelConfig = TemporalModelConfig()
    training: TemporalTrainingConfig = TemporalTrainingConfig()
    hierarchical: HierarchicalConfig = HierarchicalConfig()
    evaluation: TemporalEvaluationConfig = TemporalEvaluationConfig()
    output: OutputConfig = OutputConfig()


def _require_positive_int(path: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")


def _require_probability(path: str, value: object) -> float:
    numeric = float(cast(float, value))
    if not 0.0 < numeric < 1.0:
        raise ValueError(f"{path} must satisfy 0 < value < 1")
    return numeric


def validate_temporal_config(config: TemporalExperimentConfig) -> None:
    data = config.data
    model = config.model
    training = config.training
    hierarchical = config.hierarchical
    evaluation = config.evaluation

    for name in (
        "num_entities",
        "context_dim",
        "trajectory_length",
        "observation_dim",
        "num_train_trajectories",
        "num_val_trajectories",
        "num_test_trajectories",
    ):
        _require_positive_int(f"data.{name}", getattr(data, name))
    if data.num_entities < 2:
        raise ValueError("data.num_entities must be at least 2")
    _require_probability("data.entity_switch_probability", data.entity_switch_probability)
    if not -1.0 < data.context_rho < 1.0:
        raise ValueError("data.context_rho must satisfy -1 < value < 1")
    if data.observation_noise_std < 0:
        raise ValueError("data.observation_noise_std must be non-negative")
    if data.observation_mode not in {"linear", "nonlinear"}:
        raise ValueError(f"data.observation_mode has invalid value {data.observation_mode!r}")
    if data.trajectory_length < 4:
        raise ValueError("data.trajectory_length must be at least 4")

    if model.kind not in {"standard", "hierarchical", "random"}:
        raise ValueError(f"model.kind has invalid value {model.kind!r}")
    if model.architecture not in {"linear", "nonlinear"}:
        raise ValueError(f"model.architecture has invalid value {model.architecture!r}")
    for name in ("latent_dim", "hidden_dim", "hidden_layers"):
        _require_positive_int(f"model.{name}", getattr(model, name))
    if model.kind == "hierarchical":
        for name in ("entity_latent_dim", "context_latent_dim"):
            _require_positive_int(f"model.{name}", getattr(model, name))
        if model.entity_latent_dim + model.context_latent_dim != model.latent_dim:
            raise ValueError(
                "model.entity_latent_dim + model.context_latent_dim must equal model.latent_dim"
            )

    _require_positive_int("training.batch_size", training.batch_size)
    _require_positive_int("training.epochs", training.epochs)
    _require_positive_int("training.prediction_horizon", training.prediction_horizon)
    if training.prediction_horizon >= data.trajectory_length:
        raise ValueError("training.prediction_horizon must be less than data.trajectory_length")
    if training.target_pairing not in {"temporal", "shuffled", "shuffled_same_entity"}:
        raise ValueError(f"training.target_pairing has invalid value {training.target_pairing!r}")
    if training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")

    if hierarchical.long_horizon >= data.trajectory_length:
        raise ValueError("hierarchical.long_horizon must be less than data.trajectory_length")
    if hierarchical.short_horizon < 1 or hierarchical.long_horizon < 1:
        raise ValueError("hierarchical horizons must be at least 1")
    for name in ("lambda_entity", "lambda_context", "lambda_var", "lambda_cross"):
        if getattr(hierarchical, name) < 0:
            raise ValueError(f"hierarchical.{name} must be non-negative")
    if hierarchical.epsilon <= 0:
        raise ValueError("hierarchical.epsilon must be positive")

    if not evaluation.entity_subspace_dims:
        raise ValueError("evaluation.entity_subspace_dims must be non-empty")
    for dim in evaluation.entity_subspace_dims:
        _require_positive_int("evaluation.entity_subspace_dims[*]", dim)
        if dim > model.latent_dim:
            raise ValueError(
                "evaluation.entity_subspace_dims values must not exceed model.latent_dim"
            )
    if evaluation.probe_ridge <= 0:
        raise ValueError("evaluation.probe_ridge must be positive")
    if evaluation.counterfactual_pairs <= 0:
        raise ValueError("evaluation.counterfactual_pairs must be positive")


def _build_config(raw: Mapping[str, Any]) -> TemporalExperimentConfig:
    data = EntityContextDataConfig(**raw["data"])
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
    config = TemporalExperimentConfig(data, model, training, hierarchical, evaluation, output)
    validate_temporal_config(config)
    return config


def load_temporal_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> TemporalExperimentConfig:
    raw = asdict(TemporalExperimentConfig())
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


def temporal_config_from_dict(values: Mapping[str, Any]) -> TemporalExperimentConfig:
    raw = asdict(TemporalExperimentConfig())
    _merge_strict(raw, values)
    return _build_config(raw)


def temporal_config_to_dict(config: TemporalExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def temporal_config_identity_hash(config: TemporalExperimentConfig) -> str:
    import hashlib

    payload = temporal_config_to_dict(config)
    payload.pop("output")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "EntityContextDataConfig",
    "HierarchicalConfig",
    "TemporalEvaluationConfig",
    "TemporalExperimentConfig",
    "TemporalModelConfig",
    "TemporalTrainingConfig",
    "derive_seed",
    "load_temporal_config",
    "temporal_config_from_dict",
    "temporal_config_identity_hash",
    "temporal_config_to_dict",
    "validate_temporal_config",
]
