"""Strict typed configuration and deterministic seed ownership."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from jepa.models.encoders import Architecture

SystemKind = Literal["linear", "nonlinear"]
DeviceName = Literal["cpu", "cuda", "mps", "auto"]


@dataclass(frozen=True, slots=True)
class DataConfig:
    system_kind: SystemKind = "linear"
    num_samples: int = 4096
    validation_samples: int = 1024
    test_samples: int = 1024
    sequence_length: int = 64
    window_size: int = 16
    burn_in_steps: int = 64
    latent_state_dim: int = 4
    observation_dim: int = 4
    process_noise: float = 0.01
    observation_noise: float = 0.01
    transition_norm: float = 0.9
    system_seed: int = 123
    train_sample_seed: int = 1000
    validation_sample_seed: int = 2000
    test_sample_seed: int = 3000


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: Architecture = "linear"
    latent_dim: int = 16
    hidden_dim: int = 64
    hidden_layers: int = 1


@dataclass(frozen=True, slots=True)
class EMAConfig:
    enabled: bool = True
    decay: float = 0.99


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 0
    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1.0e-8
    amsgrad: bool = False
    batch_size: int = 128
    epochs: int = 100
    shuffle: bool = True
    device: DeviceName = "auto"
    evaluation_every_epochs: int = 1
    checkpoint_every_epochs: int = 10
    stop_gradient: bool = True
    ema: EMAConfig = EMAConfig()


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    covariance_epsilon: float = 1.0e-8
    collapse_std_threshold: float = 1.0e-3
    collapse_rank_threshold: float = 0.1
    probe_ridge: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: str = "outputs"
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    output: OutputConfig = OutputConfig()


@dataclass(frozen=True, slots=True)
class PairedReplicateSeeds:
    replicate_seed: int
    system_seed: int
    train_sample_seed: int
    validation_sample_seed: int
    test_sample_seed: int
    training_seed: int

    def model_seed(self, architecture: Architecture) -> int:
        return derive_seed(self.training_seed, architecture, "model")

    def epoch_order_seed(self, epoch: int) -> int:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        return derive_seed(self.training_seed, "train-order", epoch)


_ALIASES = {
    "architecture": "model.architecture",
    "ema": "training.ema.enabled",
    "sg": "training.stop_gradient",
    "system_kind": "data.system_kind",
}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_bool(path: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")


def _require_int(path: str, value: object, *, positive: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{path} must be positive")


def _require_number(path: str, value: object) -> float:
    if not _is_number(value):
        raise ValueError(f"{path} must be numeric")
    numeric = float(cast(int | float, value))
    if not math.isfinite(numeric):
        raise ValueError(f"{path} must be finite")
    return numeric


def validate_data_config(data: DataConfig) -> None:
    if data.system_kind not in {"linear", "nonlinear"}:
        raise ValueError(f"data.system_kind has invalid value {data.system_kind!r}")
    for name in (
        "num_samples",
        "validation_samples",
        "test_samples",
        "sequence_length",
        "window_size",
        "latent_state_dim",
        "observation_dim",
    ):
        _require_int(f"data.{name}", getattr(data, name), positive=True)
    _require_int("data.burn_in_steps", data.burn_in_steps)
    if data.burn_in_steps < 0:
        raise ValueError("data.burn_in_steps must be non-negative")
    if data.sequence_length < 2 * data.window_size:
        raise ValueError("data.sequence_length must be at least 2 * data.window_size")
    if data.observation_dim < data.latent_state_dim:
        raise ValueError("data.observation_dim must be >= data.latent_state_dim")
    for name in ("process_noise", "observation_noise"):
        value = _require_number(f"data.{name}", getattr(data, name))
        if value < 0:
            raise ValueError(f"data.{name} must be non-negative")
    transition_norm = _require_number("data.transition_norm", data.transition_norm)
    if not 0.0 < transition_norm < 1.0:
        raise ValueError("data.transition_norm must satisfy 0 < value < 1")
    for name in (
        "system_seed",
        "train_sample_seed",
        "validation_sample_seed",
        "test_sample_seed",
    ):
        _require_int(f"data.{name}", getattr(data, name))


def validate_config(config: ExperimentConfig) -> None:
    data = config.data
    model = config.model
    training = config.training
    evaluation = config.evaluation

    validate_data_config(data)

    if model.architecture not in {"linear", "nonlinear"}:
        raise ValueError(f"model.architecture has invalid value {model.architecture!r}")
    for name in ("latent_dim", "hidden_dim", "hidden_layers"):
        _require_int(f"model.{name}", getattr(model, name), positive=True)

    _require_int("training.seed", training.seed)
    if _require_number("training.learning_rate", training.learning_rate) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if len(training.betas) != 2:
        raise ValueError("training.betas must contain exactly two values")
    for index, beta in enumerate(training.betas):
        numeric_beta = _require_number(f"training.betas[{index}]", beta)
        if not 0.0 <= numeric_beta < 1.0:
            raise ValueError(f"training.betas[{index}] must satisfy 0 <= value < 1")
    if _require_number("training.epsilon", training.epsilon) <= 0:
        raise ValueError("training.epsilon must be positive")
    _require_bool("training.amsgrad", training.amsgrad)
    _require_bool("training.shuffle", training.shuffle)
    _require_bool("training.stop_gradient", training.stop_gradient)
    for name in (
        "batch_size",
        "epochs",
        "evaluation_every_epochs",
        "checkpoint_every_epochs",
    ):
        _require_int(f"training.{name}", getattr(training, name), positive=True)
    if training.device not in {"cpu", "cuda", "mps", "auto"}:
        raise ValueError(f"training.device has invalid value {training.device!r}")
    _require_bool("training.ema.enabled", training.ema.enabled)
    ema_decay = _require_number("training.ema.decay", training.ema.decay)
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("training.ema.decay must satisfy 0 <= value < 1")

    for name in (
        "covariance_epsilon",
        "collapse_std_threshold",
        "collapse_rank_threshold",
        "probe_ridge",
    ):
        value = _require_number(f"evaluation.{name}", getattr(evaluation, name))
        if value < 0 or (name in {"covariance_epsilon", "probe_ridge"} and value == 0):
            raise ValueError(f"evaluation.{name} has an invalid value")
    if evaluation.collapse_rank_threshold > 1:
        raise ValueError("evaluation.collapse_rank_threshold must be <= 1")
    if not isinstance(config.output.root, str) or not config.output.root.strip():
        raise ValueError("output.root must be a non-empty string")
    _require_bool("output.overwrite", config.output.overwrite)


def _merge_strict(destination: dict[str, Any], incoming: Mapping[str, Any], path: str = "") -> None:
    for key, value in incoming.items():
        dotted = f"{path}.{key}" if path else key
        if key not in destination:
            raise ValueError(f"unknown configuration key: {dotted}")
        if isinstance(destination[key], dict):
            if not isinstance(value, Mapping):
                raise ValueError(f"{dotted} must be a mapping")
            _merge_strict(destination[key], value, dotted)
        else:
            destination[key] = value


def _parse_override_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if lowered == "on":
        return True
    if lowered == "off":
        return False
    return yaml.safe_load(value)


def _apply_overrides(raw: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    canonical: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for source_key, source_value in overrides.items():
        target_key = _ALIASES.get(source_key, source_key)
        if target_key in canonical:
            raise ValueError(
                f"conflicting overrides {sources[target_key]!r} and {source_key!r} for {target_key}"
            )
        canonical[target_key] = _parse_override_value(source_value)
        sources[target_key] = source_key

    for dotted, value in canonical.items():
        parts = dotted.split(".")
        cursor: dict[str, Any] = raw
        for part in parts[:-1]:
            if part not in cursor:
                raise ValueError(f"unknown configuration key: {dotted}")
            child = cursor[part]
            if not isinstance(child, dict):
                raise ValueError(f"configuration key is not a section: {part}")
            cursor = child
        leaf = parts[-1]
        if leaf not in cursor:
            raise ValueError(f"unknown configuration key: {dotted}")
        cursor[leaf] = value


def _build_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw["model"])
    ema = EMAConfig(**raw["training"]["ema"])
    training_values = dict(raw["training"])
    training_values["ema"] = ema
    training_values["betas"] = tuple(training_values["betas"])
    training = TrainingConfig(**training_values)
    evaluation = EvaluationConfig(**raw["evaluation"])
    output = OutputConfig(**raw["output"])
    config = ExperimentConfig(data, model, training, evaluation, output)
    validate_config(config)
    return config


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Load typed defaults, then strict YAML, then optional CLI-style overrides."""
    raw = asdict(ExperimentConfig())
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text())
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise ValueError("configuration root must be a mapping")
        _merge_strict(raw, loaded)
    if overrides:
        _apply_overrides(raw, overrides)
    return _build_config(raw)


def config_from_dict(values: Mapping[str, Any]) -> ExperimentConfig:
    """Rebuild and validate a config stored in an artifact or checkpoint."""
    raw = asdict(ExperimentConfig())
    _merge_strict(raw, values)
    return _build_config(raw)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def config_identity_hash(config: ExperimentConfig) -> str:
    """Hash semantic configuration while excluding output location and overwrite policy."""
    payload = config_to_dict(config)
    payload.pop("output")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def derive_seed(root_seed: int, *parts: str | int) -> int:
    _require_int("root_seed", root_seed)
    payload = json.dumps([root_seed, *parts], separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def derive_paired_replicate(replicate_seed: int) -> PairedReplicateSeeds:
    _require_int("replicate_seed", replicate_seed)
    return PairedReplicateSeeds(
        replicate_seed=replicate_seed,
        system_seed=derive_seed(replicate_seed, "system"),
        train_sample_seed=derive_seed(replicate_seed, "samples", "train"),
        validation_sample_seed=derive_seed(replicate_seed, "samples", "validation"),
        test_sample_seed=derive_seed(replicate_seed, "samples", "test"),
        training_seed=derive_seed(replicate_seed, "training"),
    )


def apply_paired_replicate(
    config: ExperimentConfig, replicate_seed: int
) -> tuple[ExperimentConfig, PairedReplicateSeeds]:
    """Replace every stochastic owner without touching model or policy axes."""
    seeds = derive_paired_replicate(replicate_seed)
    data = replace(
        config.data,
        system_seed=seeds.system_seed,
        train_sample_seed=seeds.train_sample_seed,
        validation_sample_seed=seeds.validation_sample_seed,
        test_sample_seed=seeds.test_sample_seed,
    )
    training = replace(config.training, seed=seeds.training_seed)
    paired = replace(config, data=data, training=training)
    validate_config(paired)
    return paired, seeds
