"""Executable SG/EMA semantics for the JEPA research core."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from jepa.analysis.metrics import (
    MetricValue,
    RepresentationMetrics,
    WeightedMean,
    compute_representation_metrics,
    fit_ridge_probe,
    global_gradient_norm,
    ridge_r2_score,
)
from jepa.configs.base import (
    EvaluationConfig,
    ExperimentConfig,
    config_from_dict,
    config_identity_hash,
    config_to_dict,
    derive_seed,
    validate_config,
)
from jepa.data.timeseries.dynamics import DatasetSplits, LatentDynamicsDataset, build_dataset_splits
from jepa.models.encoders import Architecture, build_model_pair

SCHEMA_VERSION = 1

_REPRESENTATION_FIELDS = (
    "latent_std_mean",
    "effective_rank",
    "effective_rank_normalized",
    "top_eigen_fraction",
    "norm_mean",
)
_GRADIENT_FIELDS = (
    "context_encoder_grad_norm",
    "predictor_grad_norm",
    "target_encoder_grad_norm",
    "context_latent_grad_norm",
    "target_latent_grad_norm",
)
HISTORY_COLUMNS = (
    "run_id",
    "epoch",
    "split",
    "mse",
    *(
        field
        for branch in ("context", "target")
        for metric in _REPRESENTATION_FIELDS
        for field in (f"{branch}_{metric}", f"{branch}_{metric}_reason")
    ),
    "prediction_norm_mean",
    "prediction_norm_mean_reason",
    *(field for metric in _GRADIENT_FIELDS for field in (metric, f"{metric}_reason")),
    "context_target_parameter_distance",
    "context_target_parameter_distance_reason",
    "elapsed_seconds",
)


@dataclass(frozen=True, slots=True)
class OptimizationPolicy:
    """The complete optimization semantics of one SG/EMA cell."""

    stop_gradient: bool
    ema_enabled: bool
    separate_target: bool
    optimize_target: bool
    variant: str


@dataclass(slots=True)
class JEPACore:
    """The three JEPA modules plus their explicit optimization policy."""

    context_encoder: nn.Module
    predictor: nn.Module
    target_encoder: nn.Module
    policy: OptimizationPolicy


@dataclass(slots=True)
class ForwardPass:
    context_latent: torch.Tensor
    target_latent: torch.Tensor
    prediction: torch.Tensor
    loss: torch.Tensor


def resolve_policy(*, stop_gradient: bool, ema_enabled: bool) -> OptimizationPolicy:
    """Resolve the four-cell truth table without implicit model-side branching.

    SG off, EMA off ─┐ shared target, target branch contributes gradients
    SG on,  EMA off ─┘ shared target, target latent is detached
    SG on,  EMA on  ── separate frozen target, updated only by EMA
    SG off, EMA on  ── separate trainable target, Adam then EMA (hybrid)
    """
    if not ema_enabled:
        return OptimizationPolicy(
            stop_gradient=stop_gradient,
            ema_enabled=False,
            separate_target=False,
            optimize_target=False,
            variant=("shared-stop-gradient-target" if stop_gradient else "shared-gradient-target"),
        )
    if stop_gradient:
        return OptimizationPolicy(
            stop_gradient=True,
            ema_enabled=True,
            separate_target=True,
            optimize_target=False,
            variant="ema-stop-gradient-target",
        )
    return OptimizationPolicy(
        stop_gradient=False,
        ema_enabled=True,
        separate_target=True,
        optimize_target=True,
        variant="hybrid-gradient-ema-target",
    )


def build_jepa_core(
    architecture: Architecture,
    *,
    input_dim: int,
    latent_dim: int,
    stop_gradient: bool,
    ema_enabled: bool,
    hidden_dim: int = 64,
    hidden_layers: int = 1,
) -> JEPACore:
    """Build modules with identity/separation dictated only by the policy."""
    context_encoder, predictor = build_model_pair(
        architecture,
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    )
    policy = resolve_policy(stop_gradient=stop_gradient, ema_enabled=ema_enabled)
    target_encoder = deepcopy(context_encoder) if policy.separate_target else context_encoder
    if policy.separate_target and not policy.optimize_target:
        target_encoder.requires_grad_(False)
    return JEPACore(context_encoder, predictor, target_encoder, policy)


def optimizer_parameters(core: JEPACore) -> tuple[nn.Parameter, ...]:
    """Return the exact, de-duplicated parameter set for Adam."""
    groups = [core.context_encoder.parameters(), core.predictor.parameters()]
    if core.policy.optimize_target:
        groups.append(core.target_encoder.parameters())

    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for group in groups:
        for parameter in group:
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return tuple(parameters)


def compute_loss(
    core: JEPACore,
    context: torch.Tensor,
    target: torch.Tensor,
) -> ForwardPass:
    """Compute the sole v1 objective while preserving raw branch tensors for diagnostics."""
    context_latent = core.context_encoder(context)
    target_latent = core.target_encoder(target)
    prediction = core.predictor(context_latent)
    objective_target = target_latent.detach() if core.policy.stop_gradient else target_latent
    loss = F.mse_loss(prediction, objective_target)
    return ForwardPass(context_latent, target_latent, prediction, loss)


@torch.no_grad()
def ema_update(target_encoder: nn.Module, context_encoder: nn.Module, decay: float) -> None:
    """Apply the exact post-optimizer target pull."""
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"EMA decay must satisfy 0 <= decay < 1, got {decay}")

    target_parameters = tuple(target_encoder.parameters())
    context_parameters = tuple(context_encoder.parameters())
    if len(target_parameters) != len(context_parameters):
        raise ValueError("target and context encoders have different parameter counts")

    for target_parameter, context_parameter in zip(
        target_parameters, context_parameters, strict=True
    ):
        if target_parameter.shape != context_parameter.shape:
            raise ValueError("target and context encoder parameter shapes differ")
        target_parameter.mul_(decay).add_(context_parameter, alpha=1.0 - decay)


def train_step(
    core: JEPACore,
    optimizer: torch.optim.Optimizer,
    context: torch.Tensor,
    target: torch.Tensor,
    *,
    ema_decay: float = 0.99,
) -> ForwardPass:
    """Run backward → optimizer → optional EMA in the agreed order."""
    if core.policy.ema_enabled and not 0.0 <= ema_decay < 1.0:
        raise ValueError(f"EMA decay must satisfy 0 <= decay < 1, got {ema_decay}")
    optimizer.zero_grad(set_to_none=True)
    forward = compute_loss(core, context, target)
    forward.loss.backward()
    optimizer.step()
    if core.policy.ema_enabled:
        ema_update(core.target_encoder, core.context_encoder, ema_decay)
    return forward


@dataclass(frozen=True, slots=True)
class SplitOutputs:
    """Materialized evaluation outputs used by metrics and linear probes."""

    mse: float
    context_latents: torch.Tensor
    target_latents: torch.Tensor
    predictions: torch.Tensor
    context_states: torch.Tensor
    target_states: torch.Tensor
    target_windows: torch.Tensor


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Locations and final values returned by a completed experiment."""

    run_id: str
    run_dir: Path
    history: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]


@dataclass(slots=True)
class _OptionalWeightedMean:
    mean: WeightedMean
    missing_reason: str = "missing_gradient"

    def update(self, metric: MetricValue, weight: int) -> None:
        if metric.value is None:
            self.missing_reason = metric.reason or self.missing_reason
        else:
            self.mean.update(metric.value, weight)

    def compute(self) -> MetricValue:
        result = self.mean.compute()
        if result.value is None:
            return MetricValue(None, self.missing_reason)
        return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def build_run_id(config: ExperimentConfig, seed_label: int | None = None) -> str:
    """Build a readable, collision-resistant identity for a fresh run."""
    label = config.training.seed if seed_label is None else seed_label
    sg = "on" if config.training.stop_gradient else "off"
    ema = "on" if config.training.ema.enabled else "off"
    return (
        f"dynamics-{config.data.system_kind}_model-{config.model.architecture}"
        f"_sg-{sg}_ema-{ema}_seed-{label}_cfg-{config_identity_hash(config)[:8]}"
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path, serialized.encode("utf-8"))


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = yaml.safe_dump(dict(payload), sort_keys=False)
    _atomic_bytes(path, serialized.encode("utf-8"))


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column) for column in HISTORY_COLUMNS})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    numeric = set(HISTORY_COLUMNS) - {
        "run_id",
        "split",
        *(column for column in HISTORY_COLUMNS if column.endswith("_reason")),
    }
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HISTORY_COLUMNS:
            raise ValueError("history.csv has an incompatible schema")
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key == "epoch":
                    row[key] = int(value)
                elif key in numeric:
                    row[key] = None if value == "" else float(value)
                else:
                    row[key] = None if value == "" else value
            identity = (row["epoch"], row["split"])
            if identity in seen:
                raise ValueError(f"history.csv contains duplicate row {identity}")
            seen.add(identity)
            rows.append(row)
    return rows


def _append_history(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    identity = (row["epoch"], row["split"])
    if any((existing["epoch"], existing["split"]) == identity for existing in rows):
        return
    rows.append(row)


def _metric_columns(prefix: str, metrics: RepresentationMetrics) -> dict[str, Any]:
    values = {
        "latent_std_mean": metrics.latent_std_mean,
        "effective_rank": metrics.effective_rank,
        "effective_rank_normalized": metrics.effective_rank_normalized,
        "top_eigen_fraction": metrics.top_eigen_fraction,
        "norm_mean": metrics.vector_norm_mean,
    }
    result: dict[str, Any] = {}
    for name, metric in values.items():
        result[f"{prefix}_{name}"] = metric.value
        result[f"{prefix}_{name}_reason"] = metric.reason
    return result


def _parameter_distance(core: JEPACore) -> MetricValue:
    if not core.policy.separate_target:
        return MetricValue(0.0)
    total = torch.zeros((), dtype=torch.float64)
    for context, target in zip(
        core.context_encoder.parameters(), core.target_encoder.parameters(), strict=True
    ):
        difference = context.detach().cpu().double() - target.detach().cpu().double()
        total += difference.square().sum()
    return MetricValue(total.sqrt().item())


def _batch_indices(size: int, batch_size: int, *, seed: int, shuffle: bool) -> list[torch.Tensor]:
    if shuffle:
        order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    else:
        order = torch.arange(size)
    return list(order.split(batch_size))


def _train_epoch(
    core: JEPACore,
    optimizer: torch.optim.Optimizer,
    dataset: LatentDynamicsDataset,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, MetricValue]:
    core.context_encoder.train()
    core.predictor.train()
    core.target_encoder.train(core.policy.optimize_target)
    aggregates = {name: _OptionalWeightedMean(WeightedMean()) for name in _GRADIENT_FIELDS}
    loss_mean = WeightedMean()
    for indices in _batch_indices(
        len(dataset),
        config.training.batch_size,
        seed=derive_seed(config.training.seed, "train-order", epoch),
        shuffle=config.training.shuffle,
    ):
        context = dataset.contexts[indices].flatten(1).to(device)
        target = dataset.targets[indices].flatten(1).to(device)
        optimizer.zero_grad(set_to_none=True)
        forward = compute_loss(core, context, target)
        if not torch.isfinite(forward.loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        if forward.context_latent.requires_grad:
            forward.context_latent.retain_grad()
        if forward.target_latent.requires_grad:
            forward.target_latent.retain_grad()
        forward.loss.backward()
        weight = len(indices)
        loss_mean.update(forward.loss.item(), weight)
        aggregates["context_encoder_grad_norm"].update(
            global_gradient_norm(core.context_encoder.parameters()), weight
        )
        aggregates["predictor_grad_norm"].update(
            global_gradient_norm(core.predictor.parameters()), weight
        )
        if core.policy.separate_target:
            reason = "stop_gradient" if core.policy.stop_gradient else "missing_gradient"
            target_parameter_gradient = global_gradient_norm(
                core.target_encoder.parameters(), missing_reason=reason
            )
        else:
            target_parameter_gradient = MetricValue(None, "shared_target_module")
        aggregates["target_encoder_grad_norm"].update(target_parameter_gradient, weight)
        context_gradient = forward.context_latent.grad
        target_gradient = forward.target_latent.grad
        aggregates["context_latent_grad_norm"].update(
            MetricValue(
                torch.linalg.vector_norm(context_gradient).item()
                if context_gradient is not None
                else None,
                None if context_gradient is not None else "missing_gradient",
            ),
            weight,
        )
        aggregates["target_latent_grad_norm"].update(
            MetricValue(
                torch.linalg.vector_norm(target_gradient).item()
                if target_gradient is not None
                else None,
                None
                if target_gradient is not None
                else ("stop_gradient" if core.policy.stop_gradient else "missing_gradient"),
            ),
            weight,
        )
        optimizer.step()
        if core.policy.ema_enabled:
            ema_update(
                core.target_encoder,
                core.context_encoder,
                config.training.ema.decay,
            )
    return {"mse": loss_mean.compute()} | {
        name: aggregate.compute() for name, aggregate in aggregates.items()
    }


@torch.no_grad()
def _collect_split_outputs(
    core: JEPACore,
    dataset: LatentDynamicsDataset,
    batch_size: int,
    device: torch.device,
) -> SplitOutputs:
    core.context_encoder.eval()
    core.predictor.eval()
    core.target_encoder.eval()
    loss_mean = WeightedMean()
    context_latents: list[torch.Tensor] = []
    target_latents: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for indices in _batch_indices(len(dataset), batch_size, seed=0, shuffle=False):
        context = dataset.contexts[indices].flatten(1).to(device)
        target = dataset.targets[indices].flatten(1).to(device)
        forward = compute_loss(core, context, target)
        if not torch.isfinite(forward.loss):
            raise FloatingPointError(f"non-finite evaluation loss on {dataset.split}")
        loss_mean.update(forward.loss.item(), len(indices))
        context_latents.append(forward.context_latent.cpu())
        target_latents.append(forward.target_latent.cpu())
        predictions.append(forward.prediction.cpu())
    mse = loss_mean.compute()
    if mse.value is None:
        raise RuntimeError(f"empty evaluation split: {dataset.split}")
    return SplitOutputs(
        mse=mse.value,
        context_latents=torch.cat(context_latents),
        target_latents=torch.cat(target_latents),
        predictions=torch.cat(predictions),
        context_states=dataset.context_states.detach().cpu(),
        target_states=dataset.target_states.detach().cpu(),
        target_windows=dataset.targets.flatten(1).detach().cpu(),
    )


def _history_row(
    *,
    run_id: str,
    epoch: int,
    split: str,
    outputs: SplitOutputs,
    evaluation_config: EvaluationConfig,
    gradients: Mapping[str, MetricValue] | None,
    distance: MetricValue,
    elapsed_seconds: float,
) -> dict[str, Any]:
    context = compute_representation_metrics(outputs.context_latents, evaluation_config)
    target = compute_representation_metrics(outputs.target_latents, evaluation_config)
    prediction_norm = MetricValue(
        torch.linalg.vector_norm(outputs.predictions.double(), dim=1).mean().item()
    )
    row: dict[str, Any] = {
        "run_id": run_id,
        "epoch": epoch,
        "split": split,
        "mse": outputs.mse,
        **_metric_columns("context", context),
        **_metric_columns("target", target),
        "prediction_norm_mean": prediction_norm.value,
        "prediction_norm_mean_reason": prediction_norm.reason,
        "context_target_parameter_distance": distance.value,
        "context_target_parameter_distance_reason": distance.reason,
        "elapsed_seconds": elapsed_seconds,
    }
    for name in _GRADIENT_FIELDS:
        metric = gradients[name] if gradients is not None else MetricValue(None, "not_training")
        row[name] = metric.value
        row[f"{name}_reason"] = metric.reason
    return row


def _environment_metadata() -> dict[str, Any]:
    try:
        package_version = version("jepa")
    except PackageNotFoundError:
        package_version = "source"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "jepa": package_version,
        "git_commit": git_commit,
    }


def _config_artifact(
    config: ExperimentConfig,
    *,
    run_id: str,
    initial_hash: str,
    resume_history: list[dict[str, Any]],
    datasets: DatasetSplits,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config": config_to_dict(config),
        "identity": {
            "initial_config_hash": initial_hash,
            "current_config_hash": config_identity_hash(config),
            "model_seed": derive_seed(config.training.seed, config.model.architecture, "model"),
        },
        "derived_seeds": {
            "system": config.data.system_seed,
            "train_samples": config.data.train_sample_seed,
            "validation_samples": config.data.validation_sample_seed,
            "test_samples": config.data.test_sample_seed,
            "training": config.training.seed,
            "epoch_order_rule": "sha256(training.seed, 'train-order', epoch)",
        },
        "resume_history": resume_history,
        "system": {
            "transition": datasets.system.transition.tolist(),
            "observation": datasets.system.observation.tolist(),
        },
        "environment": _environment_metadata(),
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(
    *,
    config: ExperimentConfig,
    initial_config: Mapping[str, Any],
    initial_hash: str,
    run_id: str,
    epoch: int,
    device: torch.device,
    core: JEPACore,
    optimizer: torch.optim.Optimizer,
    datasets: DatasetSplits,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "epoch": epoch,
        "config": config_to_dict(config),
        "initial_config": dict(initial_config),
        "config_hash": config_identity_hash(config),
        "initial_config_hash": initial_hash,
        "device": str(device),
        "context_encoder": core.context_encoder.state_dict(),
        "predictor": core.predictor.state_dict(),
        "target_encoder": core.target_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "system": {
            "transition": datasets.system.transition,
            "observation": datasets.system.observation,
        },
        "rng": _rng_state(),
        "elapsed_seconds": elapsed_seconds,
    }


def _resume_compatible(stored: ExperimentConfig, current: ExperimentConfig, epoch: int) -> None:
    stored_values = config_to_dict(stored)
    current_values = config_to_dict(current)
    stored_epochs = stored_values["training"].pop("epochs")
    current_epochs = current_values["training"].pop("epochs")
    if stored_values != current_values:
        raise ValueError("resume config differs from checkpoint outside training.epochs")
    if current_epochs < stored_epochs or current_epochs < epoch:
        raise ValueError("resume may only preserve or increase training.epochs")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint has an incompatible schema")
    return checkpoint


def _probe_metric(metric: MetricValue) -> dict[str, Any]:
    return {"value": metric.value, "reason": metric.reason}


def _representation_payload(metrics: RepresentationMetrics) -> dict[str, Any]:
    return {
        "latent_std_mean": _probe_metric(metrics.latent_std_mean),
        "effective_rank": _probe_metric(metrics.effective_rank),
        "effective_rank_normalized": _probe_metric(metrics.effective_rank_normalized),
        "top_eigen_fraction": _probe_metric(metrics.top_eigen_fraction),
        "norm_mean": _probe_metric(metrics.vector_norm_mean),
        "collapsed": metrics.collapsed,
        "collapse_reasons": list(metrics.collapse_reasons),
        "eigenvalues": list(metrics.eigenvalues) if metrics.eigenvalues is not None else None,
    }


def _final_metrics(outputs: Mapping[str, SplitOutputs], config: ExperimentConfig) -> dict[str, Any]:
    specifications = {
        "context_to_context_state": ("context_latents", "context_states"),
        "target_to_target_state": ("target_latents", "target_states"),
        "context_to_target_window": ("context_latents", "target_windows"),
    }
    probes: dict[str, Any] = {}
    train = outputs["train"]
    for name, (feature_name, target_name) in specifications.items():
        probe = fit_ridge_probe(
            getattr(train, feature_name),
            getattr(train, target_name),
            alpha=config.evaluation.probe_ridge,
        )
        probes[name] = {
            split: _probe_metric(
                ridge_r2_score(
                    probe,
                    getattr(split_outputs, feature_name),
                    getattr(split_outputs, target_name),
                )
            )
            for split, split_outputs in outputs.items()
        }
    representations = {
        split: {
            "context": _representation_payload(
                compute_representation_metrics(value.context_latents, config.evaluation)
            ),
            "target": _representation_payload(
                compute_representation_metrics(value.target_latents, config.evaluation)
            ),
            "prediction_norm_mean": torch.linalg.vector_norm(value.predictions.double(), dim=1)
            .mean()
            .item(),
        }
        for split, value in outputs.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "objective": {split: value.mse for split, value in outputs.items()},
        "representations": representations,
        "probes": probes,
        "artifacts": {
            "config": "config.yaml",
            "status": "status.json",
            "history": "history.csv",
            "metrics": "metrics.json",
            "checkpoint": "checkpoint.pt",
        },
    }


def train_experiment(
    config: ExperimentConfig,
    *,
    resume_from: str | Path | None = None,
    seed_label: int | None = None,
) -> TrainResult:
    """Train one JEPA run and durably record enough state for exact CPU resume."""
    validate_config(config)
    started_at = _utc_now()
    start_time = time.monotonic()
    checkpoint: dict[str, Any] | None = None
    resume_history: list[dict[str, Any]] = []
    if resume_from is None:
        run_id = build_run_id(config, seed_label)
        run_dir = Path(config.output.root).expanduser().resolve() / run_id
        if run_dir.exists() and any(run_dir.iterdir()):
            if not config.output.overwrite:
                raise FileExistsError(f"run directory already exists: {run_dir}")
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        initial_hash = config_identity_hash(config)
        initial_config = config_to_dict(config)
        completed_epoch = 0
        elapsed_offset = 0.0
        history: list[dict[str, Any]] = []
    else:
        checkpoint_path = Path(resume_from).expanduser().resolve()
        checkpoint = _load_checkpoint(checkpoint_path)
        run_dir = checkpoint_path.parent
        run_id = checkpoint["run_id"]
        status_path = run_dir / "status.json"
        if not status_path.exists():
            raise ValueError("resume requires status.json beside the checkpoint")
        stored_status = json.loads(status_path.read_text())
        if stored_status.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("status.json has an incompatible schema")
        if stored_status.get("run_id") != run_id:
            raise ValueError("status.json run ID differs from checkpoint")
        if stored_status.get("state") == "complete":
            raise ValueError("completed runs cannot be resumed")
        stored_config = config_from_dict(checkpoint["config"])
        if checkpoint.get("config_hash") != config_identity_hash(stored_config):
            raise ValueError("checkpoint config hash does not match its stored config")
        completed_epoch = int(checkpoint["epoch"])
        if completed_epoch < 0 or completed_epoch > stored_config.training.epochs:
            raise ValueError("checkpoint epoch is inconsistent with its stored config")
        _resume_compatible(stored_config, config, completed_epoch)
        if checkpoint["device"] != "cpu" or config.training.device not in {"cpu", "auto"}:
            raise ValueError("exact resume is supported only for CPU checkpoints")
        initial_hash = checkpoint["initial_config_hash"]
        initial_config = checkpoint.get("initial_config", checkpoint["config"])
        elapsed_offset = float(checkpoint.get("elapsed_seconds", 0.0))
        history = _read_history(run_dir / "history.csv")
        if any(row["run_id"] != run_id for row in history):
            raise ValueError("history.csv run ID differs from checkpoint")
        artifact_path = run_dir / "config.yaml"
        if not artifact_path.exists():
            raise ValueError("resume requires config.yaml beside the checkpoint")
        existing_artifact = yaml.safe_load(artifact_path.read_text()) or {}
        if existing_artifact.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("config.yaml has an incompatible schema")
        if existing_artifact.get("run_id") != run_id:
            raise ValueError("config.yaml run ID differs from checkpoint")
        artifact_initial_hash = existing_artifact.get("identity", {}).get("initial_config_hash")
        if artifact_initial_hash != initial_hash:
            raise ValueError("config.yaml initial identity differs from checkpoint")
        history = [
            row for row in history if row["epoch"] <= completed_epoch and row["split"] != "test"
        ]
        _atomic_history(run_dir / "history.csv", history)
        resume_history = list(existing_artifact.get("resume_history", []))
        resume_history.append(
            {
                "resumed_at": started_at,
                "checkpoint_epoch": completed_epoch,
                "previous_config_hash": checkpoint["config_hash"],
                "current_config_hash": config_identity_hash(config),
            }
        )

    status_path = run_dir / "status.json"
    status = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "state": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "completed_epoch": completed_epoch,
        "config_hash": config_identity_hash(config),
        "error": None,
    }
    _atomic_json(status_path, status)
    try:
        device = _resolve_device(config.training.device)
        if checkpoint is not None:
            device = torch.device("cpu")
        _seed_everything(derive_seed(config.training.seed, config.model.architecture, "model"))
        datasets = build_dataset_splits(config.data)
        input_dim = config.data.window_size * config.data.observation_dim
        core = build_jepa_core(
            config.model.architecture,
            input_dim=input_dim,
            latent_dim=config.model.latent_dim,
            stop_gradient=config.training.stop_gradient,
            ema_enabled=config.training.ema.enabled,
            hidden_dim=config.model.hidden_dim,
            hidden_layers=config.model.hidden_layers,
        )
        core.context_encoder.to(device)
        core.predictor.to(device)
        core.target_encoder.to(device)
        optimizer = torch.optim.Adam(
            optimizer_parameters(core),
            lr=config.training.learning_rate,
            betas=config.training.betas,
            eps=config.training.epsilon,
            amsgrad=config.training.amsgrad,
        )
        if checkpoint is not None:
            expected = checkpoint["system"]
            if not torch.equal(datasets.system.transition, expected["transition"]):
                raise ValueError("regenerated transition matrix differs from checkpoint")
            if not torch.equal(datasets.system.observation, expected["observation"]):
                raise ValueError("regenerated observation matrix differs from checkpoint")
            core.context_encoder.load_state_dict(checkpoint["context_encoder"])
            core.predictor.load_state_dict(checkpoint["predictor"])
            core.target_encoder.load_state_dict(checkpoint["target_encoder"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            _restore_rng_state(checkpoint["rng"])

        _atomic_yaml(
            run_dir / "config.yaml",
            _config_artifact(
                config,
                run_id=run_id,
                initial_hash=initial_hash,
                resume_history=resume_history,
                datasets=datasets,
            ),
        )
        latest_outputs: dict[str, SplitOutputs] = {}
        for epoch in range(completed_epoch + 1, config.training.epochs + 1):
            gradients = _train_epoch(core, optimizer, datasets.train, config, device, epoch)
            due_evaluation = (
                epoch % config.training.evaluation_every_epochs == 0
                or epoch == config.training.epochs
            )
            elapsed = elapsed_offset + time.monotonic() - start_time
            if due_evaluation:
                for split, dataset in (
                    ("train", datasets.train),
                    ("validation", datasets.validation),
                ):
                    outputs = _collect_split_outputs(
                        core, dataset, config.training.batch_size, device
                    )
                    latest_outputs[split] = outputs
                    _append_history(
                        history,
                        _history_row(
                            run_id=run_id,
                            epoch=epoch,
                            split=split,
                            outputs=outputs,
                            evaluation_config=config.evaluation,
                            gradients=gradients if split == "train" else None,
                            distance=_parameter_distance(core),
                            elapsed_seconds=elapsed,
                        ),
                    )
                _atomic_history(run_dir / "history.csv", history)
            due_checkpoint = (
                epoch % config.training.checkpoint_every_epochs == 0
                or epoch == config.training.epochs
            )
            if due_checkpoint:
                _atomic_torch_save(
                    run_dir / "checkpoint.pt",
                    _checkpoint_payload(
                        config=config,
                        initial_config=initial_config,
                        initial_hash=initial_hash,
                        run_id=run_id,
                        epoch=epoch,
                        device=device,
                        core=core,
                        optimizer=optimizer,
                        datasets=datasets,
                        elapsed_seconds=elapsed,
                    ),
                )
            completed_epoch = epoch
            status.update(
                state="running",
                updated_at=_utc_now(),
                completed_epoch=completed_epoch,
            )
            _atomic_json(status_path, status)

        final_epoch = config.training.epochs
        for split, dataset in (
            ("train", datasets.train),
            ("validation", datasets.validation),
            ("test", datasets.test),
        ):
            if split not in latest_outputs or split == "test":
                latest_outputs[split] = _collect_split_outputs(
                    core, dataset, config.training.batch_size, device
                )
            if split == "test":
                _append_history(
                    history,
                    _history_row(
                        run_id=run_id,
                        epoch=final_epoch,
                        split=split,
                        outputs=latest_outputs[split],
                        evaluation_config=config.evaluation,
                        gradients=None,
                        distance=_parameter_distance(core),
                        elapsed_seconds=elapsed_offset + time.monotonic() - start_time,
                    ),
                )
        _atomic_history(run_dir / "history.csv", history)
        metrics = _final_metrics(latest_outputs, config)
        metrics.update(
            run_id=run_id,
            epoch=final_epoch,
            policy=asdict(core.policy),
            config_hash=config_identity_hash(config),
        )
        validation_rows = [row for row in history if row["split"] == "validation"]
        best_validation = min(validation_rows, key=lambda row: row["mse"])
        metrics["validation"] = {
            "final_mse": latest_outputs["validation"].mse,
            "best_mse": best_validation["mse"],
            "best_epoch": best_validation["epoch"],
        }
        _atomic_json(run_dir / "metrics.json", metrics)
        status.update(
            state="complete",
            updated_at=_utc_now(),
            completed_at=_utc_now(),
            completed_epoch=final_epoch,
        )
        _atomic_json(status_path, status)
        return TrainResult(run_id, run_dir, tuple(history), metrics)
    except Exception as error:
        status.update(
            state="failed",
            updated_at=_utc_now(),
            completed_epoch=completed_epoch,
            error={"type": type(error).__name__, "message": str(error)},
        )
        _atomic_json(status_path, status)
        raise
