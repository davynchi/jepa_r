"""Training loop for the rendered-image entity/context experiment.

``temporal_image`` / ``temporal_image_shuffled`` reuse every per-epoch training
helper from :mod:`jepa.temporal_training` (``_standard_epoch_loss``,
``_hierarchical_epoch_loss``, ``build_jepa_core``, ``build_hierarchical_core``,
``encode_split_standard``, ``encode_split_hierarchical``) unmodified: those
functions only touch ``config.training`` / ``config.model`` / ``config.hierarchical``
and a duck-typed dataset interface, both of which :class:`ImageExperimentConfig`
and :class:`EntityContextImageTrajectoryDataset` satisfy exactly (see
:mod:`jepa.temporal_image_data`). Only the static spatial-JEPA control (no
horizon, no temporal pairing) needed a small dedicated loop.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from jepa.config import derive_seed
from jepa.metrics import WeightedMean, compute_representation_metrics
from jepa.temporal_image_config import (
    ImageExperimentConfig,
    image_config_identity_hash,
    image_config_to_dict,
)
from jepa.temporal_image_data import (
    StaticImageDataset,
    build_image_dataset_splits,
    build_static_spatial_dataset,
)
from jepa.temporal_training import (
    _batch_slices,
    _hierarchical_epoch_loss,
    build_hierarchical_core,
    encode_split_standard,
)
from jepa.temporal_training import _standard_epoch_loss as _standard_epoch_loss
from jepa.training import (
    SCHEMA_VERSION,
    _atomic_torch_save,
    _atomic_yaml,
    _resolve_device,
    _rng_state,
    _seed_everything,
    _utc_now,
    build_jepa_core,
    compute_loss,
    ema_update,
    optimizer_parameters,
)
from jepa.training import _atomic_json as _atomic_json
from jepa.training import _environment_metadata as _environment_metadata


def build_image_run_id(config: ImageExperimentConfig, seed_label: int | None = None) -> str:
    label = config.training.seed if seed_label is None else seed_label
    return (
        f"image_{config.dataset_type}_kind-{config.model.kind}_arch-{config.model.architecture}"
        f"_pE-{config.data.entity_switch_probability}_pair-{config.training.target_pairing}"
        f"_seed-{label}_cfg-{image_config_identity_hash(config)[:8]}"
    )


def _static_epoch_loss(
    core,
    visible: torch.Tensor,
    target: torch.Tensor,
    config: ImageExperimentConfig,
    device: torch.device,
    epoch: int,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    n = visible.shape[0]
    loss_mean = WeightedMean()
    core.context_encoder.train(train)
    core.predictor.train(train)
    core.target_encoder.train(train and core.policy.optimize_target)
    for indices in _batch_slices(
        n,
        config.training.batch_size,
        seed=derive_seed(config.training.seed, "train-order", epoch),
        shuffle=train,
    ):
        context = visible[indices].reshape(len(indices), -1).to(device)
        target_batch = target[indices].reshape(len(indices), -1).to(device)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            forward = compute_loss(core, context, target_batch)
            forward.loss.backward()
            optimizer.step()
            if core.policy.ema_enabled:
                ema_update(core.target_encoder, core.context_encoder, config.training.ema.decay)
        else:
            with torch.no_grad():
                forward = compute_loss(core, context, target_batch)
        loss_mean.update(forward.loss.item(), len(indices))
    result = loss_mean.compute()
    if result.value is None:
        raise RuntimeError("empty epoch")
    return result.value


def _system_free_checkpoint(**payload: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, **payload}


@dataclass(frozen=True, slots=True)
class ImageTrainResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]


def train_temporal_image_experiment(
    config: ImageExperimentConfig, *, seed_label: int | None = None
) -> ImageTrainResult:
    started_at = _utc_now()
    start_time = time.monotonic()
    run_id = build_image_run_id(config, seed_label)
    run_dir = Path(config.output.root).expanduser().resolve() / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        if not config.output.overwrite:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "status.json"
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "state": "running",
        "started_at": started_at,
        "error": None,
    }
    _atomic_json(status_path, status)
    try:
        device = _resolve_device(config.training.device)
        _seed_everything(derive_seed(config.training.seed, config.model.architecture, "model"))
        history: list[dict[str, Any]] = []
        is_random = config.model.kind == "random"

        if config.dataset_type == "temporal_image":
            datasets = build_image_dataset_splits(config.data)
            effective_kind = "standard" if is_random else config.model.kind
            if effective_kind == "hierarchical":
                core = build_hierarchical_core(config)
                core.online.to(device)
                core.target_encoder.to(device)
                optimizer = torch.optim.Adam(
                    _hierarchical_parameters(core),
                    lr=config.training.learning_rate,
                    betas=config.training.betas,
                    eps=config.training.epsilon,
                )
                epochs = 0 if is_random else config.training.epochs
                for epoch in range(1, epochs + 1):
                    train_losses = _hierarchical_epoch_loss(
                        core, datasets.train, config, device, epoch, train=True, optimizer=optimizer
                    )
                    val_losses = _hierarchical_epoch_loss(
                        core, datasets.validation, config, device, epoch, train=False
                    )
                    history.append(
                        {
                            "epoch": epoch,
                            "split": "train",
                            **{f"loss_{k}": v for k, v in train_losses.items()},
                        }
                    )
                    history.append(
                        {
                            "epoch": epoch,
                            "split": "validation",
                            **{f"loss_{k}": v for k, v in val_losses.items()},
                        }
                    )
                test_losses = _hierarchical_epoch_loss(
                    core, datasets.test, config, device, epochs, train=False
                )
                final_test_loss = test_losses["total"]
                checkpoint = _system_free_checkpoint(
                    run_id=run_id,
                    model_kind="hierarchical",
                    dataset_type=config.dataset_type,
                    config=image_config_to_dict(config),
                    online_encoder=core.online.encoder.state_dict(),
                    entity_head=core.online.entity_head.state_dict(),
                    context_head=core.online.context_head.state_dict(),
                    target_encoder=core.target_encoder.state_dict(),
                    rng=_rng_state(),
                )
            else:
                core = build_jepa_core(
                    config.model.architecture,
                    input_dim=config.data.observation_dim,
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
                )
                epochs = 0 if is_random else config.training.epochs
                for epoch in range(1, epochs + 1):
                    train_loss = _standard_epoch_loss(
                        core, datasets.train, config, device, epoch, train=True, optimizer=optimizer
                    )
                    val_loss = _standard_epoch_loss(
                        core, datasets.validation, config, device, epoch, train=False
                    )
                    history.append({"epoch": epoch, "split": "train", "loss_total": train_loss})
                    history.append({"epoch": epoch, "split": "validation", "loss_total": val_loss})
                    if epoch % max(1, config.training.evaluation_every_epochs) == 0:
                        representations = encode_split_standard(core, datasets.validation)
                        rank_metrics = compute_representation_metrics(
                            representations.reshape(-1, representations.shape[-1])
                        )
                        history[-1]["effective_rank"] = rank_metrics.effective_rank.value
                final_test_loss = _standard_epoch_loss(
                    core, datasets.test, config, device, max(epochs, 1), train=False
                )
                checkpoint = _system_free_checkpoint(
                    run_id=run_id,
                    model_kind=config.model.kind,
                    dataset_type=config.dataset_type,
                    config=image_config_to_dict(config),
                    context_encoder=core.context_encoder.state_dict(),
                    predictor=core.predictor.state_dict(),
                    target_encoder=core.target_encoder.state_dict(),
                    rng=_rng_state(),
                )
        else:
            if config.model.kind == "hierarchical":
                raise ValueError(
                    "the hierarchical model is only supported for dataset_type=temporal_image"
                )
            exclude_none = config.dataset_type == "static_image_spatial_control"
            train_static = StaticImageDataset(config.data, "train", exclude_none=exclude_none)
            val_static = StaticImageDataset(config.data, "validation", exclude_none=exclude_none)
            test_static = StaticImageDataset(config.data, "test", exclude_none=exclude_none)
            train_spatial = build_static_spatial_dataset(
                train_static,
                config.spatial,
                seed=derive_seed(config.training.seed, "mask", "train"),
            )
            val_spatial = build_static_spatial_dataset(
                val_static,
                config.spatial,
                seed=derive_seed(config.training.seed, "mask", "validation"),
            )
            test_spatial = build_static_spatial_dataset(
                test_static, config.spatial, seed=derive_seed(config.training.seed, "mask", "test")
            )
            core = build_jepa_core(
                config.model.architecture,
                input_dim=config.data.observation_dim,
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
            )
            epochs = 0 if is_random else config.training.epochs
            for epoch in range(1, epochs + 1):
                train_loss = _static_epoch_loss(
                    core,
                    train_spatial.visible,
                    train_spatial.target,
                    config,
                    device,
                    epoch,
                    train=True,
                    optimizer=optimizer,
                )
                val_loss = _static_epoch_loss(
                    core,
                    val_spatial.visible,
                    val_spatial.target,
                    config,
                    device,
                    epoch,
                    train=False,
                )
                history.append({"epoch": epoch, "split": "train", "loss_total": train_loss})
                history.append({"epoch": epoch, "split": "validation", "loss_total": val_loss})
            final_test_loss = _static_epoch_loss(
                core,
                test_spatial.visible,
                test_spatial.target,
                config,
                device,
                max(epochs, 1),
                train=False,
            )
            checkpoint = _system_free_checkpoint(
                run_id=run_id,
                model_kind=config.model.kind,
                dataset_type=config.dataset_type,
                config=image_config_to_dict(config),
                context_encoder=core.context_encoder.state_dict(),
                predictor=core.predictor.state_dict(),
                target_encoder=core.target_encoder.state_dict(),
                rng=_rng_state(),
            )

        _atomic_torch_save(run_dir / "checkpoint.pt", checkpoint)
        _atomic_yaml(
            run_dir / "config.yaml",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "config": image_config_to_dict(config),
                "environment": _environment_metadata(),
            },
        )
        with (run_dir / "history.json").open("w") as stream:
            json.dump(history, stream, indent=2)

        metrics = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "test_loss": final_test_loss,
            "elapsed_seconds": time.monotonic() - start_time,
        }
        _atomic_json(run_dir / "metrics.json", metrics)
        status.update(state="complete", updated_at=_utc_now(), completed_at=_utc_now())
        _atomic_json(status_path, status)
        return ImageTrainResult(run_id, run_dir, metrics)
    except Exception as error:
        status.update(
            state="failed",
            updated_at=_utc_now(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _atomic_json(status_path, status)
        raise


def _hierarchical_parameters(core):
    groups = [core.online.parameters()]
    if core.policy.optimize_target:
        groups.append(core.target_encoder.parameters())
    parameters = []
    seen: set[int] = set()
    for group in groups:
        for parameter in group:
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return tuple(parameters)


def load_image_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint has an incompatible schema")
    return checkpoint
