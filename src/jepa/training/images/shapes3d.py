"""Training loop for the Shapes3D-backed entity/context world.

Reuses
``_standard_epoch_loss``, ``_hierarchical_epoch_loss``, ``build_jepa_core``,
``build_hierarchical_core``, ``encode_split_standard`` from
:mod:`jepa.training.timeseries` unmodified, because ``Shapes3DExperimentConfig``
and ``Shapes3DEntityContextTrajectoryDataset`` satisfy the same duck-typed
interface (``config.training`` / ``config.model`` / ``config.hierarchical``,
``.observations`` / ``.entities`` / ``.contexts`` / ``config.trajectory_length``
/ ``config.observation_dim``) as the vector world.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from jepa.analysis.metrics import compute_representation_metrics
from jepa.configs.base import derive_seed
from jepa.configs.images.shapes3d import (
    Shapes3DExperimentConfig,
    shapes3d_config_identity_hash,
    shapes3d_config_to_dict,
)
from jepa.data.images.shapes3d import Shapes3DDatasetSplits, build_shapes3d_dataset_splits
from jepa.training.core import (
    SCHEMA_VERSION,
    _atomic_torch_save,
    _atomic_yaml,
    _resolve_device,
    _rng_state,
    _seed_everything,
    _utc_now,
    build_jepa_core,
    optimizer_parameters,
)
from jepa.training.core import _atomic_json as _atomic_json
from jepa.training.core import _environment_metadata as _environment_metadata
from jepa.training.timeseries import (
    _hierarchical_epoch_loss,
    build_hierarchical_core,
    encode_split_standard,
)
from jepa.training.timeseries import _standard_epoch_loss as _standard_epoch_loss


def build_shapes3d_run_id(config: Shapes3DExperimentConfig, seed_label: int | None = None) -> str:
    label = config.training.seed if seed_label is None else seed_label
    return (
        f"shapes3d_kind-{config.model.kind}_arch-{config.model.architecture}"
        f"_pE-{config.data.entity_switch_probability}_pair-{config.training.target_pairing}"
        f"_seed-{label}_cfg-{shapes3d_config_identity_hash(config)[:8]}"
    )


@dataclass(frozen=True, slots=True)
class Shapes3DTrainResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]


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


def train_shapes3d_experiment(
    config: Shapes3DExperimentConfig,
    *,
    seed_label: int | None = None,
    datasets: Shapes3DDatasetSplits | None = None,
) -> Shapes3DTrainResult:
    started_at = _utc_now()
    start_time = time.monotonic()
    run_id = build_shapes3d_run_id(config, seed_label)
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
        # config.data is invariant across a whole diagnostic grid (only model/training
        # fields vary between cells), so callers sweeping many cells against the same
        # data config can build the trajectories once and pass them in here, instead
        # of paying this ~20s CPU-bound trajectory-generation cost on every single
        # cell -- see _investigate_shapes3d_pairing_controls.py's dataset cache.
        if datasets is None:
            datasets = build_shapes3d_dataset_splits(config.data)
        history: list[dict[str, Any]] = []
        is_random = config.model.kind == "random"
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
            checkpoint = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "model_kind": "hierarchical",
                "config": shapes3d_config_to_dict(config),
                "online_encoder": core.online.encoder.state_dict(),
                "entity_head": core.online.entity_head.state_dict(),
                "context_head": core.online.context_head.state_dict(),
                "target_encoder": core.target_encoder.state_dict(),
                "rng": _rng_state(),
            }
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
            pairs_cache: dict[int, Any] = {}
            for epoch in range(1, epochs + 1):
                train_loss = _standard_epoch_loss(
                    core,
                    datasets.train,
                    config,
                    device,
                    epoch,
                    train=True,
                    optimizer=optimizer,
                    pairs_cache=pairs_cache,
                )
                val_loss = _standard_epoch_loss(
                    core,
                    datasets.validation,
                    config,
                    device,
                    epoch,
                    train=False,
                    pairs_cache=pairs_cache,
                )
                history.append({"epoch": epoch, "split": "train", "loss_total": train_loss})
                history.append({"epoch": epoch, "split": "validation", "loss_total": val_loss})
                if epoch % max(1, config.training.evaluation_every_epochs) == 0:
                    representations = encode_split_standard(core, datasets.validation)
                    rank_metrics = compute_representation_metrics(
                        representations.reshape(-1, representations.shape[-1])
                    )
                    history[-1]["effective_rank"] = rank_metrics.effective_rank.value
                if not is_random and (
                    epoch % max(1, config.training.checkpoint_every_epochs) == 0 or epoch == epochs
                ):
                    # Per-epoch snapshots (distinct from the final checkpoint.pt below)
                    # so post-hoc analysis (PCA scatter, MI, Q_E) can be run against
                    # intermediate training states, not only the converged encoder.
                    _atomic_torch_save(
                        run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                        {
                            "schema_version": SCHEMA_VERSION,
                            "epoch": epoch,
                            "context_encoder": core.context_encoder.state_dict(),
                        },
                    )
            final_test_loss = _standard_epoch_loss(
                core, datasets.test, config, device, max(epochs, 1), train=False
            )
            checkpoint = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "model_kind": config.model.kind,
                "config": shapes3d_config_to_dict(config),
                "context_encoder": core.context_encoder.state_dict(),
                "predictor": core.predictor.state_dict(),
                "target_encoder": core.target_encoder.state_dict(),
                "rng": _rng_state(),
            }

        _atomic_torch_save(run_dir / "checkpoint.pt", checkpoint)
        _atomic_yaml(
            run_dir / "config.yaml",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "config": shapes3d_config_to_dict(config),
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
        return Shapes3DTrainResult(run_id, run_dir, metrics)
    except Exception as error:
        status.update(
            state="failed",
            updated_at=_utc_now(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _atomic_json(status_path, status)
        raise


def load_shapes3d_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint has an incompatible schema")
    return checkpoint
