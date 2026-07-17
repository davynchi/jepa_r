"""Training loop for the entity/context temporal experiment.

Reuses the existing repository's stop-gradient/EMA policy machinery
(:mod:`jepa.training.core`), atomic artifact writers, representation metrics, and
ridge-probe utilities wherever the objective matches. The hierarchical
entity/context objective (Section 5.4) is new and lives entirely in this
module; it never touches :mod:`jepa.training.core`.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from jepa.analysis.metrics import WeightedMean, compute_representation_metrics
from jepa.configs.base import derive_seed
from jepa.configs.timeseries import (
    TemporalExperimentConfig,
    temporal_config_from_dict,
    temporal_config_identity_hash,
    temporal_config_to_dict,
)
from jepa.data.timeseries.entity_context import (
    DatasetSplits,
    EntityContextTrajectoryDataset,
    build_dataset_splits,
    make_hierarchical_pairs,
    make_temporal_pairs,
)
from jepa.models.hierarchical import HierarchicalEncoderPredictor
from jepa.training.core import (
    SCHEMA_VERSION,
    JEPACore,
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
    resolve_policy,
)
from jepa.training.core import _atomic_json as _atomic_json  # noqa: F401 (re-exported helper)
from jepa.training.core import _environment_metadata as _environment_metadata


def build_temporal_run_id(config: TemporalExperimentConfig, seed_label: int | None = None) -> str:
    label = config.training.seed if seed_label is None else seed_label
    return (
        f"temporal_kind-{config.model.kind}_arch-{config.model.architecture}"
        f"_obs-{config.data.observation_mode}_pE-{config.data.entity_switch_probability}"
        f"_h-{config.training.prediction_horizon}_pair-{config.training.target_pairing}"
        f"_seed-{label}_cfg-{temporal_config_identity_hash(config)[:8]}"
    )


@dataclass(slots=True)
class HierarchicalCore:
    online: HierarchicalEncoderPredictor
    target_encoder: nn.Module
    policy: Any


def build_hierarchical_core(config: TemporalExperimentConfig) -> HierarchicalCore:
    online = HierarchicalEncoderPredictor(
        config.model.architecture,
        input_dim=config.data.observation_dim,
        latent_dim=config.model.latent_dim,
        entity_latent_dim=config.model.entity_latent_dim,
        context_latent_dim=config.model.context_latent_dim,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
    )
    policy = resolve_policy(
        stop_gradient=config.training.stop_gradient, ema_enabled=config.training.ema.enabled
    )
    target_encoder = deepcopy(online.encoder) if policy.separate_target else online.encoder
    if policy.separate_target and not policy.optimize_target:
        target_encoder.requires_grad_(False)
    return HierarchicalCore(online, target_encoder, policy)


def _batch_std(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    return x.std(dim=0, unbiased=True)


def _cov_trace(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    centered = x - x.mean(dim=0, keepdim=True)
    return (centered.square().sum(dim=0) / (x.shape[0] - 1)).sum()


def _cross_covariance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[0] < 2:
        return torch.zeros(a.shape[-1], b.shape[-1], dtype=a.dtype, device=a.device)
    a_centered = a - a.mean(dim=0, keepdim=True)
    b_centered = b - b.mean(dim=0, keepdim=True)
    return (a_centered.T @ b_centered) / (a.shape[0] - 1)


@dataclass(slots=True)
class HierarchicalForward:
    outputs: Any
    loss: torch.Tensor
    loss_entity: float
    loss_context: float
    loss_var: float
    loss_cross: float


def hierarchical_loss(
    core: HierarchicalCore,
    context_view: torch.Tensor,
    short_target_view: torch.Tensor,
    long_target_view: torch.Tensor,
    config: TemporalExperimentConfig,
) -> HierarchicalForward:
    hierarchical = config.hierarchical
    outputs = core.online(context_view)

    with torch.no_grad() if not core.policy.optimize_target else torch.enable_grad():
        target_short_latent = core.target_encoder(short_target_view)
        target_long_latent = core.target_encoder(long_target_view)
    entity_dim = config.model.entity_latent_dim
    target_context_block = target_short_latent[..., entity_dim:]
    target_entity_block = target_long_latent[..., :entity_dim]
    if core.policy.stop_gradient:
        target_context_block = target_context_block.detach()
        target_entity_block = target_entity_block.detach()

    entity_error = (outputs.predicted_entity - target_entity_block).square().sum(dim=-1).mean()
    context_error = (outputs.predicted_context - target_context_block).square().sum(dim=-1).mean()
    # Detached denominators: this is a *normalizer*, not a training signal. Without
    # detach(), gradients flow through entity_scale/context_scale too, and the model
    # can cheaply shrink the loss by inflating z_E/z_C variance (which the variance
    # floor below does not penalize -- it only penalizes variance being too *low*)
    # instead of actually improving prediction accuracy.
    entity_scale = _cov_trace(outputs.entity_latent).detach() + hierarchical.epsilon
    context_scale = _cov_trace(outputs.context_latent).detach() + hierarchical.epsilon
    loss_entity = entity_error / entity_scale
    loss_context = context_error / context_scale

    if hierarchical.use_variance_reg:
        entity_std = _batch_std(outputs.entity_latent)
        context_std = _batch_std(outputs.context_latent)
        loss_var = (
            torch.relu(hierarchical.variance_gamma - entity_std).sum()
            + torch.relu(hierarchical.variance_gamma - context_std).sum()
        )
    else:
        loss_var = torch.zeros((), dtype=outputs.latent.dtype, device=outputs.latent.device)

    if hierarchical.use_cross_cov_reg:
        cross = _cross_covariance(outputs.entity_latent, outputs.context_latent)
        loss_cross = cross.square().sum()
    else:
        loss_cross = torch.zeros((), dtype=outputs.latent.dtype, device=outputs.latent.device)

    total = (
        hierarchical.lambda_entity * loss_entity
        + hierarchical.lambda_context * loss_context
        + hierarchical.lambda_var * loss_var
        + hierarchical.lambda_cross * loss_cross
    )
    return HierarchicalForward(
        outputs=outputs,
        loss=total,
        loss_entity=float(loss_entity.detach().item()),
        loss_context=float(loss_context.detach().item()),
        loss_var=float(loss_var.detach().item()),
        loss_cross=float(loss_cross.detach().item()),
    )


def _batch_slices(n: int, batch_size: int, *, seed: int, shuffle: bool) -> list[torch.Tensor]:
    if shuffle:
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    else:
        order = torch.arange(n)
    return list(order.split(batch_size))


@dataclass(frozen=True, slots=True)
class TemporalTrainResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]


def _effective_horizon(config: TemporalExperimentConfig) -> tuple[int, int]:
    if config.hierarchical.use_horizon_split:
        return config.hierarchical.short_horizon, config.hierarchical.long_horizon
    return config.hierarchical.short_horizon, config.hierarchical.short_horizon


@dataclass(frozen=True, slots=True)
class _DeviceResidentPairs:
    context: torch.Tensor
    target: torch.Tensor


def _standard_epoch_loss(
    core: JEPACore,
    dataset: EntityContextTrajectoryDataset,
    config: TemporalExperimentConfig,
    device: torch.device,
    epoch: int,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    pairs_cache: dict[int, _DeviceResidentPairs] | None = None,
) -> float:
    cache_key = id(dataset)
    resident = pairs_cache.get(cache_key) if pairs_cache is not None else None
    if resident is None:
        pairs = make_temporal_pairs(
            dataset,
            horizon=config.training.prediction_horizon,
            pairing=config.training.target_pairing,
            seed=derive_seed(config.training.seed, "pairs", "standard", epoch),
        )
        # One bulk host->device transfer instead of one per mini-batch -- for
        # image worlds the per-observation payload (e.g. Shapes3D's 3x64x64
        # floats) makes many small transfers the dominant per-epoch cost and
        # starves the GPU between them (observed: near-0% utilization on the
        # CNN encoder despite the conv forward/backward itself being cheap).
        resident = _DeviceResidentPairs(
            context=pairs.context_view.to(device), target=pairs.target_view.to(device)
        )
        # "temporal" pairing is provably epoch-invariant: target_traj_index ==
        # source_traj_index regardless of seed (see make_temporal_pairs), so
        # rebuilding it from scratch -- and re-transferring it -- every epoch is
        # pure waste. "shuffled"/"shuffled_same_entity" intentionally
        # re-randomize per epoch and are therefore not cached.
        if config.training.target_pairing == "temporal" and pairs_cache is not None:
            pairs_cache[cache_key] = resident
    n = resident.context.shape[0]
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
        device_indices = indices.to(device)
        context = resident.context[device_indices]
        target = resident.target[device_indices]
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            forward = compute_loss(core, context, target)
            forward.loss.backward()
            optimizer.step()
            if core.policy.ema_enabled:
                ema_update(core.target_encoder, core.context_encoder, config.training.ema.decay)
        else:
            with torch.no_grad():
                forward = compute_loss(core, context, target)
        loss_mean.update(forward.loss.item(), len(indices))
    result = loss_mean.compute()
    if result.value is None:
        raise RuntimeError("empty epoch")
    return result.value


def _hierarchical_epoch_loss(
    core: HierarchicalCore,
    dataset: EntityContextTrajectoryDataset,
    config: TemporalExperimentConfig,
    device: torch.device,
    epoch: int,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    short_horizon, long_horizon = _effective_horizon(config)
    pairs = make_hierarchical_pairs(
        dataset,
        short_horizon=short_horizon,
        long_horizon=long_horizon,
        pairing=config.training.target_pairing,
        seed=derive_seed(config.training.seed, "pairs", "hierarchical", epoch),
    )
    n = pairs.context_view.shape[0]
    aggregates = {name: WeightedMean() for name in ("total", "entity", "context", "var", "cross")}
    core.online.train(train)
    core.target_encoder.train(train and core.policy.optimize_target)
    for indices in _batch_slices(
        n,
        config.training.batch_size,
        seed=derive_seed(config.training.seed, "train-order", epoch),
        shuffle=train,
    ):
        context_view = pairs.context_view[indices].to(device)
        short_target = pairs.short_target_view[indices].to(device)
        long_target = pairs.long_target_view[indices].to(device)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            forward = hierarchical_loss(core, context_view, short_target, long_target, config)
            forward.loss.backward()
            optimizer.step()
            if core.policy.ema_enabled:
                ema_update(core.target_encoder, core.online.encoder, config.training.ema.decay)
        else:
            with torch.no_grad():
                forward = hierarchical_loss(core, context_view, short_target, long_target, config)
        weight = len(indices)
        aggregates["total"].update(forward.loss.item(), weight)
        aggregates["entity"].update(forward.loss_entity, weight)
        aggregates["context"].update(forward.loss_context, weight)
        aggregates["var"].update(forward.loss_var, weight)
        aggregates["cross"].update(forward.loss_cross, weight)
    return {name: mean.compute().value for name, mean in aggregates.items()}


@torch.no_grad()
@torch.no_grad()
def encode_split_standard(core: JEPACore, dataset: EntityContextTrajectoryDataset) -> torch.Tensor:
    core.context_encoder.eval()
    device = next(core.context_encoder.parameters()).device
    flat = dataset.observations.reshape(-1, dataset.config.observation_dim).to(device)
    latents = core.context_encoder(flat)
    return latents.reshape(*dataset.observations.shape[:-1], -1)


@torch.no_grad()
def encode_split_hierarchical(
    core: HierarchicalCore, dataset: EntityContextTrajectoryDataset
) -> torch.Tensor:
    core.online.eval()
    device = next(core.online.parameters()).device
    flat = dataset.observations.reshape(-1, dataset.config.observation_dim).to(device)
    latents = core.online.encode(flat)
    return latents.reshape(*dataset.observations.shape[:-1], -1)


def train_temporal_experiment(
    config: TemporalExperimentConfig, *, seed_label: int | None = None
) -> TemporalTrainResult:
    """Train (or, for the random baseline, simply initialize) one run."""
    started_at = _utc_now()
    start_time = time.monotonic()
    run_id = build_temporal_run_id(config, seed_label)
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
        datasets: DatasetSplits = build_dataset_splits(config.data)

        history: list[dict[str, Any]] = []
        is_random = config.model.kind == "random"
        # The random-encoder baseline (Section 5.1) always uses the plain
        # encoder/predictor structure with zero optimization steps; only a
        # trained run may use the hierarchical entity/context architecture.
        effective_kind = "standard" if is_random else config.model.kind
        if effective_kind == "hierarchical":
            hier_core = build_hierarchical_core(config)
            hier_core.online.to(device)
            hier_core.target_encoder.to(device)
            optimizer = torch.optim.Adam(
                _hierarchical_parameters(hier_core),
                lr=config.training.learning_rate,
                betas=config.training.betas,
                eps=config.training.epsilon,
            )
            epochs = 0 if is_random else config.training.epochs
            for epoch in range(1, epochs + 1):
                train_losses = _hierarchical_epoch_loss(
                    hier_core,
                    datasets.train,
                    config,
                    device,
                    epoch,
                    train=True,
                    optimizer=optimizer,
                )
                val_losses = _hierarchical_epoch_loss(
                    hier_core, datasets.validation, config, device, epoch, train=False
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
                hier_core, datasets.test, config, device, epochs, train=False
            )
            checkpoint = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "model_kind": "hierarchical",
                "config": temporal_config_to_dict(config),
                "online_encoder": hier_core.online.encoder.state_dict(),
                "entity_head": hier_core.online.entity_head.state_dict(),
                "context_head": hier_core.online.context_head.state_dict(),
                "target_encoder": hier_core.target_encoder.state_dict(),
                "system": _system_payload(datasets.system),
                "rng": _rng_state(),
            }
            final_test_loss = test_losses["total"]
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
            pairs_cache: dict[int, _DeviceResidentPairs] = {}
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
            test_loss = _standard_epoch_loss(
                core, datasets.test, config, device, max(epochs, 1), train=False
            )
            checkpoint = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "model_kind": config.model.kind,
                "config": temporal_config_to_dict(config),
                "context_encoder": core.context_encoder.state_dict(),
                "predictor": core.predictor.state_dict(),
                "target_encoder": core.target_encoder.state_dict(),
                "system": _system_payload(datasets.system),
                "rng": _rng_state(),
            }
            final_test_loss = test_loss

        _atomic_torch_save(run_dir / "checkpoint.pt", checkpoint)
        _atomic_yaml(
            run_dir / "config.yaml",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "config": temporal_config_to_dict(config),
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
        return TemporalTrainResult(run_id, run_dir, metrics)
    except Exception as error:
        status.update(
            state="failed",
            updated_at=_utc_now(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _atomic_json(status_path, status)
        raise


def _hierarchical_parameters(core: HierarchicalCore) -> tuple[nn.Parameter, ...]:
    groups = [core.online.parameters()]
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


def _system_payload(system: Any) -> dict[str, Any]:
    payload = {"mode": system.mode, "linear_map": system.linear_map}
    if system.hidden_map is not None:
        payload.update(
            hidden_map=system.hidden_map,
            hidden_bias=system.hidden_bias,
            output_map=system.output_map,
            output_bias=system.output_bias,
        )
    return payload


def load_temporal_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint has an incompatible schema")
    return checkpoint


def rebuild_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> TemporalExperimentConfig:
    return temporal_config_from_dict(checkpoint["config"])
