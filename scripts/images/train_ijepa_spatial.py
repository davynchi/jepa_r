#!/usr/bin/env python3
"""Train the CNN I-JEPA-style spatial model on Shapes3D frames and watch for
representational collapse (the failure mode that killed every earlier
whole-frame variant). Purely spatial: no temporal pairing at all, so this does
not depend on the frames being related in time.

    CUDA_VISIBLE_DEVICES=1 python scripts/images/train_ijepa_spatial.py --run-name uniform_seed0
    tensorboard --logdir outputs/ijepa_spatial
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from jepa.analysis.quality_features import LiveSpatialAdapter  # noqa: E402
from jepa.analysis.quality_manifests import (  # noqa: E402
    QualitySplitManifest,
    build_quality_manifest,
)
from jepa.analysis.quality_metrics import METRIC_SPEC_BY_NAME  # noqa: E402
from jepa.analysis.quality_store import QualityStore  # noqa: E402
from jepa.analysis.quality_timing import (  # noqa: E402
    QualityTimingRecorder,
    isolated_quality_evaluation,
    synchronize_device,
)
from jepa.analysis.subspace import compute_latent_spectrum  # noqa: E402
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import (  # noqa: E402
    load_shapes3d_config,
    shapes3d_config_identity_hash,
    shapes3d_config_to_dict,
)
from jepa.data.images.shapes3d import (  # noqa: E402
    build_shapes3d_counterfactual_pairs,
    build_shapes3d_static_dataset_splits,
)
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    build_tiny_imagenet_static_dataset_splits,
)
from jepa.models.patches import patchify  # noqa: E402
from jepa.training.core import ema_update  # noqa: E402
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    MaskConfig,
    build_spatial_ijepa_core,
    encode_frames_pooled,
    load_spatial_checkpoint,
    sample_masks,
    save_spatial_checkpoint,
    spatial_ijepa_loss,
)
from jepa.training.images.spatial_curriculum import (  # noqa: E402
    SpatialWeightingConfig,
    SpatialWeightingState,
    init_spatial_weighting,
    sample_frame_indices,
    score_frames_by_coordinate_importance,
    score_frames_by_loss,
    score_frames_by_ras,
    select_reference_indices,
    should_update_weights,
    update_coordinate_weighting_state,
    update_spatial_weights,
    weighting_diagnostics,
)
from jepa.training.images.spatial_logging import (  # noqa: E402
    SpatialRunLogger,
    eigenvalue_scalars,
)
from scripts.analysis.evaluate_spatial_quality import (  # noqa: E402
    _evaluate_checkpoint as evaluate_quality_observation,
)
from scripts.analysis.evaluate_spatial_quality import (  # noqa: E402
    _heldout_loss as quality_heldout_loss,
)
from scripts.analysis.evaluate_spatial_quality import (  # noqa: E402
    _write_checkpoint_loss as write_quality_loss,
)

ARCHITECTURE = "cnn"
PATCH_SIZE = 8
PATCH_LATENT_DIM = 16
LR = 0.001
EMA_DECAY = 0.99
BATCH_SIZE = 128
TOTAL_EPOCHS = 1500
SEED = 0
EVAL_EVERY_EPOCHS = 10
CHECKPOINT_EVERY_EPOCHS = 50
CHECKPOINT_EVERY_STEPS = 0
LOG_EVERY_STEPS = 25
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "ijepa_spatial"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None, help="Directory name under --output-root")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dataset", choices=("shapes3d", "tiny-imagenet"), default="shapes3d")
    parser.add_argument("--tiny-imagenet-root", default="data/tiny-imagenet-200")
    parser.add_argument("--num-train-samples", type=int, default=16000)
    parser.add_argument("--num-val-samples", type=int, default=1000)
    parser.add_argument("--num-test-samples", type=int, default=1000)
    parser.add_argument("--eval-every-epochs", type=int, default=EVAL_EVERY_EPOCHS)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=CHECKPOINT_EVERY_EPOCHS)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=CHECKPOINT_EVERY_STEPS,
        help="0 disables step-cadence checkpoints",
    )
    parser.add_argument("--log-every-steps", type=int, default=LOG_EVERY_STEPS)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--resident-device-data",
        action="store_true",
        help="Keep patch tensors on the training device to reduce CPU/GPU transfer overhead",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume model, optimizer, global step, and weighting state from this checkpoint",
    )
    parser.add_argument(
        "--weighting-method",
        choices=("uniform", "loss", "ras", "coord"),
        default="uniform",
    )
    parser.add_argument("--weighting-warmup-epochs", type=int, default=0)
    parser.add_argument("--weighting-update-every-epochs", type=int, default=1)
    parser.add_argument("--weighting-temperature", type=float, default=1.0)
    parser.add_argument("--weighting-replay-beta", type=float, default=1.0)
    parser.add_argument("--weighting-uniform-mix", type=float, default=0.05)
    parser.add_argument(
        "--weighting-score-batch-size",
        type=int,
        default=0,
        help="0 reuses --batch-size",
    )
    parser.add_argument("--weighting-ref-size", type=int, default=1024)
    parser.add_argument(
        "--weighting-richness",
        choices=("logdet", "rbar", "pr"),
        default="logdet",
        help="Richness functional used by --weighting-method ras",
    )
    parser.add_argument("--weighting-richness-delta", type=float, default=1.0e-4)
    parser.add_argument("--weighting-richness-trace-target", type=float, default=1.0)
    parser.add_argument("--weighting-richness-trace-beta", type=float, default=0.01)
    parser.add_argument(
        "--coordinate-importance",
        choices=("covariance", "transformation", "dynamics"),
        default="covariance",
        help="Coordinate importance estimator used by --weighting-method coord",
    )
    parser.add_argument("--coordinate-ema-beta", type=float, default=0.1)
    parser.add_argument("--coordinate-delta", type=float, default=1.0e-6)
    parser.add_argument(
        "--online-quality",
        action="store_true",
        help="Evaluate every registered Q metric directly from the live model after each epoch",
    )
    parser.add_argument(
        "--online-quality-every-epochs",
        type=int,
        default=1,
        help="Live quality cadence; the 100-epoch study uses 1",
    )
    parser.add_argument(
        "--online-quality-root",
        default=None,
        help="Defaults to RUN_DIR/online-quality",
    )
    parser.add_argument("--online-quality-manifest-seed", type=int, default=58_031)
    parser.add_argument("--online-quality-feature-batch-size", type=int, default=128)
    return parser.parse_args()


def _default_run_name(seed: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"spatial_ijepa_cnn_seed{seed}_{stamp}"


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _checkpoint_metadata(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    return {
        "architecture": ARCHITECTURE,
        "patch_size": PATCH_SIZE,
        "patch_latent_dim": PATCH_LATENT_DIM,
        "ema_decay": EMA_DECAY,
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "data_source": f"static_{args.dataset}_images",
    }


def _weighting_checkpoint_state(
    weighting_state: SpatialWeightingState,
    ref_indices: torch.Tensor,
) -> dict[str, object]:
    return {
        "weighting_memory": weighting_state.memory,
        "weighting_probabilities": weighting_state.probabilities,
        "weighting_last_scores": weighting_state.last_scores,
        "weighting_ref_indices": ref_indices,
        "coordinate_importance": weighting_state.coordinate_importance,
        "coordinate_previous_ref_latents": weighting_state.coordinate_previous_ref_latents,
    }


def _restore_weighting_state(
    checkpoint: dict[str, object],
    *,
    num_frames: int,
) -> tuple[SpatialWeightingState, torch.Tensor]:
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, dict):
        raise ValueError("resume checkpoint is missing weighting extra_state")

    memory = extra_state.get("weighting_memory")
    probabilities = extra_state.get("weighting_probabilities")
    last_scores = extra_state.get("weighting_last_scores")
    ref_indices = extra_state.get("weighting_ref_indices")
    if not all(isinstance(value, torch.Tensor) for value in (memory, probabilities, last_scores)):
        raise ValueError("resume checkpoint has incomplete weighting tensors")
    if not isinstance(ref_indices, torch.Tensor):
        raise ValueError("resume checkpoint is missing weighting_ref_indices")
    if memory.shape != (num_frames,) or probabilities.shape != (num_frames,):
        raise ValueError("resume checkpoint weighting state does not match train set size")
    if last_scores.shape != (num_frames,):
        raise ValueError("resume checkpoint weighting scores do not match train set size")

    coordinate_importance = extra_state.get("coordinate_importance")
    coordinate_previous_ref_latents = extra_state.get("coordinate_previous_ref_latents")
    if isinstance(coordinate_importance, torch.Tensor):
        coordinate_importance = coordinate_importance.detach().cpu().to(torch.float64)
    else:
        coordinate_importance = None
    if isinstance(coordinate_previous_ref_latents, torch.Tensor):
        coordinate_previous_ref_latents = (
            coordinate_previous_ref_latents.detach().cpu().to(torch.float64)
        )
    else:
        coordinate_previous_ref_latents = None

    state = SpatialWeightingState(
        memory=memory.detach().cpu().to(torch.float64),
        probabilities=probabilities.detach().cpu().to(torch.float64),
        last_scores=last_scores.detach().cpu().to(torch.float64),
        coordinate_importance=coordinate_importance,
        coordinate_previous_ref_latents=coordinate_previous_ref_latents,
    )
    return state, ref_indices.detach().cpu().to(torch.long)


@torch.no_grad()
def _evaluate_spatial_loss(
    core,
    patches: torch.Tensor,
    *,
    grid: int,
    mask_config: MaskConfig,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    core.context_encoder.eval()
    core.predictor.eval()
    core.target_encoder.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_examples = 0
    for batch in patches.split(batch_size):
        batch = batch.to(device)
        context_masks, target_masks = sample_masks(grid, grid, mask_config, generator)
        loss = torch.zeros((), device=device)
        for context_mask in context_masks:
            for target_mask in target_masks:
                loss = loss + spatial_ijepa_loss(
                    core, batch, context_mask.to(device), target_mask.to(device)
                )
        loss = loss / (len(context_masks) * len(target_masks))
        total_loss += float(loss.item()) * batch.shape[0]
        total_examples += batch.shape[0]
    return total_loss / total_examples


@torch.no_grad()
def _encode_patches_pooled(
    core,
    patches: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    core.context_encoder.eval()
    encoded = []
    for batch in patches.split(batch_size):
        encoded.append(core.context_encoder(batch.to(device)).mean(dim=-2).detach().cpu())
    return torch.cat(encoded, dim=0)


def main() -> None:
    args = _parse_args()
    if args.online_quality:
        if args.dataset != "shapes3d":
            raise ValueError("--online-quality currently requires --dataset shapes3d")
        if args.weighting_method != "uniform":
            raise ValueError("--online-quality requires --weighting-method uniform")
        if args.online_quality_every_epochs <= 0:
            raise ValueError("--online-quality-every-epochs must be positive")
        if args.online_quality_feature_batch_size <= 0:
            raise ValueError("--online-quality-feature-batch-size must be positive")
    run_name = args.run_name or _default_run_name(args.seed)
    run_dir = Path(args.output_root).expanduser().resolve() / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=args.resume_from is not None)
    logger = SpatialRunLogger(run_dir, enable_tensorboard=not args.no_tensorboard)
    weighting_config = SpatialWeightingConfig(
        method=args.weighting_method,
        warmup_epochs=args.weighting_warmup_epochs,
        update_every_epochs=args.weighting_update_every_epochs,
        temperature=args.weighting_temperature,
        replay_beta=args.weighting_replay_beta,
        uniform_mix=args.weighting_uniform_mix,
        score_batch_size=args.weighting_score_batch_size,
        ref_size=args.weighting_ref_size,
        richness_functional=args.weighting_richness,
        richness_delta=args.weighting_richness_delta,
        richness_trace_target=args.weighting_richness_trace_target,
        richness_trace_beta=args.weighting_richness_trace_beta,
        coordinate_importance=args.coordinate_importance,
        coordinate_ema_beta=args.coordinate_ema_beta,
        coordinate_delta=args.coordinate_delta,
    )

    if args.dataset == "shapes3d":
        config = load_shapes3d_config(
            "configs/images/shapes3d/quick.yaml",
            overrides={
                "data.num_train_samples": str(args.num_train_samples),
                "data.num_val_samples": str(args.num_val_samples),
                "data.num_test_samples": str(args.num_test_samples),
                "training.device": args.device,
            },
        )
        datasets = build_shapes3d_static_dataset_splits(config.data)
        config_payload: dict[str, object] = {"config": shapes3d_config_to_dict(config)}
    elif args.dataset == "tiny-imagenet":
        tiny_config = TinyImageNetDataConfig(
            root=args.tiny_imagenet_root,
            num_train_samples=args.num_train_samples,
            num_val_samples=args.num_val_samples,
            num_test_samples=args.num_test_samples,
        )
        datasets = build_tiny_imagenet_static_dataset_splits(tiny_config)
        config_payload = {
            "tiny_imagenet": {
                "data": asdict(tiny_config),
                "classes": [
                    {"index": index, "wnid": wnid, "name": name}
                    for index, (wnid, name) in enumerate(
                        zip(datasets.train.wnids, datasets.train.class_names, strict=True)
                    )
                ],
            }
        }
    else:
        raise ValueError(f"unsupported dataset: {args.dataset!r}")

    logger.write_config(
        {
            **config_payload,
            "spatial": {
                "architecture": ARCHITECTURE,
                "patch_size": PATCH_SIZE,
                "patch_latent_dim": PATCH_LATENT_DIM,
                "learning_rate": args.lr,
                "ema_decay": EMA_DECAY,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "dataset": args.dataset,
                "data_source": f"static_{args.dataset}_images",
                "eval_every_epochs": args.eval_every_epochs,
                "checkpoint_every_epochs": args.checkpoint_every_epochs,
                "checkpoint_every_steps": args.checkpoint_every_steps,
                "log_every_steps": args.log_every_steps,
                "resident_device_data": args.resident_device_data,
                "online_quality": {
                    "enabled": args.online_quality,
                    "every_epochs": args.online_quality_every_epochs,
                    "manifest_seed": args.online_quality_manifest_seed,
                    "feature_batch_size": args.online_quality_feature_batch_size,
                    "source_kind": "live_epoch",
                },
                "weighting": {
                    "method": weighting_config.method,
                    "warmup_epochs": weighting_config.warmup_epochs,
                    "update_every_epochs": weighting_config.update_every_epochs,
                    "temperature": weighting_config.temperature,
                    "replay_beta": weighting_config.replay_beta,
                    "uniform_mix": weighting_config.uniform_mix,
                    "score_batch_size": weighting_config.score_batch_size,
                    "ref_size": weighting_config.ref_size,
                    "richness_functional": weighting_config.richness_functional,
                    "richness_delta": weighting_config.richness_delta,
                    "richness_trace_target": weighting_config.richness_trace_target,
                    "richness_trace_beta": weighting_config.richness_trace_beta,
                    "coordinate_importance": weighting_config.coordinate_importance,
                    "coordinate_ema_beta": weighting_config.coordinate_ema_beta,
                    "coordinate_delta": weighting_config.coordinate_delta,
                },
            },
        }
    )
    device = _resolve_device(args.device)

    train_frames = datasets.train.images
    test_frames = datasets.test.images
    train_patches = patchify(train_frames, PATCH_SIZE)
    test_patches = patchify(test_frames, PATCH_SIZE)
    transform_pair_patches: tuple[torch.Tensor, torch.Tensor] | None = None
    if (
        weighting_config.method == "coord"
        and weighting_config.coordinate_importance == "transformation"
    ):
        if args.dataset != "shapes3d":
            raise ValueError("transformation coordinate importance is currently Shapes3D-only")
        pairs = build_shapes3d_counterfactual_pairs(
            config.data,
            datasets.train.source,
            num_pairs=weighting_config.ref_size,
            seed=derive_seed(args.seed, "coordinate-transform-pairs"),
        )
        transform_pair_patches = (
            patchify(pairs.same_entity_x1, PATCH_SIZE),
            patchify(pairs.same_entity_x2, PATCH_SIZE),
        )
    grid = 64 // PATCH_SIZE
    num_patches = train_patches.shape[1]
    patch_dim = train_patches.shape[2]
    if args.resident_device_data:
        train_patches = train_patches.to(device)
        test_patches = test_patches.to(device)
        if transform_pair_patches is not None:
            transform_pair_patches = (
                transform_pair_patches[0].to(device),
                transform_pair_patches[1].to(device),
            )
    print(
        f"run_dir={run_dir}\n"
        f"dataset={args.dataset} device={device} train_images={train_frames.shape[0]} "
        f"num_patches={num_patches} patch_dim={patch_dim}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    core = build_spatial_ijepa_core(
        ARCHITECTURE,
        patch_dim=patch_dim,
        patch_latent_dim=PATCH_LATENT_DIM,
        num_patches=num_patches,
    )
    core.context_encoder.to(device)
    core.predictor.to(device)
    core.target_encoder.to(device)
    params = list(core.context_encoder.parameters()) + list(core.predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    mask_config = MaskConfig()

    n = train_patches.shape[0]
    quality_output: Path | None = None
    quality_manifest: QualitySplitManifest | None = None
    quality_store: QualityStore | None = None
    quality_previous = None
    if args.online_quality:
        quality_output = (
            Path(args.online_quality_root).expanduser().resolve()
            if args.online_quality_root
            else run_dir / "online-quality"
        )
        quality_output.mkdir(parents=True, exist_ok=True)
        manifest_path = quality_output / "quality_split_manifest.json"
        data_config_hash = shapes3d_config_identity_hash(config)
        if manifest_path.exists():
            quality_manifest = QualitySplitManifest.read(manifest_path)
            if quality_manifest.data_config_hash != data_config_hash:
                raise ValueError(
                    "existing online-quality manifest was created for a different "
                    "Shapes3D data configuration; choose another --online-quality-root"
                )
            if quality_manifest.manifest_seed != args.online_quality_manifest_seed:
                raise ValueError(
                    "existing online-quality manifest uses a different seed; choose "
                    "another --online-quality-root or reuse its "
                    "--online-quality-manifest-seed"
                )
        else:
            quality_manifest = build_quality_manifest(
                config.data,
                data_config_hash=data_config_hash,
                manifest_seed=args.online_quality_manifest_seed,
            )
            quality_manifest.write(manifest_path)
        quality_store = QualityStore(quality_output / "quality.sqlite")
        quality_store.recover_interrupted()
        (quality_output / "metric_dictionary.json").write_text(
            json.dumps(
                [
                    {
                        "name": spec.name,
                        "q_number": spec.q_number,
                        "version": spec.version,
                        "direction": spec.direction.value,
                        "role": spec.role.value,
                        "family": spec.family,
                        "cost_tier": spec.cost_tier.value,
                        "primary_eligible": spec.primary_eligible,
                    }
                    for spec in METRIC_SPEC_BY_NAME.values()
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    start = time.time()
    global_step = 0
    online_epoch_wall_times: list[float] = []
    metadata = _checkpoint_metadata(args, run_dir)
    weighting_state = init_spatial_weighting(n)
    weighting_ref_indices = select_reference_indices(
        n,
        ref_size=weighting_config.ref_size,
        seed=derive_seed(args.seed, "weighting-ref"),
    )
    start_epoch = 1
    if args.resume_from is not None:
        checkpoint_path = Path(args.resume_from).expanduser().resolve()
        checkpoint = load_spatial_checkpoint(checkpoint_path)
        completed_epoch = int(checkpoint["epoch"])
        if completed_epoch >= args.epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {completed_epoch}, "
                f"but --epochs is {args.epochs}"
            )
        core.context_encoder.load_state_dict(checkpoint["context_encoder"])
        core.predictor.load_state_dict(checkpoint["predictor"])
        core.target_encoder.load_state_dict(checkpoint["target_encoder"])
        if "optimizer" not in checkpoint:
            raise ValueError("resume checkpoint is missing optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(
            checkpoint.get("global_step") or completed_epoch * max(n // args.batch_size, 1)
        )
        weighting_state, weighting_ref_indices = _restore_weighting_state(
            checkpoint,
            num_frames=n,
        )
        start_epoch = completed_epoch + 1
        print(
            f"resumed_from={checkpoint_path} start_epoch={start_epoch} global_step={global_step}",
            flush=True,
        )
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            core.context_encoder.train()
            core.predictor.train()
            if weighting_config.method != "uniform" and epoch > weighting_config.warmup_epochs:
                order = sample_frame_indices(
                    weighting_state,
                    num_draws=n,
                    seed=derive_seed(args.seed, "weighted-order", epoch),
                )
            else:
                order = torch.randperm(
                    n,
                    generator=torch.Generator().manual_seed(derive_seed(args.seed, "order", epoch)),
                )
            if args.resident_device_data:
                order = order.to(device)
            batch_chunks = order.split(args.batch_size)
            mask_generator = torch.Generator().manual_seed(derive_seed(args.seed, "masks", epoch))
            epoch_loss, batches = 0.0, 0
            synchronize_device(device)
            epoch_started_ns = time.perf_counter_ns()
            for indices in batch_chunks:
                global_step += 1
                batch = train_patches[indices].to(device)
                context_masks, target_masks = sample_masks(grid, grid, mask_config, mask_generator)
                optimizer.zero_grad(set_to_none=True)
                # Upstream averages the loss over every (context, target) mask pair.
                loss = torch.zeros((), device=device)
                for context_mask in context_masks:
                    for target_mask in target_masks:
                        loss = loss + spatial_ijepa_loss(
                            core, batch, context_mask.to(device), target_mask.to(device)
                        )
                loss = loss / (len(context_masks) * len(target_masks))
                loss.backward()
                optimizer.step()
                if core.policy.ema_enabled:
                    ema_update(core.target_encoder, core.context_encoder, EMA_DECAY)
                loss_value = loss.item()
                epoch_loss += loss_value
                batches += 1

                if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                    logger.log(
                        step=global_step,
                        epoch=epoch,
                        event="train_step",
                        scalars={
                            "train/loss": loss_value,
                            "train/lr": args.lr,
                            "train/epoch_fraction": epoch + batches / max(len(batch_chunks), 1),
                        },
                    )
                if (
                    args.checkpoint_every_steps > 0
                    and global_step % args.checkpoint_every_steps == 0
                ):
                    save_spatial_checkpoint(
                        checkpoint_dir / f"step_{global_step:08d}.pt",
                        core,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer=optimizer,
                        metadata=metadata,
                        extra_state=_weighting_checkpoint_state(
                            weighting_state, weighting_ref_indices
                        ),
                    )
                    save_spatial_checkpoint(
                        checkpoint_dir / "latest.pt",
                        core,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer=optimizer,
                        metadata=metadata,
                        extra_state=_weighting_checkpoint_state(
                            weighting_state, weighting_ref_indices
                        ),
                    )

            synchronize_device(device)
            train_epoch_seconds = (
                time.perf_counter_ns() - epoch_started_ns
            ) / 1_000_000_000
            epoch_mean_loss = epoch_loss / batches
            logger.log(
                step=global_step,
                epoch=epoch,
                event="epoch",
                scalars={
                    "train/epoch_loss": epoch_mean_loss,
                    "train/epoch_seconds": train_epoch_seconds,
                    "train/batches_per_epoch": batches,
                    "train/elapsed_seconds": time.time() - start,
                },
            )

            if should_update_weights(epoch, weighting_config):
                score_batch_size = weighting_config.score_batch_size or args.batch_size
                if weighting_config.method == "loss":
                    scores = score_frames_by_loss(
                        core,
                        train_patches,
                        grid=grid,
                        mask_config=mask_config,
                        batch_size=score_batch_size,
                        seed=derive_seed(args.seed, "weighting", epoch),
                        device=device,
                    )
                    score_metadata: dict[str, float] = {}
                elif weighting_config.method == "ras":
                    scores, score_metadata = score_frames_by_ras(
                        core,
                        train_patches,
                        ref_indices=weighting_ref_indices,
                        grid=grid,
                        mask_config=mask_config,
                        batch_size=score_batch_size,
                        seed=derive_seed(args.seed, "weighting", epoch),
                        device=device,
                        richness_functional=weighting_config.richness_functional,
                        richness_delta=weighting_config.richness_delta,
                        richness_trace_target=weighting_config.richness_trace_target,
                        richness_trace_beta=weighting_config.richness_trace_beta,
                    )
                elif weighting_config.method == "coord":
                    (
                        scores,
                        score_metadata,
                        coordinate_importance,
                        coordinate_previous_ref_latents,
                    ) = score_frames_by_coordinate_importance(
                        core,
                        train_patches,
                        ref_indices=weighting_ref_indices,
                        state=weighting_state,
                        grid=grid,
                        mask_config=mask_config,
                        batch_size=score_batch_size,
                        seed=derive_seed(args.seed, "weighting", epoch),
                        device=device,
                        coordinate_importance=weighting_config.coordinate_importance,
                        coordinate_ema_beta=weighting_config.coordinate_ema_beta,
                        coordinate_delta=weighting_config.coordinate_delta,
                        transform_pair_patches=transform_pair_patches,
                    )
                else:
                    raise ValueError(f"unsupported weighting method: {weighting_config.method}")
                weighting_state = update_spatial_weights(weighting_state, scores, weighting_config)
                if weighting_config.method == "coord":
                    weighting_state = update_coordinate_weighting_state(
                        weighting_state,
                        coordinate_importance=coordinate_importance,
                        previous_ref_latents=coordinate_previous_ref_latents,
                    )
                logger.log(
                    step=global_step,
                    epoch=epoch,
                    event="weighting_update",
                    scalars={**weighting_diagnostics(weighting_state), **score_metadata},
                    histograms={
                        "hist/weighting_scores": weighting_state.last_scores,
                        "hist/weighting_probabilities": weighting_state.probabilities,
                    },
                )

            if (
                args.online_quality
                and epoch % args.online_quality_every_epochs == 0
            ):
                assert quality_output is not None
                assert quality_manifest is not None
                assert quality_store is not None
                recorder = QualityTimingRecorder(device)
                observation_id = f"live_epoch_{epoch:04d}"
                adapter = LiveSpatialAdapter(
                    run_dir=run_dir,
                    observation_id=observation_id,
                    epoch=epoch,
                    global_step=global_step,
                    model_seed=args.seed,
                    curriculum="uniform",
                    training_uses_shape_metadata=False,
                    core=core,
                )
                with recorder.stage("quality_total"):
                    with isolated_quality_evaluation(core, device):
                        with recorder.stage("heldout_jepa_loss"):
                            heldout_loss = quality_heldout_loss(
                                adapter,
                                datasets.train.source,
                                quality_manifest,
                                batch_size=args.online_quality_feature_batch_size,
                            )
                        write_quality_loss(
                            quality_output,
                            adapter=adapter,
                            loss=heldout_loss,
                            sample_count=len(
                                quality_manifest.banks["classifier_test"]
                            ),
                        )
                        quality_previous = evaluate_quality_observation(
                            adapter,
                            datasets.train.source,
                            quality_manifest,
                            quality_output,
                            quality_store,
                            batch_size=args.online_quality_feature_batch_size,
                            previous=quality_previous,
                            enabled_metrics=set(METRIC_SPEC_BY_NAME),
                            run_expensive=True,
                            retry_failed=True,
                            cache_features=False,
                            timing=recorder,
                            temporal_null_reason="no_previous_epoch",
                        )
                    with recorder.stage("storage_export"):
                        quality_store.export(quality_output)
                quality_seconds = recorder.stages["quality_total"]
                epoch_wall_seconds = train_epoch_seconds + quality_seconds
                recorder.stages["train_epoch"] = train_epoch_seconds
                recorder.stages["epoch_wall"] = epoch_wall_seconds
                recorder.upsert_jsonl(
                    quality_output / "timings.jsonl",
                    run_id=run_dir.name,
                    epoch=epoch,
                    global_step=global_step,
                )
                online_epoch_wall_times.append(epoch_wall_seconds)
                accuracy_row = quality_store.accuracy_rows()[-1]
                logger.log(
                    step=global_step,
                    epoch=epoch,
                    event="online_quality",
                    scalars={
                        "quality/heldout_jepa_loss": heldout_loss,
                        "quality/classification_accuracy": accuracy_row["accuracy"],
                        "quality/balanced_accuracy": accuracy_row["balanced_accuracy"],
                        "quality/total_seconds": quality_seconds,
                        "quality/factorization_label_free_seconds": recorder.stages.get(
                            "factorization_label_free", 0.0
                        ),
                        "quality/factorization_supervised_seconds": recorder.stages.get(
                            "factorization_supervised", 0.0
                        ),
                        "quality/epoch_wall_seconds": epoch_wall_seconds,
                        "quality/overhead_ratio": quality_seconds
                        / max(train_epoch_seconds, 1e-12),
                    },
                )
                print(
                    f"quality epoch={epoch:4d} loss={heldout_loss:.6f} "
                    f"accuracy={float(accuracy_row['accuracy']):.4f} "
                    f"train_s={train_epoch_seconds:.1f} quality_s={quality_seconds:.1f} "
                    f"overhead={quality_seconds / max(train_epoch_seconds, 1e-12):.2f}x",
                    flush=True,
                )
                if len(online_epoch_wall_times) == 3:
                    median_seconds = sorted(online_epoch_wall_times)[1]
                    remaining = max(args.epochs - epoch, 0)
                    print(
                        f"online_quality_eta_seconds={median_seconds * remaining:.0f} "
                        f"based_on_first_3_epochs=true",
                        flush=True,
                    )

            if epoch % args.eval_every_epochs == 0 or epoch == 1:
                test_loss = _evaluate_spatial_loss(
                    core,
                    test_patches,
                    grid=grid,
                    mask_config=mask_config,
                    batch_size=args.batch_size,
                    seed=derive_seed(args.seed, "eval-loss", epoch),
                    device=device,
                )
                if args.resident_device_data:
                    test_z = _encode_patches_pooled(
                        core,
                        test_patches,
                        batch_size=args.batch_size,
                        device=device,
                    )
                else:
                    test_z = encode_frames_pooled(core, test_frames, patch_size=PATCH_SIZE)
                spectrum = compute_latent_spectrum(test_z.reshape(-1, PATCH_LATENT_DIM))
                scalars = {
                    "eval/test_loss": test_loss,
                    "repr/effective_rank": spectrum.effective_rank,
                    "repr/trace_covariance": spectrum.trace_covariance,
                    "repr/mean_latent_norm": float(test_z.norm(dim=-1).mean().item()),
                    **eigenvalue_scalars(spectrum.eigenvalues),
                }
                logger.log(
                    step=global_step,
                    epoch=epoch,
                    event="eval_spectrum",
                    scalars=scalars,
                    histograms={
                        "hist/latent_values": test_z.reshape(-1),
                        "hist/eigenvalues": torch.as_tensor(spectrum.eigenvalues),
                    },
                )
                print(
                    f"epoch={epoch:4d} step={global_step:7d} loss={epoch_mean_loss:10.6f} "
                    f"test_loss={test_loss:10.6f} "
                    f"effective_rank={spectrum.effective_rank:6.3f} "
                    f"trace_cov={spectrum.trace_covariance:9.4f} "
                    f"elapsed={time.time() - start:.0f}s",
                    flush=True,
                )
            if epoch % args.checkpoint_every_epochs == 0 or epoch == args.epochs:
                save_spatial_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pt",
                    core,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer=optimizer,
                    metadata=metadata,
                    extra_state=_weighting_checkpoint_state(weighting_state, weighting_ref_indices),
                )
                save_spatial_checkpoint(
                    checkpoint_dir / "latest.pt",
                    core,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer=optimizer,
                    metadata=metadata,
                    extra_state=_weighting_checkpoint_state(weighting_state, weighting_ref_indices),
                )
    finally:
        if quality_store is not None:
            quality_store.close()
        logger.close()


if __name__ == "__main__":
    main()
