#!/usr/bin/env python3
"""Run the legacy patch-CNN weighting benchmark on corrupted Shapes3D.

This intentionally preserves the pre-upstream-I-JEPA architecture so the
corruption hypothesis can be tested independently of the architecture rewrite.

    CUDA_VISIBLE_DEVICES=0 python scripts/images/train_shapes3d_corrupted_legacy.py \
        --run-name corrupted_uniform --weighting-method uniform
    tensorboard --logdir outputs/shapes3d_corrupted_legacy
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from jepa.analysis.subspace import compute_latent_spectrum  # noqa: E402
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import (  # noqa: E402
    load_shapes3d_config,
    shapes3d_config_to_dict,
)
from jepa.data.images.corruptions import (  # noqa: E402
    CORRUPTION_NAMES,
    corrupt_images,
)
from jepa.data.images.shapes3d import (  # noqa: E402
    build_shapes3d_counterfactual_pairs,
    build_shapes3d_static_dataset_splits,
)
from jepa.models.patches import patchify  # noqa: E402
from jepa.training.core import ema_update  # noqa: E402
from jepa.training.images.legacy_ijepa_spatial import (  # noqa: E402
    MaskConfig,
    build_spatial_ijepa_core,
    encode_frames_pooled_batched,
    encode_samples_pooled,
    load_spatial_checkpoint,
    sample_masks,
    save_spatial_checkpoint,
    spatial_ijepa_loss,
)
from jepa.training.images.legacy_spatial_curriculum import (  # noqa: E402
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

ARCHITECTURE = "cnn"
PATCH_SIZE = 8
PATCH_LATENT_DIM = 16
LR = 0.001
EMA_DECAY = 0.99
BATCH_SIZE = 128
TOTAL_EPOCHS = 700
SEED = 0
EVAL_EVERY_EPOCHS = 10
CHECKPOINT_EVERY_EPOCHS = 50
CHECKPOINT_EVERY_STEPS = 0
LOG_EVERY_STEPS = 25
OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2] / "outputs" / "shapes3d_corrupted_legacy"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None, help="Directory name under --output-root")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--architecture",
        choices=("cnn", "resnet"),
        default=ARCHITECTURE,
        help="Patch encoder architecture. resnet is randomly initialized, not pretrained.",
    )
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
        choices=("uniform", "loss", "ras", "coord", "oracle-clean"),
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
    parser.add_argument("--corruption-fraction", type=float, default=0.2)
    parser.add_argument(
        "--corruption-mode",
        choices=("mixed", "noise", "blur", "occlusion", "blank"),
        default="mixed",
    )
    parser.add_argument("--corruption-seed", type=int, default=1701)
    parser.add_argument("--corruption-noise-std", type=float, default=0.75)
    return parser.parse_args()


def _default_run_name(seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")  # noqa: UP017
    return f"spatial_ijepa_seed{seed}_{stamp}"


def _checkpoint_metadata(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    return {
        "architecture": args.architecture,
        "model_family": "legacy_patch_encoder",
        "patch_size": PATCH_SIZE,
        "patch_latent_dim": PATCH_LATENT_DIM,
        "ema_decay": EMA_DECAY,
        "run_dir": str(run_dir),
        "dataset": "shapes3d-corrupted",
        "data_source": "static_shapes3d_corrupted_images",
        "corruption_fraction": args.corruption_fraction,
        "corruption_mode": args.corruption_mode,
        "corruption_seed": args.corruption_seed,
        "corruption_noise_std": args.corruption_noise_std,
    }


def _weighting_checkpoint_state(
    weighting_state: SpatialWeightingState,
    ref_indices: torch.Tensor,
    corrupted: torch.Tensor,
    corruption_kind_ids: torch.Tensor,
) -> dict[str, object]:
    return {
        "weighting_memory": weighting_state.memory,
        "weighting_probabilities": weighting_state.probabilities,
        "weighting_last_scores": weighting_state.last_scores,
        "weighting_ref_indices": ref_indices,
        "coordinate_importance": weighting_state.coordinate_importance,
        "coordinate_previous_ref_latents": weighting_state.coordinate_previous_ref_latents,
        "corrupted_mask": corrupted,
        "corruption_kind_ids": corruption_kind_ids,
    }


def _corruption_diagnostics(
    state: SpatialWeightingState,
    *,
    corrupted: torch.Tensor,
    kind_ids: torch.Tensor,
) -> dict[str, float]:
    probabilities = state.probabilities.detach().cpu()
    scores = state.last_scores.detach().cpu()
    clean = ~corrupted
    top_count = max(1, int(round(0.1 * probabilities.numel())))
    top_indices = torch.topk(probabilities, top_count).indices
    mean_clean_probability = probabilities[clean].mean().clamp_min(1e-12)
    if corrupted.any():
        probability_ratio = probabilities[corrupted].mean() / mean_clean_probability
        score_mean_corrupted = scores[corrupted].mean()
    else:
        probability_ratio = torch.zeros((), dtype=probabilities.dtype)
        score_mean_corrupted = torch.zeros((), dtype=scores.dtype)
    result = {
        "corruption/probability_mass": float(probabilities[corrupted].sum().item()),
        "corruption/mean_probability_ratio": float(probability_ratio.item()),
        "corruption/top_10pct_fraction": float(
            corrupted[top_indices].to(torch.float64).mean().item()
        ),
        "corruption/score_mean_clean": float(scores[clean].mean().item()),
        "corruption/score_mean_corrupted": float(score_mean_corrupted.item()),
    }
    for kind_id, kind_name in enumerate(CORRUPTION_NAMES[1:], start=1):
        kind_mask = kind_ids == kind_id
        result[f"corruption/probability_mass_{kind_name}"] = float(
            probabilities[kind_mask].sum().item()
        )
    return result


def _restore_weighting_state(
    checkpoint: dict[str, object],
    *,
    num_frames: int,
    corrupted: torch.Tensor,
    corruption_kind_ids: torch.Tensor,
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
    saved_corrupted = extra_state.get("corrupted_mask")
    saved_kind_ids = extra_state.get("corruption_kind_ids")
    if not isinstance(saved_corrupted, torch.Tensor) or not torch.equal(
        saved_corrupted.detach().cpu().to(torch.bool), corrupted
    ):
        raise ValueError("resume checkpoint corruption mask does not match this run")
    if not isinstance(saved_kind_ids, torch.Tensor) or not torch.equal(
        saved_kind_ids.detach().cpu().to(torch.long), corruption_kind_ids
    ):
        raise ValueError("resume checkpoint corruption kinds do not match this run")

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
    samples: torch.Tensor,
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
    for batch in samples.split(batch_size):
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
    samples: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    core.context_encoder.eval()
    encoded = []
    for batch in samples.split(batch_size):
        encoded.append(encode_samples_pooled(core, batch.to(device)).detach().cpu())
    return torch.cat(encoded, dim=0)


def main() -> None:
    args = _parse_args()
    run_name = args.run_name or _default_run_name(args.seed)
    run_dir = Path(args.output_root).expanduser().resolve() / run_name
    checkpoint_dir = run_dir / "network"
    checkpoint_dir.mkdir(parents=True, exist_ok=args.resume_from is not None)
    logger = SpatialRunLogger(run_dir, enable_tensorboard=not args.no_tensorboard)
    curriculum_method = (
        "uniform" if args.weighting_method == "oracle-clean" else args.weighting_method
    )
    weighting_config = SpatialWeightingConfig(
        method=curriculum_method,
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

    logger.write_config(
        {
            **config_payload,
            "spatial": {
                "architecture": args.architecture,
                "patch_size": PATCH_SIZE,
                "patch_latent_dim": PATCH_LATENT_DIM,
                "learning_rate": args.lr,
                "ema_decay": EMA_DECAY,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "dataset": "shapes3d-corrupted",
                "data_source": "static_shapes3d_corrupted_images",
                "eval_every_epochs": args.eval_every_epochs,
                "checkpoint_every_epochs": args.checkpoint_every_epochs,
                "checkpoint_every_steps": args.checkpoint_every_steps,
                "log_every_steps": args.log_every_steps,
                "resident_device_data": args.resident_device_data,
                "corruption": {
                    "fraction": args.corruption_fraction,
                    "mode": args.corruption_mode,
                    "seed": args.corruption_seed,
                    "noise_std": args.corruption_noise_std,
                },
                "weighting": {
                    "method": args.weighting_method,
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
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    corrupted_batch = corrupt_images(
        datasets.train.images,
        fraction=args.corruption_fraction,
        seed=args.corruption_seed,
        mode=args.corruption_mode,
        noise_std=args.corruption_noise_std,
    )
    train_frames = corrupted_batch.images
    corrupted_mask = corrupted_batch.corrupted
    corruption_kind_ids = corrupted_batch.kind_ids
    test_frames = datasets.test.images
    train_patches = patchify(train_frames, PATCH_SIZE)
    test_patches = patchify(test_frames, PATCH_SIZE)
    transform_pair_patches: tuple[torch.Tensor, torch.Tensor] | None = None
    if (
        weighting_config.method == "coord"
        and weighting_config.coordinate_importance == "transformation"
    ):
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
        f"dataset=shapes3d-corrupted architecture={args.architecture} device={device} "
        f"train_images={train_frames.shape[0]} "
        f"corrupted={int(corrupted_mask.sum())}/{len(corrupted_mask)} "
        f"num_patches={num_patches} patch_dim={patch_dim}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    core = build_spatial_ijepa_core(
        args.architecture,
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
    clean_mask = ~corrupted_mask
    if not clean_mask.any():
        raise ValueError("corruption setup must leave at least one clean training image")
    start = time.time()
    global_step = 0
    metadata = _checkpoint_metadata(args, run_dir)
    weighting_state = init_spatial_weighting(n)
    if args.weighting_method == "oracle-clean":
        oracle_probabilities = clean_mask.to(torch.float64)
        oracle_probabilities /= oracle_probabilities.sum()
        weighting_state = SpatialWeightingState(
            memory=torch.zeros(n, dtype=torch.float64),
            probabilities=oracle_probabilities,
            last_scores=torch.zeros(n, dtype=torch.float64),
        )
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
            corrupted=corrupted_mask,
            corruption_kind_ids=corruption_kind_ids,
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
            if args.weighting_method == "oracle-clean":
                order = sample_frame_indices(
                    weighting_state,
                    num_draws=n,
                    seed=derive_seed(args.seed, "oracle-order", epoch),
                )
            elif weighting_config.method != "uniform" and epoch > weighting_config.warmup_epochs:
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
            sampled_corruption_fraction = float(
                corrupted_mask[order].to(torch.float64).mean().item()
            )
            if args.resident_device_data:
                order = order.to(device)
            batch_chunks = order.split(args.batch_size)
            mask_generator = torch.Generator().manual_seed(derive_seed(args.seed, "masks", epoch))
            epoch_loss, batches = 0.0, 0
            epoch_start = time.time()
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
                            weighting_state,
                            weighting_ref_indices,
                            corrupted_mask,
                            corruption_kind_ids,
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
                            weighting_state,
                            weighting_ref_indices,
                            corrupted_mask,
                            corruption_kind_ids,
                        ),
                    )

            epoch_mean_loss = epoch_loss / batches
            logger.log(
                step=global_step,
                epoch=epoch,
                event="epoch",
                scalars={
                    "train/epoch_loss": epoch_mean_loss,
                    "train/epoch_seconds": time.time() - epoch_start,
                    "train/batches_per_epoch": batches,
                    "train/elapsed_seconds": time.time() - start,
                    "corruption/sampled_fraction": sampled_corruption_fraction,
                    **weighting_diagnostics(weighting_state),
                    **_corruption_diagnostics(
                        weighting_state,
                        corrupted=corrupted_mask,
                        kind_ids=corruption_kind_ids,
                    ),
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
                    scalars={
                        **weighting_diagnostics(weighting_state),
                        **score_metadata,
                        **_corruption_diagnostics(
                            weighting_state,
                            corrupted=corrupted_mask,
                            kind_ids=corruption_kind_ids,
                        ),
                    },
                    histograms={
                        "hist/weighting_scores": weighting_state.last_scores,
                        "hist/weighting_probabilities": weighting_state.probabilities,
                    },
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
                    test_z = encode_frames_pooled_batched(
                        core,
                        test_frames,
                        patch_size=PATCH_SIZE,
                        batch_size=args.batch_size,
                    )
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
                    extra_state=_weighting_checkpoint_state(
                        weighting_state,
                        weighting_ref_indices,
                        corrupted_mask,
                        corruption_kind_ids,
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
                        weighting_state,
                        weighting_ref_indices,
                        corrupted_mask,
                        corruption_kind_ids,
                    ),
                )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
