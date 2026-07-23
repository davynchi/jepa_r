#!/usr/bin/env python3
"""Train upstream-style I-JEPA with project datasets and sample weighting.

CUDA_VISIBLE_DEVICES=1 python scripts/images/train_ijepa_spatial.py --run-name uniform_seed0
tensorboard --logdir outputs/ijepa_spatial
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jepa.analysis.subspace import compute_latent_spectrum  # noqa: E402
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import (  # noqa: E402
    load_shapes3d_config,
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
from jepa.training.images.bandit_weighting import (  # noqa: E402
    DiscountedLinearThompsonSampler,
    DiscountedRewardNormalizer,
    LatentContextCache,
    LinearThompsonConfig,
    RichnessGradientSnapshot,
    batch_ras_from_parameter_gradients,
    capture_richness_gradient,
)
from jepa.training.images.ijepa_schedulers import (  # noqa: E402
    CosineWDSchedule,
    WarmupCosineSchedule,
)
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    MaskConfig,
    build_spatial_ijepa_core,
    encode_frames_pooled_batched,
    encode_samples_pooled,
    load_spatial_checkpoint,
    normalize_ijepa_images,
    sample_masks,
    save_spatial_checkpoint,
    spatial_ijepa_loss,
    spatial_ijepa_loss_with_context,
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

MODEL_NAME = "vit_tiny"
IMAGE_SIZE = 64
PATCH_SIZE = 8
LR = 0.001
START_LR = 0.0002
FINAL_LR = 0.000001
WEIGHT_DECAY = 0.04
FINAL_WEIGHT_DECAY = 0.4
EMA_START = 0.996
EMA_END = 1.0
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
    parser.add_argument("--lr", type=float, default=LR, help="Peak/reference learning rate")
    parser.add_argument("--start-lr", type=float, default=START_LR)
    parser.add_argument("--final-lr", type=float, default=FINAL_LR)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--final-weight-decay", type=float, default=FINAL_WEIGHT_DECAY)
    parser.add_argument("--ipe-scale", type=float, default=1.0)
    parser.add_argument("--ema-start", type=float, default=EMA_START)
    parser.add_argument("--ema-end", type=float, default=EMA_END)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-name",
        choices=("vit_tiny", "vit_small", "vit_base", "vit_large"),
        default=MODEL_NAME,
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--predictor-embed-dim", type=int, default=192)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-scale", type=float, nargs=2, default=(0.3, 1.0))
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.0)
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
        help="Keep image tensors on the training device to reduce CPU/GPU transfer overhead",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume model, optimizer, global step, and weighting state from this checkpoint",
    )
    parser.add_argument(
        "--weighting-method",
        choices=("uniform", "loss", "ras", "coord", "ras-thompson"),
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
        "--ras-score-granularity",
        choices=("sample", "batch"),
        default="sample",
        help="Batch mode assigns one gradient-alignment score per shuffled scoring batch.",
    )
    parser.add_argument(
        "--coordinate-importance",
        choices=("covariance", "transformation", "dynamics"),
        default="covariance",
        help="Coordinate importance estimator used by --weighting-method coord",
    )
    parser.add_argument("--coordinate-ema-beta", type=float, default=0.1)
    parser.add_argument("--coordinate-delta", type=float, default=1.0e-6)
    parser.add_argument("--bandit-discount", type=float, default=0.999)
    parser.add_argument("--bandit-prior-precision", type=float, default=1.0)
    parser.add_argument("--bandit-observation-noise", type=float, default=1.0)
    parser.add_argument("--bandit-exploration-scale", type=float, default=1.0)
    parser.add_argument("--bandit-context-ema-beta", type=float, default=0.1)
    parser.add_argument("--bandit-reward-normalization-decay", type=float, default=0.99)
    parser.add_argument("--bandit-richness-refresh-steps", type=int, default=500)
    return parser.parse_args()


def _default_run_name(seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")  # noqa: UP017
    return f"spatial_ijepa_seed{seed}_{stamp}"


def _checkpoint_metadata(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    return {
        "model_family": "upstream_ijepa",
        "model_name": args.model_name,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "embed_dim": None,
        "predictor_embed_dim": args.predictor_embed_dim,
        "predictor_depth": args.predictor_depth,
        "ema": [args.ema_start, args.ema_end],
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "data_source": f"{args.dataset}_images",
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0 or args.patch_size <= 0:
        raise ValueError("image and patch sizes must be positive")
    if args.image_size % args.patch_size:
        raise ValueError("--image-size must be divisible by --patch-size")
    if args.predictor_embed_dim <= 0 or args.predictor_depth <= 0:
        raise ValueError("predictor dimensions must be positive")
    if (
        args.predictor_embed_dim
        % {"vit_tiny": 3, "vit_small": 6, "vit_base": 12, "vit_large": 16}[args.model_name]
    ):
        raise ValueError("--predictor-embed-dim must be divisible by the encoder head count")
    if not 0.0 <= args.horizontal_flip_prob <= 1.0:
        raise ValueError("--horizontal-flip-prob must be in [0, 1]")
    if len(args.crop_scale) != 2 or not 0 < args.crop_scale[0] <= args.crop_scale[1] <= 1:
        raise ValueError("--crop-scale must satisfy 0 < min <= max <= 1")
    if not 0 <= args.ema_start <= args.ema_end <= 1:
        raise ValueError("EMA schedule must satisfy 0 <= start <= end <= 1")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.bandit_richness_refresh_steps <= 0:
        raise ValueError("--bandit-richness-refresh-steps must be positive")
    if not 0 <= args.bandit_reward_normalization_decay < 1:
        raise ValueError("--bandit-reward-normalization-decay must be in [0, 1)")


def _random_resized_crop_batch(
    images: torch.Tensor,
    *,
    output_size: int,
    scale: tuple[float, float],
    horizontal_flip_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Batched tensor implementation of upstream RandomResizedCrop and flip."""
    batch_size, channels, source_h, source_w = images.shape
    area = source_h * source_w
    crop_h = torch.full((batch_size,), source_h, dtype=torch.long)
    crop_w = torch.full((batch_size,), source_w, dtype=torch.long)
    valid = torch.zeros(batch_size, dtype=torch.bool)
    for _ in range(10):
        target_area = area * (
            scale[0] + torch.rand(batch_size, generator=generator) * (scale[1] - scale[0])
        )
        log_ratio = math.log(3 / 4) + torch.rand(batch_size, generator=generator) * (
            math.log(4 / 3) - math.log(3 / 4)
        )
        ratio = log_ratio.exp()
        proposed_w = (target_area * ratio).sqrt().round().to(torch.long)
        proposed_h = (target_area / ratio).sqrt().round().to(torch.long)
        accepted = (
            ~valid
            & (proposed_h > 0)
            & (proposed_h <= source_h)
            & (proposed_w > 0)
            & (proposed_w <= source_w)
        )
        crop_h[accepted] = proposed_h[accepted]
        crop_w[accepted] = proposed_w[accepted]
        valid |= accepted
        if bool(valid.all()):
            break

    top_random = torch.rand(batch_size, generator=generator)
    left_random = torch.rand(batch_size, generator=generator)
    top = (top_random * (source_h - crop_h + 1)).floor()
    left = (left_random * (source_w - crop_w + 1)).floor()
    theta = torch.zeros((batch_size, 2, 3), dtype=torch.float32, device=images.device)
    theta[:, 0, 0] = crop_w.to(device=images.device, dtype=torch.float32) / source_w
    theta[:, 1, 1] = crop_h.to(device=images.device, dtype=torch.float32) / source_h
    theta[:, 0, 2] = (
        2.0 * (left.to(device=images.device) + crop_w.to(device=images.device) / 2.0) / source_w
        - 1.0
    )
    theta[:, 1, 2] = (
        2.0 * (top.to(device=images.device) + crop_h.to(device=images.device) / 2.0) / source_h
        - 1.0
    )
    grid = F.affine_grid(
        theta,
        size=(batch_size, channels, output_size, output_size),
        align_corners=False,
    )
    crops = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    flip = (torch.rand(batch_size, generator=generator) < horizontal_flip_probability).to(
        images.device
    )
    return torch.where(flip[:, None, None, None], crops.flip(-1), crops)


def _init_upstream_optimizer(
    core,
    *,
    iterations_per_epoch: int,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, WarmupCosineSchedule, CosineWDSchedule]:
    param_groups = [
        {
            "params": [
                p
                for name, p in core.context_encoder.named_parameters()
                if "bias" not in name and len(p.shape) != 1
            ]
        },
        {
            "params": [
                p
                for name, p in core.predictor.named_parameters()
                if "bias" not in name and len(p.shape) != 1
            ]
        },
        {
            "params": [
                p
                for name, p in core.context_encoder.named_parameters()
                if "bias" in name or len(p.shape) == 1
            ],
            "WD_exclude": True,
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for name, p in core.predictor.named_parameters()
                if "bias" in name or len(p.shape) == 1
            ],
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = int(args.ipe_scale * args.epochs * iterations_per_epoch)
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=args.warmup_epochs * iterations_per_epoch,
        start_lr=args.start_lr,
        ref_lr=args.lr,
        final_lr=args.final_lr,
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=args.weight_decay,
        final_wd=args.final_weight_decay,
        T_max=total_steps,
    )
    return optimizer, lr_scheduler, wd_scheduler


def _ema_momentum(step: int, *, total_steps: int, start: float, end: float) -> float:
    progress = min(max(step, 0), total_steps) / max(total_steps, 1)
    return start + progress * (end - start)


@torch.no_grad()
def _ema_update(target: torch.nn.Module, online: torch.nn.Module, momentum: float) -> None:
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters(), strict=True
    ):
        target_parameter.data.mul_(momentum).add_(
            online_parameter.detach().data, alpha=1.0 - momentum
        )


def _weighting_checkpoint_state(
    weighting_state: SpatialWeightingState,
    ref_indices: torch.Tensor,
    scaler=None,
    *,
    bandit_sampler: DiscountedLinearThompsonSampler | None = None,
    bandit_cache: LatentContextCache | None = None,
    bandit_reward_normalizer: DiscountedRewardNormalizer | None = None,
    richness_snapshot: RichnessGradientSnapshot | None = None,
) -> dict[str, object]:
    state: dict[str, object] = {
        "weighting_memory": weighting_state.memory,
        "weighting_probabilities": weighting_state.probabilities,
        "weighting_last_scores": weighting_state.last_scores,
        "weighting_ref_indices": ref_indices,
        "coordinate_importance": weighting_state.coordinate_importance,
        "coordinate_previous_ref_latents": weighting_state.coordinate_previous_ref_latents,
    }
    if scaler is not None and scaler.is_enabled():
        state["amp_scaler"] = scaler.state_dict()
    if bandit_sampler is not None:
        state["bandit_sampler"] = bandit_sampler.state_dict()
    if bandit_cache is not None:
        state["bandit_cache"] = bandit_cache.state_dict()
    if bandit_reward_normalizer is not None:
        state["bandit_reward_normalizer"] = bandit_reward_normalizer.state_dict()
    if richness_snapshot is not None:
        state["bandit_richness_snapshot"] = {
            "gradients": tuple(gradient.detach().cpu() for gradient in richness_snapshot.gradients),
            "metadata": richness_snapshot.metadata,
            "step": richness_snapshot.step,
        }
    return state


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


def _restore_bandit_state(
    checkpoint: dict[str, object],
    *,
    sampler: DiscountedLinearThompsonSampler,
    cache: LatentContextCache,
    reward_normalizer: DiscountedRewardNormalizer,
    device: torch.device,
) -> RichnessGradientSnapshot | None:
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, dict):
        raise ValueError("resume checkpoint is missing bandit extra_state")
    sampler_state = extra_state.get("bandit_sampler")
    cache_state = extra_state.get("bandit_cache")
    normalizer_state = extra_state.get("bandit_reward_normalizer")
    if not isinstance(sampler_state, dict):
        raise ValueError("resume checkpoint has incomplete bandit sampler state")
    if not isinstance(cache_state, dict):
        raise ValueError("resume checkpoint has incomplete bandit cache state")
    if not isinstance(normalizer_state, dict):
        raise ValueError("resume checkpoint has incomplete bandit state")
    sampler.load_state_dict(sampler_state)
    cache.load_state_dict(cache_state)
    reward_normalizer.load_state_dict(normalizer_state)
    snapshot_state = extra_state.get("bandit_richness_snapshot")
    if not isinstance(snapshot_state, dict):
        return None
    gradients = snapshot_state.get("gradients")
    metadata = snapshot_state.get("metadata")
    if not isinstance(gradients, tuple | list) or not isinstance(metadata, dict):
        raise ValueError("resume checkpoint has invalid richness snapshot")
    return RichnessGradientSnapshot(
        gradients=tuple(torch.as_tensor(value).to(device) for value in gradients),
        metadata={str(key): float(value) for key, value in metadata.items()},
        step=int(snapshot_state["step"]),
    )


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
        context_masks, target_masks = sample_masks(
            grid,
            grid,
            mask_config,
            generator,
            batch_size=batch.shape[0],
        )
        loss = spatial_ijepa_loss(core, batch, context_masks, target_masks)
        total_loss += float(loss.item()) * batch.shape[0]
        total_examples += batch.shape[0]
    return total_loss / total_examples


@torch.no_grad()
def _encode_images_pooled(
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
    _validate_args(args)
    run_name = args.run_name or _default_run_name(args.seed)
    run_dir = Path(args.output_root).expanduser().resolve() / run_name
    checkpoint_dir = run_dir / "network"
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
        ras_score_granularity=args.ras_score_granularity,
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
                "model_family": "upstream_ijepa",
                "model_name": args.model_name,
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "predictor_embed_dim": args.predictor_embed_dim,
                "predictor_depth": args.predictor_depth,
                "learning_rate": args.lr,
                "start_learning_rate": args.start_lr,
                "final_learning_rate": args.final_lr,
                "warmup_epochs": args.warmup_epochs,
                "weight_decay": args.weight_decay,
                "final_weight_decay": args.final_weight_decay,
                "ipe_scale": args.ipe_scale,
                "ema": [args.ema_start, args.ema_end],
                "bfloat16": args.bfloat16,
                "crop_scale": list(args.crop_scale),
                "horizontal_flip_probability": args.horizontal_flip_prob,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "dataset": args.dataset,
                "data_source": f"{args.dataset}_images",
                "eval_every_epochs": args.eval_every_epochs,
                "checkpoint_every_epochs": args.checkpoint_every_epochs,
                "checkpoint_every_steps": args.checkpoint_every_steps,
                "log_every_steps": args.log_every_steps,
                "resident_device_data": args.resident_device_data,
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
                    "ras_score_granularity": weighting_config.ras_score_granularity,
                    "coordinate_importance": weighting_config.coordinate_importance,
                    "coordinate_ema_beta": weighting_config.coordinate_ema_beta,
                    "coordinate_delta": weighting_config.coordinate_delta,
                    "bandit": {
                        "discount": args.bandit_discount,
                        "prior_precision": args.bandit_prior_precision,
                        "observation_noise": args.bandit_observation_noise,
                        "exploration_scale": args.bandit_exploration_scale,
                        "context_ema_beta": args.bandit_context_ema_beta,
                        "reward_normalization_decay": args.bandit_reward_normalization_decay,
                        "richness_refresh_steps": args.bandit_richness_refresh_steps,
                    },
                },
            },
        }
    )
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_images = normalize_ijepa_images(datasets.train.images, inplace=True)
    test_images = normalize_ijepa_images(datasets.test.images, inplace=True)
    transform_pair_images: tuple[torch.Tensor, torch.Tensor] | None = None
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
        transform_pair_images = (
            normalize_ijepa_images(pairs.same_entity_x1, inplace=True),
            normalize_ijepa_images(pairs.same_entity_x2, inplace=True),
        )
        del pairs
    del datasets
    if train_images.shape[-2:] != (args.image_size, args.image_size):
        raise ValueError(
            f"dataset images are {tuple(train_images.shape[-2:])}, "
            f"but --image-size is {args.image_size}"
        )
    grid = args.image_size // args.patch_size
    num_patches = grid * grid
    if args.resident_device_data:
        train_images = train_images.to(device)
        test_images = test_images.to(device)
        if transform_pair_images is not None:
            transform_pair_images = (
                transform_pair_images[0].to(device),
                transform_pair_images[1].to(device),
            )
    print(
        f"run_dir={run_dir}\n"
        f"dataset={args.dataset} model={args.model_name} device={device} "
        f"train_images={train_images.shape[0]} num_patches={num_patches}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    core = build_spatial_ijepa_core(
        args.model_name,
        image_size=args.image_size,
        patch_size=args.patch_size,
        predictor_embed_dim=args.predictor_embed_dim,
        predictor_depth=args.predictor_depth,
    )
    core.context_encoder.to(device)
    core.predictor.to(device)
    core.target_encoder.to(device)
    mask_config = MaskConfig()

    n = train_images.shape[0]
    iterations_per_epoch = max(n // args.batch_size, 1)
    optimizer, lr_scheduler, wd_scheduler = _init_upstream_optimizer(
        core,
        iterations_per_epoch=iterations_per_epoch,
        args=args,
    )
    total_schedule_steps = int(args.ipe_scale * args.epochs * iterations_per_epoch)
    use_bfloat16 = args.bfloat16 and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_bfloat16)
    except (AttributeError, TypeError):  # PyTorch versions used by older DataSphere images.
        scaler = torch.cuda.amp.GradScaler(enabled=use_bfloat16)
    start = time.time()
    global_step = 0
    metadata = _checkpoint_metadata(args, run_dir)
    metadata["embed_dim"] = core.embed_dim
    weighting_state = init_spatial_weighting(n)
    weighting_ref_indices = select_reference_indices(
        n,
        ref_size=weighting_config.ref_size,
        seed=derive_seed(args.seed, "weighting-ref"),
    )
    bandit_sampler: DiscountedLinearThompsonSampler | None = None
    bandit_cache: LatentContextCache | None = None
    bandit_reward_normalizer: DiscountedRewardNormalizer | None = None
    bandit_encoder_parameters: tuple[torch.nn.Parameter, ...] = ()
    richness_snapshot: RichnessGradientSnapshot | None = None
    if weighting_config.method == "ras-thompson":
        bandit_sampler = DiscountedLinearThompsonSampler(
            LinearThompsonConfig(
                context_dim=core.embed_dim,
                prior_precision=args.bandit_prior_precision,
                observation_noise=args.bandit_observation_noise,
                discount=args.bandit_discount,
                exploration_scale=args.bandit_exploration_scale,
                temperature=weighting_config.temperature,
                uniform_mix=weighting_config.uniform_mix,
            )
        )
        bandit_cache = LatentContextCache(n, core.embed_dim, ema_beta=args.bandit_context_ema_beta)
        bandit_reward_normalizer = DiscountedRewardNormalizer(
            decay=args.bandit_reward_normalization_decay
        )
        bandit_encoder_parameters = tuple(
            parameter for parameter in core.context_encoder.parameters() if parameter.requires_grad
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
        lr_scheduler._step = float(global_step)
        wd_scheduler._step = float(global_step)
        checkpoint_extra = checkpoint.get("extra_state")
        if isinstance(checkpoint_extra, dict) and isinstance(
            checkpoint_extra.get("amp_scaler"), dict
        ):
            scaler.load_state_dict(checkpoint_extra["amp_scaler"])
        weighting_state, weighting_ref_indices = _restore_weighting_state(
            checkpoint,
            num_frames=n,
        )
        if bandit_sampler is not None:
            assert bandit_cache is not None
            assert bandit_reward_normalizer is not None
            richness_snapshot = _restore_bandit_state(
                checkpoint,
                sampler=bandit_sampler,
                cache=bandit_cache,
                reward_normalizer=bandit_reward_normalizer,
                device=device,
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
            core.target_encoder.train()
            num_draws = n if n < args.batch_size else iterations_per_epoch * args.batch_size
            if bandit_sampler is not None and epoch > weighting_config.warmup_epochs:
                assert bandit_cache is not None
                policy_generator = torch.Generator().manual_seed(
                    derive_seed(args.seed, "bandit-policy", epoch)
                )
                policy_scores, policy_probabilities = bandit_sampler.draw_policy(
                    bandit_cache.contexts,
                    generator=policy_generator,
                )
                weighting_state = SpatialWeightingState(
                    memory=weighting_state.memory,
                    probabilities=policy_probabilities.detach().cpu().to(torch.float64),
                    last_scores=policy_scores.detach().cpu().to(torch.float64),
                    coordinate_importance=weighting_state.coordinate_importance,
                    coordinate_previous_ref_latents=(
                        weighting_state.coordinate_previous_ref_latents
                    ),
                )
                logger.log(
                    step=global_step,
                    epoch=epoch,
                    event="bandit_policy",
                    scalars={
                        **weighting_diagnostics(weighting_state),
                        **bandit_sampler.diagnostics(),
                        "bandit/cache_coverage": bandit_cache.coverage,
                    },
                    histograms={
                        "hist/weighting_scores": weighting_state.last_scores,
                        "hist/weighting_probabilities": weighting_state.probabilities,
                    },
                )
            if weighting_config.method != "uniform" and epoch > weighting_config.warmup_epochs:
                order = sample_frame_indices(
                    weighting_state,
                    num_draws=num_draws,
                    seed=derive_seed(args.seed, "weighted-order", epoch),
                )
            else:
                order = torch.randperm(
                    n,
                    generator=torch.Generator().manual_seed(derive_seed(args.seed, "order", epoch)),
                )[:num_draws]
            if args.resident_device_data:
                order = order.to(device)
            batch_chunks = order.split(args.batch_size)
            mask_generator = torch.Generator().manual_seed(derive_seed(args.seed, "masks", epoch))
            transform_generator = torch.Generator().manual_seed(
                derive_seed(args.seed, "transforms", epoch)
            )
            epoch_loss, batches = 0.0, 0
            epoch_bandit_rewards: list[float] = []
            epoch_bandit_prediction_errors: list[float] = []
            epoch_start = time.time()
            for indices in batch_chunks:
                global_step += 1
                batch = train_images[indices].to(device)
                batch = _random_resized_crop_batch(
                    batch,
                    output_size=args.image_size,
                    scale=tuple(args.crop_scale),
                    horizontal_flip_probability=args.horizontal_flip_prob,
                    generator=transform_generator,
                )
                context_masks, target_masks = sample_masks(
                    grid,
                    grid,
                    mask_config,
                    mask_generator,
                    batch_size=batch.shape[0],
                )
                current_lr = lr_scheduler.step()
                current_wd = wd_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if bandit_sampler is not None and (
                    richness_snapshot is None
                    or global_step - richness_snapshot.step >= args.bandit_richness_refresh_steps
                ):
                    reference_images = train_images[weighting_ref_indices].to(device)
                    richness_snapshot = capture_richness_gradient(
                        core,
                        reference_images,
                        functional=weighting_config.richness_functional,
                        delta=weighting_config.richness_delta,
                        trace_target=weighting_config.richness_trace_target,
                        trace_beta=weighting_config.richness_trace_beta,
                        step=global_step,
                    )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bfloat16,
                ):
                    if bandit_sampler is not None:
                        loss, batch_contexts = spatial_ijepa_loss_with_context(
                            core, batch, context_masks, target_masks
                        )
                    else:
                        loss = spatial_ijepa_loss(core, batch, context_masks, target_masks)
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                bandit_scalars: dict[str, float] = {}
                if bandit_sampler is not None:
                    assert bandit_cache is not None
                    assert bandit_reward_normalizer is not None
                    assert richness_snapshot is not None
                    raw_reward = float(
                        batch_ras_from_parameter_gradients(
                            bandit_encoder_parameters, richness_snapshot
                        ).item()
                    )
                    contexts_for_bandit = F.layer_norm(
                        batch_contexts.detach().float(),
                        (batch_contexts.shape[-1],),
                    ).cpu()
                    normalized_reward = bandit_reward_normalizer.update(raw_reward)
                    predicted_reward = bandit_sampler.predict_batch_reward(contexts_for_bandit)
                    bandit_sampler.update_batch(contexts_for_bandit, normalized_reward)
                    bandit_cache.update(
                        indices.detach().cpu(),
                        contexts_for_bandit,
                        step=global_step,
                    )
                    prediction_error = normalized_reward - predicted_reward
                    epoch_bandit_rewards.append(normalized_reward)
                    epoch_bandit_prediction_errors.append(prediction_error)
                    bandit_scalars = {
                        "bandit/raw_reward": raw_reward,
                        "bandit/normalized_reward": normalized_reward,
                        "bandit/predicted_reward": predicted_reward,
                        "bandit/prediction_error": prediction_error,
                        "bandit/cache_coverage": bandit_cache.coverage,
                        **richness_snapshot.metadata,
                    }
                if use_bfloat16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                momentum = _ema_momentum(
                    global_step - 1,
                    total_steps=total_schedule_steps,
                    start=args.ema_start,
                    end=args.ema_end,
                )
                if core.policy.ema_enabled:
                    _ema_update(core.target_encoder, core.context_encoder, momentum)
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
                            "train/lr": current_lr,
                            "train/weight_decay": current_wd,
                            "train/ema_momentum": momentum,
                            "train/epoch_fraction": epoch + batches / max(len(batch_chunks), 1),
                            **bandit_scalars,
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
                            scaler,
                            bandit_sampler=bandit_sampler,
                            bandit_cache=bandit_cache,
                            bandit_reward_normalizer=bandit_reward_normalizer,
                            richness_snapshot=richness_snapshot,
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
                            scaler,
                            bandit_sampler=bandit_sampler,
                            bandit_cache=bandit_cache,
                            bandit_reward_normalizer=bandit_reward_normalizer,
                            richness_snapshot=richness_snapshot,
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
                    **(
                        {
                            "bandit/reward_mean": float(
                                torch.tensor(epoch_bandit_rewards).mean().item()
                            ),
                            "bandit/reward_std": float(
                                torch.tensor(epoch_bandit_rewards).std(unbiased=False).item()
                            ),
                            "bandit/prediction_rmse": float(
                                torch.tensor(epoch_bandit_prediction_errors)
                                .square()
                                .mean()
                                .sqrt()
                                .item()
                            ),
                            **bandit_sampler.diagnostics(),
                            "bandit/cache_coverage": bandit_cache.coverage,
                        }
                        if bandit_sampler is not None
                        and bandit_cache is not None
                        and epoch_bandit_rewards
                        else {}
                    ),
                },
            )
            if should_update_weights(epoch, weighting_config):
                score_batch_size = weighting_config.score_batch_size or args.batch_size
                if weighting_config.method == "loss":
                    scores = score_frames_by_loss(
                        core,
                        train_images,
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
                        train_images,
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
                        score_granularity=weighting_config.ras_score_granularity,
                    )
                elif weighting_config.method == "coord":
                    (
                        scores,
                        score_metadata,
                        coordinate_importance,
                        coordinate_previous_ref_latents,
                    ) = score_frames_by_coordinate_importance(
                        core,
                        train_images,
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
                        transform_pair_images=transform_pair_images,
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

            if epoch % args.eval_every_epochs == 0 or epoch == 1:
                test_loss = _evaluate_spatial_loss(
                    core,
                    test_images,
                    grid=grid,
                    mask_config=mask_config,
                    batch_size=args.batch_size,
                    seed=derive_seed(args.seed, "eval-loss", epoch),
                    device=device,
                )
                if args.resident_device_data:
                    test_z = _encode_images_pooled(
                        core,
                        test_images,
                        batch_size=args.batch_size,
                        device=device,
                    )
                else:
                    test_z = encode_frames_pooled_batched(
                        core,
                        test_images,
                        patch_size=args.patch_size,
                        batch_size=args.batch_size,
                    )
                spectrum = compute_latent_spectrum(test_z.reshape(-1, core.embed_dim))
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
            if (
                args.checkpoint_every_epochs > 0 and epoch % args.checkpoint_every_epochs == 0
            ) or epoch == args.epochs:
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
                        scaler,
                        bandit_sampler=bandit_sampler,
                        bandit_cache=bandit_cache,
                        bandit_reward_normalizer=bandit_reward_normalizer,
                        richness_snapshot=richness_snapshot,
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
                        scaler,
                        bandit_sampler=bandit_sampler,
                        bandit_cache=bandit_cache,
                        bandit_reward_normalizer=bandit_reward_normalizer,
                        richness_snapshot=richness_snapshot,
                    ),
                )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
