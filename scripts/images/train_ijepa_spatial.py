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
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from jepa.analysis.subspace import compute_latent_spectrum  # noqa: E402
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import (  # noqa: E402
    load_shapes3d_config,
    shapes3d_config_to_dict,
)
from jepa.data.images.shapes3d import build_shapes3d_static_dataset_splits  # noqa: E402
from jepa.models.patches import patchify  # noqa: E402
from jepa.training.core import ema_update  # noqa: E402
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    MaskConfig,
    build_spatial_ijepa_core,
    encode_frames_pooled,
    sample_masks,
    save_spatial_checkpoint,
    spatial_ijepa_loss,
)
from jepa.training.images.spatial_curriculum import (  # noqa: E402
    SpatialWeightingConfig,
    SpatialWeightingState,
    init_spatial_weighting,
    sample_frame_indices,
    score_frames_by_loss,
    score_frames_by_ras,
    select_reference_indices,
    should_update_weights,
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
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--weighting-method", choices=("uniform", "loss", "ras"), default="uniform")
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
    parser.add_argument("--weighting-richness-delta", type=float, default=1.0e-4)
    return parser.parse_args()


def _default_run_name(seed: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"spatial_ijepa_cnn_seed{seed}_{stamp}"


def _checkpoint_metadata(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    return {
        "architecture": ARCHITECTURE,
        "patch_size": PATCH_SIZE,
        "patch_latent_dim": PATCH_LATENT_DIM,
        "ema_decay": EMA_DECAY,
        "run_dir": str(run_dir),
        "data_source": "static_shapes3d_images",
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
    }


def main() -> None:
    args = _parse_args()
    run_name = args.run_name or _default_run_name(args.seed)
    run_dir = Path(args.output_root).expanduser().resolve() / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
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
        richness_delta=args.weighting_richness_delta,
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
    logger.write_config(
        {
            "config": shapes3d_config_to_dict(config),
            "spatial": {
                "architecture": ARCHITECTURE,
                "patch_size": PATCH_SIZE,
                "patch_latent_dim": PATCH_LATENT_DIM,
                "learning_rate": args.lr,
                "ema_decay": EMA_DECAY,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "data_source": "static_shapes3d_images",
                "eval_every_epochs": args.eval_every_epochs,
                "checkpoint_every_epochs": args.checkpoint_every_epochs,
                "checkpoint_every_steps": args.checkpoint_every_steps,
                "log_every_steps": args.log_every_steps,
                "weighting": {
                    "method": weighting_config.method,
                    "warmup_epochs": weighting_config.warmup_epochs,
                    "update_every_epochs": weighting_config.update_every_epochs,
                    "temperature": weighting_config.temperature,
                    "replay_beta": weighting_config.replay_beta,
                    "uniform_mix": weighting_config.uniform_mix,
                    "score_batch_size": weighting_config.score_batch_size,
                    "ref_size": weighting_config.ref_size,
                    "richness_delta": weighting_config.richness_delta,
                },
            },
        }
    )
    datasets = build_shapes3d_static_dataset_splits(config.data)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_frames = datasets.train.images
    test_frames = datasets.test.images
    train_patches = patchify(train_frames, PATCH_SIZE)
    grid = 64 // PATCH_SIZE
    num_patches = train_patches.shape[1]
    patch_dim = train_patches.shape[2]
    print(
        f"run_dir={run_dir}\n"
        f"device={device} train_images={train_frames.shape[0]} "
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
    start = time.time()
    global_step = 0
    metadata = _checkpoint_metadata(args, run_dir)
    weighting_state = init_spatial_weighting(n)
    weighting_ref_indices = select_reference_indices(
        n,
        ref_size=weighting_config.ref_size,
        seed=derive_seed(args.seed, "weighting-ref"),
    )
    try:
        for epoch in range(1, args.epochs + 1):
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
                        richness_delta=weighting_config.richness_delta,
                    )
                else:
                    raise ValueError(f"unsupported weighting method: {weighting_config.method}")
                weighting_state = update_spatial_weights(weighting_state, scores, weighting_config)
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
                test_z = encode_frames_pooled(core, test_frames, patch_size=PATCH_SIZE)
                spectrum = compute_latent_spectrum(test_z.reshape(-1, PATCH_LATENT_DIM))
                scalars = {
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
        logger.close()


if __name__ == "__main__":
    main()
