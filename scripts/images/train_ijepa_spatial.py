#!/usr/bin/env python3
"""Train the CNN I-JEPA-style spatial model on Shapes3D frames and watch for
representational collapse (the failure mode that killed every earlier
whole-frame variant). Purely spatial: no temporal pairing at all, so this does
not depend on the frames being related in time.

    CUDA_VISIBLE_DEVICES=1 python scripts/_train_ijepa_spatial.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from jepa.analysis.subspace import compute_latent_spectrum  # noqa: E402
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import load_shapes3d_config  # noqa: E402
from jepa.data.images.shapes3d import build_shapes3d_dataset_splits  # noqa: E402
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

ARCHITECTURE = "cnn"
PATCH_SIZE = 8
PATCH_LATENT_DIM = 16
LR = 0.001
EMA_DECAY = 0.99
BATCH_SIZE = 128
TOTAL_EPOCHS = 1500
LOG_EVERY = 10
CHECKPOINT_EVERY = 50
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "ijepa_spatial"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


def main() -> None:
    config = load_shapes3d_config(
        "configs/images/shapes3d/quick.yaml",
        overrides={"data.num_train_trajectories": "1000", "training.device": "cuda"},
    )
    datasets = build_shapes3d_dataset_splits(config.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Frames are used as independent images here -- the temporal axis is
    # flattened away, since this objective makes no use of it.
    train_frames = datasets.train.frames.reshape(-1, 3, 64, 64)
    test_frames = datasets.test.frames.reshape(-1, 3, 64, 64)
    train_patches = patchify(train_frames, PATCH_SIZE)
    grid = 64 // PATCH_SIZE
    num_patches = train_patches.shape[1]
    patch_dim = train_patches.shape[2]
    print(
        f"train_frames={train_frames.shape[0]} num_patches={num_patches} patch_dim={patch_dim}",
        flush=True,
    )

    torch.manual_seed(0)
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
    optimizer = torch.optim.Adam(params, lr=LR)
    mask_config = MaskConfig()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    n = train_patches.shape[0]
    start = time.time()
    for epoch in range(1, TOTAL_EPOCHS + 1):
        core.context_encoder.train()
        core.predictor.train()
        order = torch.randperm(
            n, generator=torch.Generator().manual_seed(derive_seed(0, "order", epoch))
        )
        mask_generator = torch.Generator().manual_seed(derive_seed(0, "masks", epoch))
        epoch_loss, batches = 0.0, 0
        for indices in order.split(BATCH_SIZE):
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
            epoch_loss += loss.item()
            batches += 1

        if epoch % LOG_EVERY == 0 or epoch == 1:
            test_z = encode_frames_pooled(core, test_frames, patch_size=PATCH_SIZE)
            spectrum = compute_latent_spectrum(test_z.reshape(-1, PATCH_LATENT_DIM))
            print(
                f"epoch={epoch:4d} loss={epoch_loss / batches:10.6f} "
                f"effective_rank={spectrum.effective_rank:6.3f} "
                f"trace_cov={spectrum.trace_covariance:9.4f} elapsed={time.time() - start:.0f}s",
                flush=True,
            )
        if epoch % CHECKPOINT_EVERY == 0 or epoch == TOTAL_EPOCHS:
            save_spatial_checkpoint(CHECKPOINT_DIR / f"epoch_{epoch:04d}.pt", core, epoch=epoch)


if __name__ == "__main__":
    main()
