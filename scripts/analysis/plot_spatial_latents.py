#!/usr/bin/env python3
"""Visualize the frozen latents of a spatial I-JEPA checkpoint, colored by shape.

Two complementary views, both fit on train and plotted on held-out test:

  pca_*.png        top-2 directions of raw variance (unsupervised; shows what the
                   representation spends its variance on, which need not be shape)
  lda_*.png        top-2 generalized-eigenvector directions of the entity subspace --
                   the same directions Q_E is computed from, so this is a fair visual
                   audit of that metric rather than a cherry-picked projection
  pca_by_*.png     the same PCA projection, colored by each generative factor in turn
                   -- answers "if the top variance directions aren't shape, what are
                   they?" directly instead of by assertion

Usage:
    python scripts/analysis/plot_spatial_latents.py \
        --checkpoint outputs/ijepa_spatial/checkpoints/epoch_1500.pt \
        --out-dir outputs/ijepa_spatial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from jepa.analysis.plots import plot_latent_pca_scatter  # noqa: E402
from jepa.analysis.subspace import (  # noqa: E402
    compute_scatter_matrices,
    solve_generalized_eigenproblem,
)
from jepa.configs.images.shapes3d import (  # noqa: E402
    SHAPES3D_CONTEXT_FACTORS,
    SHAPES3D_ENTITY_NAMES,
    load_shapes3d_config,
)
from jepa.data.images.shapes3d import build_shapes3d_dataset_splits  # noqa: E402
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    build_spatial_ijepa_core,
    encode_frames_pooled,
    load_spatial_checkpoint,
)

PATCH_SIZE = 8
PATCH_LATENT_DIM = 16


def _pca_2d(z: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    """Project z onto the top-2 principal directions of ``reference`` (train)."""
    train = reference.detach().cpu().double().numpy()
    mean = train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
    return (z.detach().cpu().double().numpy() - mean) @ vt[:2].T


def _plot_by_continuous_factor(
    points: np.ndarray, values: np.ndarray, factor_name: str, out_path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    # hues are circular, so a cyclic colormap avoids a fake seam at 0/1
    cmap = "hsv" if factor_name.endswith("_hue") else "viridis"
    scatter = axis.scatter(points[:, 0], points[:, 1], c=values, s=8, alpha=0.7, cmap=cmap)
    fig.colorbar(scatter, ax=axis, label=factor_name)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title(f"PCA of frozen latents, colored by {factor_name}")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default="configs/images/shapes3d/quick.yaml")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=6000,
        help="Subsample frames before encoding; a scatter plot saturates long before "
        "the full split, and encoding every frame on CPU is slow.",
    )
    args = parser.parse_args()

    config = load_shapes3d_config(
        args.config,
        overrides={"data.num_train_trajectories": "1000", "training.device": "cpu"},
    )
    datasets = build_shapes3d_dataset_splits(config.data)
    train_frames = datasets.train.frames.reshape(-1, 3, 64, 64)[: args.max_frames]
    test_frames = datasets.test.frames.reshape(-1, 3, 64, 64)[: args.max_frames]
    train_entity = datasets.train.entities.reshape(-1)[: args.max_frames]
    test_entity = datasets.test.entities.reshape(-1)[: args.max_frames]
    test_context = datasets.test.contexts.reshape(-1, config.data.context_dim)[: args.max_frames]

    checkpoint = load_spatial_checkpoint(args.checkpoint)
    epoch = checkpoint["epoch"]
    core = build_spatial_ijepa_core(
        "cnn",
        patch_dim=3 * PATCH_SIZE * PATCH_SIZE,
        patch_latent_dim=PATCH_LATENT_DIM,
        num_patches=(64 // PATCH_SIZE) ** 2,
    )
    core.context_encoder.load_state_dict(checkpoint["context_encoder"])
    core.predictor.load_state_dict(checkpoint["predictor"])
    core.target_encoder.load_state_dict(checkpoint["target_encoder"])
    core.context_encoder.eval()

    train_z = encode_frames_pooled(core, train_frames, patch_size=PATCH_SIZE).cpu()
    test_z = encode_frames_pooled(core, test_frames, patch_size=PATCH_SIZE).cpu()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = list(SHAPES3D_ENTITY_NAMES)

    pca_points = _pca_2d(test_z, train_z)
    pca_path = args.out_dir / f"pca_epoch{epoch}.png"
    plot_latent_pca_scatter(pca_points, test_entity.tolist(), names, pca_path)

    # same projection, one plot per context factor
    for index, factor_name in enumerate(SHAPES3D_CONTEXT_FACTORS):
        _plot_by_continuous_factor(
            pca_points,
            test_context[:, index].numpy(),
            factor_name,
            args.out_dir / f"pca_by_{factor_name}_epoch{epoch}.png",
        )

    scatter = compute_scatter_matrices(train_z, train_entity, config.data.num_entities)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=config.evaluation.covariance_epsilon)
    directions = torch.as_tensor(eigen.eigenvectors[:, :2])
    lda_2d = (test_z.to(torch.float64) @ directions).cpu().numpy()
    lda_path = args.out_dir / f"lda_epoch{epoch}.png"
    plot_latent_pca_scatter(lda_2d, test_entity.tolist(), names, lda_path)

    print(f"wrote {pca_path}\nwrote {lda_path}")
    print(f"wrote pca_by_<factor>_epoch{epoch}.png for {len(SHAPES3D_CONTEXT_FACTORS)} factors")


if __name__ == "__main__":
    main()
