#!/usr/bin/env python3
"""Post-hoc report over already-trained cells from the Shapes3D bottleneck grids
(``_investigate_shapes3d_pairing_controls[_full|_cnn].py``): PCA scatter of
frozen latents colored by entity, plus a histogram-based mutual-information
check I(top eigenvector; entity) as a nonlinear complement to the linear
probe/Q_E metrics already computed by those scripts.

Reads existing ``checkpoint.pt`` files from a grid's output root -- never
retrains anything (safe to run alongside a still-running grid; only reads
cells that have already finished).

Usage:
    python scripts/analysis/report_shapes3d_subspace.py \
        --grid-root /tmp/shapes3d_bottleneck_grid_full \
        --architecture linear \
        --d-z 8 32 64 \
        --pairings temporal shuffled shuffled_same_entity \
        --seeds 0 1 2 \
        --out-dir outputs/shapes3d_subspace_report/linear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jepa.analysis.plots import plot_latent_pca_scatter  # noqa: E402
from jepa.analysis.subspace import (  # noqa: E402
    compute_scatter_matrices,
    entity_label_entropy,
    mutual_information_entity,
    solve_generalized_eigenproblem,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import (  # noqa: E402
    SHAPES3D_ENTITY_NAMES,
    Shapes3DExperimentConfig,
    load_shapes3d_config,
)
from jepa.data.images.shapes3d import (  # noqa: E402
    Shapes3DEntityContextTrajectoryDataset,
    Shapes3DSource,
)
from jepa.models.encoders import build_model_pair  # noqa: E402
from jepa.training.images.shapes3d import build_shapes3d_run_id  # noqa: E402

PAIRING_LABELS = {
    "temporal": "temporal",
    "shuffled": "shuffled",
    "shuffled_same_entity": "oracle_same_entity",
}


def _pca_2d(z: torch.Tensor) -> np.ndarray:
    matrix = z.detach().cpu().double().numpy()
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _load_cell_config(
    grid_root: str,
    architecture: str,
    d_z: int,
    pairing: str,
    seed: int,
    *,
    batch_size: int | None = None,
) -> Shapes3DExperimentConfig:
    overrides = {
        "output.root": grid_root,
        "output.overwrite": "off",
        "model.architecture": architecture,
        "model.kind": "standard",
        "model.latent_dim": str(d_z),
        "training.seed": str(seed),
        "training.target_pairing": pairing,
    }
    if batch_size is not None:
        # The run_id hash includes training.batch_size, so reconstructing the
        # config used by a grid trained before a later batch_size change to
        # configs/images/shapes3d/full.yaml requires overriding it back.
        overrides["training.batch_size"] = str(batch_size)
    return load_shapes3d_config("configs/images/shapes3d/full.yaml", overrides=overrides)


def _load_encoder(config: Shapes3DExperimentConfig, run_dir: Path) -> torch.nn.Module:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    encoder, _ = build_model_pair(
        config.model.architecture,
        input_dim=config.data.observation_dim,
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
    )
    encoder.load_state_dict(checkpoint["context_encoder"])
    encoder.eval()
    return encoder


def process_cell(
    grid_root: str,
    architecture: str,
    d_z: int,
    pairing: str,
    seed: int,
    out_dir: Path,
    source,
    *,
    batch_size: int | None = None,
) -> dict | None:
    config = _load_cell_config(grid_root, architecture, d_z, pairing, seed, batch_size=batch_size)
    seed_label = derive_seed(seed, "dz", d_z, "pairing", pairing)
    run_id = build_shapes3d_run_id(config, seed_label)
    run_dir = Path(grid_root) / run_id
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        return None

    encoder = _load_encoder(config, run_dir)
    train_ds = Shapes3DEntityContextTrajectoryDataset(config.data, "train", source=source)
    test_ds = Shapes3DEntityContextTrajectoryDataset(config.data, "test", source=source)
    obs_dim = config.data.observation_dim
    with torch.no_grad():
        train_z = encoder(train_ds.observations.reshape(-1, obs_dim))
        test_z = encoder(test_ds.observations.reshape(-1, obs_dim))
    train_entity = train_ds.entities.reshape(-1)
    test_entity = test_ds.entities.reshape(-1)

    # Fit the LDA subspace on train only, then evaluate/visualize on held-out test --
    # matching the official Q_E metric's train/eval split (_investigate_shapes3d_
    # pairing_controls.py). Fitting and coloring the same test set would report an
    # in-sample best case, not a genuine measurement of what the encoder captures.
    scatter = compute_scatter_matrices(train_z, train_entity, config.data.num_entities)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=config.evaluation.covariance_epsilon)
    # 1-D coordinate along the single strongest LDA direction (top eigenvector),
    # not the k-dim subspace projection used elsewhere -- MI needs a scalar.
    top_1d = test_z.to(torch.float64) @ torch.as_tensor(eigen.eigenvectors[:, 0])

    mi = mutual_information_entity(top_1d, test_entity, config.data.num_entities)
    h_entity = entity_label_entropy(test_entity, config.data.num_entities)

    label = PAIRING_LABELS.get(pairing, pairing)
    png_path = out_dir / f"pca_dz{d_z}_{label}_seed{seed}.png"
    plot_latent_pca_scatter(
        _pca_2d(test_z), test_entity.tolist(), list(SHAPES3D_ENTITY_NAMES), png_path
    )

    # A second, complementary view: test points' coordinates along the top-2
    # train-fit LDA directions (the same directions Q_E is computed from), rather
    # than the top-2 unsupervised-variance directions above. Still
    # train-fit/test-evaluated, so this is a fair visual audit of what Q_E
    # measures, not a cherry-picked view. (entity_subspace_projection/project give
    # a same-dimensional projector for the Q_E pipeline's D_same/D_diff distances;
    # here we want raw 2-D coordinates for plotting, so index eigenvectors directly.)
    k = min(2, eigen.eigenvectors.shape[1])
    top_k_directions = torch.as_tensor(eigen.eigenvectors[:, :k])
    test_lda_2d = (test_z.to(torch.float64) @ top_k_directions).cpu().numpy()
    lda_png_path = out_dir / f"lda_dz{d_z}_{label}_seed{seed}.png"
    plot_latent_pca_scatter(
        test_lda_2d, test_entity.tolist(), list(SHAPES3D_ENTITY_NAMES), lda_png_path
    )

    return {
        "architecture": architecture,
        "d_z": d_z,
        "condition": label,
        "seed": seed,
        "mutual_information_nats": mi,
        "entity_entropy_nats": h_entity,
        "mi_ratio": mi / h_entity if h_entity > 0 else float("nan"),
        "pca_png": str(png_path),
        "lda_png": str(lda_png_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", required=True)
    parser.add_argument("--architecture", required=True, choices=("linear", "nonlinear", "cnn"))
    parser.add_argument("--d-z", type=int, nargs="+", required=True)
    parser.add_argument(
        "--pairings",
        nargs="+",
        default=["temporal", "shuffled", "shuffled_same_entity"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size to reconstruct the exact config a grid was "
        "trained with, if configs/images/shapes3d/full.yaml's default has since changed "
        "(the run_id hash includes batch_size).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = load_shapes3d_config("configs/images/shapes3d/full.yaml").data.h5_path
    source = Shapes3DSource(h5_path)

    records = []
    skipped = []
    for d_z in args.d_z:
        for pairing in args.pairings:
            for seed in args.seeds:
                record = process_cell(
                    args.grid_root,
                    args.architecture,
                    d_z,
                    pairing,
                    seed,
                    out_dir,
                    source,
                    batch_size=args.batch_size,
                )
                if record is None:
                    skipped.append((d_z, pairing, seed))
                else:
                    records.append(record)
                    print(
                        f"d_z={d_z} pairing={record['condition']} seed={seed} "
                        f"MI={record['mutual_information_nats']:.4f} "
                        f"(H(E)={record['entity_entropy_nats']:.4f}, "
                        f"ratio={record['mi_ratio']:.3f})",
                        flush=True,
                    )

    with (out_dir / "subspace_diagnostics.json").open("w") as stream:
        json.dump(records, stream, indent=2)
    print(f"\nwrote {len(records)} records to {out_dir / 'subspace_diagnostics.json'}")
    if skipped:
        print(f"skipped {len(skipped)} not-yet-finished cells: {skipped}")


if __name__ == "__main__":
    main()
