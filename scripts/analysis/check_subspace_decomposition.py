#!/usr/bin/env python3
"""Does the frozen latent decompose into an entity subspace and a context complement?

Answers the two-component hypothesis directly: fit the LDA entity subspace on
train, then probe *both* it and its orthogonal complement for every generative
factor. A genuine decomposition looks like

    entity subspace   -> shape readable, context NOT readable
    complement        -> context readable, shape NOT readable

Context is probed by **classification**, not R^2. The factors are discrete
indices (10/10/10/8/15 values), and three of them (the hues) are points on a
colour wheel: 0.9 and 0.0 are neighbours, but a linear regression scores them
as maximally distant, so R^2 systematically understates hue information. Class
accuracy has no such problem and is directly comparable to the entity probe.

Usage:
    python scripts/analysis/check_subspace_decomposition.py \
        --checkpoint outputs/ijepa_spatial/checkpoints/epoch_1500.pt \
        --out-dir outputs/ijepa_spatial
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jepa.analysis.subspace import (  # noqa: E402
    classifier_accuracy,
    compute_scatter_matrices,
    entity_subspace_projection,
    fit_entity_classifier,
    project,
    solve_generalized_eigenproblem,
)
from jepa.configs.images.shapes3d import (  # noqa: E402
    SHAPES3D_CONTEXT_FACTORS,
    SHAPES3D_FACTOR_SIZES,
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


def _context_class_labels(contexts: torch.Tensor, factor_index: int) -> torch.Tensor:
    """Undo the [0, 1] normalisation back into the original class index."""
    size = SHAPES3D_FACTOR_SIZES[SHAPES3D_CONTEXT_FACTORS[factor_index]]
    return (contexts[:, factor_index] * (size - 1)).round().long()


def _probe_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    ridge: float,
) -> float:
    classifier = fit_entity_classifier(train_features, train_labels, num_classes, ridge=ridge)
    return classifier_accuracy(classifier, test_features, test_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default="configs/images/shapes3d/quick.yaml")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=6000)
    parser.add_argument("--subspace-dim", type=int, default=2)
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Score an untrained encoder instead, to establish the floor.",
    )
    args = parser.parse_args()

    config = load_shapes3d_config(
        args.config,
        overrides={"data.num_train_trajectories": "1000", "training.device": "cpu"},
    )
    datasets = build_shapes3d_dataset_splits(config.data)
    n = args.max_frames
    train_frames = datasets.train.frames.reshape(-1, 3, 64, 64)[:n]
    test_frames = datasets.test.frames.reshape(-1, 3, 64, 64)[:n]
    train_entity = datasets.train.entities.reshape(-1)[:n]
    test_entity = datasets.test.entities.reshape(-1)[:n]
    context_dim = config.data.context_dim
    train_context = datasets.train.contexts.reshape(-1, context_dim)[:n]
    test_context = datasets.test.contexts.reshape(-1, context_dim)[:n]

    core = build_spatial_ijepa_core(
        "cnn",
        patch_dim=3 * PATCH_SIZE * PATCH_SIZE,
        patch_latent_dim=PATCH_LATENT_DIM,
        num_patches=(64 // PATCH_SIZE) ** 2,
    )
    if args.random_baseline:
        epoch = 0
        torch.manual_seed(0)
    else:
        checkpoint = load_spatial_checkpoint(args.checkpoint)
        epoch = checkpoint["epoch"]
        core.context_encoder.load_state_dict(checkpoint["context_encoder"])
    core.context_encoder.eval()

    train_z = encode_frames_pooled(core, train_frames, patch_size=PATCH_SIZE).cpu()
    test_z = encode_frames_pooled(core, test_frames, patch_size=PATCH_SIZE).cpu()

    # entity subspace and its orthogonal complement, both fit on train only
    scatter = compute_scatter_matrices(train_z, train_entity, config.data.num_entities)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=config.evaluation.covariance_epsilon)
    projection = entity_subspace_projection(eigen.eigenvectors, args.subspace_dim)
    complement = np.eye(PATCH_LATENT_DIM) - projection

    views = {
        "full": (train_z, test_z),
        "entity_subspace": (project(train_z, projection), project(test_z, projection)),
        "complement": (project(train_z, complement), project(test_z, complement)),
    }
    ridge = config.evaluation.probe_ridge

    record: dict = {"epoch": epoch, "subspace_dim": args.subspace_dim, "views": {}}
    for view_name, (train_features, test_features) in views.items():
        shape_acc = _probe_accuracy(
            train_features, train_entity, test_features, test_entity,
            config.data.num_entities, ridge,
        )
        per_factor = {}
        for index, factor in enumerate(SHAPES3D_CONTEXT_FACTORS):
            per_factor[factor] = _probe_accuracy(
                train_features,
                _context_class_labels(train_context, index),
                test_features,
                _context_class_labels(test_context, index),
                SHAPES3D_FACTOR_SIZES[factor],
                ridge,
            )
        record["views"][view_name] = {
            "shape_accuracy": shape_acc,
            "context_accuracy": per_factor,
            "context_accuracy_mean": float(np.mean(list(per_factor.values()))),
        }

    label = "random_baseline" if args.random_baseline else f"epoch{epoch}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"subspace_decomposition_{label}.json"
    out_path.write_text(json.dumps(record, indent=2))

    chance = {f: 1.0 / SHAPES3D_FACTOR_SIZES[f] for f in SHAPES3D_CONTEXT_FACTORS}
    print(f"\n=== {label} (subspace_dim={args.subspace_dim}) ===")
    header = f"{'view':<16} {'shape':>7} " + " ".join(
        f"{f[:9]:>9}" for f in SHAPES3D_CONTEXT_FACTORS
    )
    print(header)
    print(f"{'chance':<16} {1 / config.data.num_entities:>7.3f} " +
          " ".join(f"{chance[f]:>9.3f}" for f in SHAPES3D_CONTEXT_FACTORS))
    for view_name, values in record["views"].items():
        row = f"{view_name:<16} {values['shape_accuracy']:>7.3f} "
        row += " ".join(f"{values['context_accuracy'][f]:>9.3f}" for f in SHAPES3D_CONTEXT_FACTORS)
        print(row)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
