#!/usr/bin/env python3
"""Does the frozen latent decompose into an entity subspace and a context complement?

Probes the LDA entity subspace and its orthogonal complement for every
generative factor, against three baselines that a naive reading would miss:

  random_subspace     k random orthonormal directions, averaged over draws.
                      LDA picks its k directions *using labels*, so the honest
                      question is not "is shape readable in the subspace?" but
                      "is it more readable than in an arbitrary k-dim slice?"
  permuted_labels     the LDA fit on shuffled shape labels. Any supervised
                      subspace search overfits somewhat; this measures how much
                      apparent structure that search invents from noise.
  untrained (--random-baseline)
                      the whole pipeline on an untrained encoder.

Context is probed by **classification**, per factor, not by R^2 and not
averaged. The factors are discrete (10/10/10/8/15 values) and three are hues --
points on a colour wheel, where 0.9 and 0.0 are neighbours but a linear
regression scores them as maximally distant. Averaging also hides that training
suppresses background hue while *raising* scale/orientation.

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
    max_useful_subspace_dim,
    project,
    solve_generalized_eigenproblem,
)
from jepa.configs.images.shapes3d import (  # noqa: E402
    SHAPES3D_CONTEXT_FACTORS,
    SHAPES3D_ENTITY_NAMES,
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
RANDOM_SUBSPACE_DRAWS = 5


def _context_class_labels(contexts: torch.Tensor, factor_index: int) -> torch.Tensor:
    """Undo the [0, 1] normalisation back into the original class index."""
    size = SHAPES3D_FACTOR_SIZES[SHAPES3D_CONTEXT_FACTORS[factor_index]]
    return (contexts[:, factor_index] * (size - 1)).round().long()


def _probe(train_features, train_labels, test_features, test_labels, num_classes, ridge):
    classifier = fit_entity_classifier(train_features, train_labels, num_classes, ridge=ridge)
    return classifier, classifier_accuracy(classifier, test_features, test_labels)


def _confusion_matrix(classifier, features: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    predicted = classifier.probe.predict(features).argmax(dim=-1)
    matrix = np.zeros((classifier.num_classes, classifier.num_classes), dtype=int)
    np.add.at(matrix, (labels.numpy(), predicted.numpy()), 1)
    return matrix


def _random_subspace_projection(dim: int, k: int, generator: torch.Generator) -> np.ndarray:
    raw = torch.randn(dim, k, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return (q @ q.T).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default="configs/images/shapes3d/quick.yaml")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-baseline", action="store_true")
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
    num_entities = config.data.num_entities

    core = build_spatial_ijepa_core(
        "cnn",
        patch_dim=3 * PATCH_SIZE * PATCH_SIZE,
        patch_latent_dim=PATCH_LATENT_DIM,
        num_patches=(64 // PATCH_SIZE) ** 2,
    )
    if args.random_baseline:
        epoch = 0
        torch.manual_seed(args.seed)
    else:
        checkpoint = load_spatial_checkpoint(args.checkpoint)
        epoch = checkpoint["epoch"]
        core.context_encoder.load_state_dict(checkpoint["context_encoder"])
    core.context_encoder.eval()

    train_z = encode_frames_pooled(core, train_frames, patch_size=PATCH_SIZE).cpu()
    test_z = encode_frames_pooled(core, test_frames, patch_size=PATCH_SIZE).cpu()

    k = max_useful_subspace_dim(num_entities, PATCH_LATENT_DIM)
    epsilon = config.evaluation.covariance_epsilon
    ridge = config.evaluation.probe_ridge
    generator = torch.Generator().manual_seed(args.seed)

    scatter = compute_scatter_matrices(train_z, train_entity, num_entities)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=epsilon)
    projection = entity_subspace_projection(eigen.eigenvectors, k)
    complement = np.eye(PATCH_LATENT_DIM) - projection

    # the same supervised search, run on noise: how much "structure" does LDA
    # invent when the labels carry none?
    permuted = train_entity[torch.randperm(train_entity.shape[0], generator=generator)]
    scatter_permuted = compute_scatter_matrices(train_z, permuted, num_entities)
    eigen_permuted = solve_generalized_eigenproblem(scatter_permuted, epsilon=epsilon)
    projection_permuted = entity_subspace_projection(eigen_permuted.eigenvectors, k)

    views = {
        "full": (train_z, test_z),
        "entity_subspace": (project(train_z, projection), project(test_z, projection)),
        "complement": (project(train_z, complement), project(test_z, complement)),
        "permuted_labels": (
            project(train_z, projection_permuted),
            project(test_z, projection_permuted),
        ),
    }

    record: dict = {"epoch": epoch, "subspace_dim": k, "seed": args.seed, "views": {}}
    confusion: dict = {}
    for view_name, (train_features, test_features) in views.items():
        classifier, shape_acc = _probe(
            train_features, train_entity, test_features, test_entity, num_entities, ridge
        )
        per_factor = {}
        for index, factor in enumerate(SHAPES3D_CONTEXT_FACTORS):
            _, per_factor[factor] = _probe(
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
        }
        if view_name in ("full", "entity_subspace"):
            matrix = _confusion_matrix(classifier, test_features, test_entity)
            confusion[view_name] = matrix.tolist()

    # k arbitrary directions -- the floor that LDA's label-informed choice must beat
    random_shape, random_context = [], {f: [] for f in SHAPES3D_CONTEXT_FACTORS}
    for _ in range(RANDOM_SUBSPACE_DRAWS):
        random_projection = _random_subspace_projection(PATCH_LATENT_DIM, k, generator)
        train_random = project(train_z, random_projection)
        test_random = project(test_z, random_projection)
        _, accuracy = _probe(
            train_random, train_entity, test_random, test_entity, num_entities, ridge
        )
        random_shape.append(accuracy)
        for index, factor in enumerate(SHAPES3D_CONTEXT_FACTORS):
            _, factor_accuracy = _probe(
                train_random,
                _context_class_labels(train_context, index),
                test_random,
                _context_class_labels(test_context, index),
                SHAPES3D_FACTOR_SIZES[factor],
                ridge,
            )
            random_context[factor].append(factor_accuracy)
    record["views"]["random_subspace"] = {
        "shape_accuracy": float(np.mean(random_shape)),
        "shape_accuracy_std": float(np.std(random_shape)),
        "context_accuracy": {f: float(np.mean(v)) for f, v in random_context.items()},
        "draws": RANDOM_SUBSPACE_DRAWS,
    }
    record["confusion_matrices"] = confusion

    label = "random_baseline" if args.random_baseline else f"epoch{epoch}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"subspace_decomposition_{label}_seed{args.seed}.json"
    out_path.write_text(json.dumps(record, indent=2))

    order = ["full", "entity_subspace", "random_subspace", "permuted_labels", "complement"]
    print(f"\n=== {label} (k={k}, seed={args.seed}) ===")
    print(f"{'view':<17} {'shape':>7} " + " ".join(f"{f[:9]:>9}" for f in SHAPES3D_CONTEXT_FACTORS))
    print(
        f"{'chance':<17} {1 / num_entities:>7.3f} "
        + " ".join(
            f"{1 / SHAPES3D_FACTOR_SIZES[f]:>9.3f}" for f in SHAPES3D_CONTEXT_FACTORS
        )
    )
    for view_name in order:
        values = record["views"][view_name]
        row = f"{view_name:<17} {values['shape_accuracy']:>7.3f} "
        row += " ".join(f"{values['context_accuracy'][f]:>9.3f}" for f in SHAPES3D_CONTEXT_FACTORS)
        print(row)

    print(f"\nconfusion (entity_subspace, rows=true, cols=pred), {SHAPES3D_ENTITY_NAMES}:")
    matrix = np.array(confusion["entity_subspace"])
    for i, name in enumerate(SHAPES3D_ENTITY_NAMES):
        counts = " ".join(f"{c:>5d}" for c in matrix[i])
        recall = matrix[i, i] / max(matrix[i].sum(), 1)
        print(f"  {name:<9} {counts}   recall={recall:.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
