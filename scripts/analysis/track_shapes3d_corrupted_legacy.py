#!/usr/bin/env python3
"""Watch one spatial I-JEPA run directory and compute held-out diagnostics.

The training script only reports train loss and effective_rank, which is not
enough to tell "learned something" from "found a cheap low-rank solution". Per
checkpoint this reports, all on the held-out test split:

  test_loss          held-out spatial I-JEPA prediction loss
  effective_rank     entropy of the covariance spectrum (collapse detector)
  top_eigs           leading covariance eigenvalues (is variance spread or in 1 dim?)
  entity_acc         linear probe: shape readable from the frozen latent?
  context_acc        linear probe per factor (classification, not R^2: the
                     factors are discrete and the hues are cyclic, which R^2
                     scores as maximally-distant neighbours)
  Q_E raw/white      counterfactual invariance in the LDA entity subspace
  MI ratio           I(top LDA direction; shape) / H(shape), a nonlinear check

Probes are fit on train and scored on test; the LDA subspace is likewise fit on
train only, so nothing here is scored in-sample. Runs independently of the
training process; writes one JSON record per checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jepa.analysis.subspace import (  # noqa: E402
    apply_whitening,
    classifier_accuracy,
    compute_latent_spectrum,
    compute_scatter_matrices,
    counterfactual_invariance,
    entity_label_entropy,
    entity_subspace_projection,
    fit_entity_classifier,
    fit_whitening,
    max_useful_subspace_dim,
    mutual_information_entity,
    project,
    solve_generalized_eigenproblem,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import (  # noqa: E402
    SHAPES3D_CONTEXT_FACTORS,
    SHAPES3D_FACTOR_SIZES,
    shapes3d_config_from_dict,
)
from jepa.data.images.shapes3d import (  # noqa: E402
    build_shapes3d_counterfactual_pairs,
    build_shapes3d_static_dataset_splits,
)
from jepa.models.patches import patchify  # noqa: E402
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    MaskConfig,
    build_spatial_ijepa_core,
    encode_frames_pooled,
    encode_frames_pooled_batched,
    load_spatial_checkpoint,
    sample_masks,
    spatial_ijepa_loss,
)
from jepa.training.images.spatial_logging import SpatialRunLogger  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "ijepa_spatial"
PATCH_SIZE = 8
PATCH_LATENT_DIM = 16
BATCH_SIZE = 128
POLL_SECONDS = 15


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help=f"Spatial run directory under {DEFAULT_OUTPUT_ROOT} or an absolute path",
    )
    parser.add_argument("--checkpoint-pattern", default="epoch_*.pt")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for diagnostics; use cpu to avoid competing with training GPU",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def _resolve_run_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = DEFAULT_OUTPUT_ROOT / path
    return path.resolve()


def _context_class_labels(contexts: torch.Tensor, factor_index: int) -> torch.Tensor:
    """Undo the [0, 1] normalisation back to the factor's original class index."""
    size = SHAPES3D_FACTOR_SIZES[SHAPES3D_CONTEXT_FACTORS[factor_index]]
    return (contexts[:, factor_index] * (size - 1)).round().long()


def _held_out_loss(core, test_samples, grid, mask_config, device) -> float:
    # Fixed mask seed so the number is comparable across checkpoints.
    mask_generator = torch.Generator().manual_seed(derive_seed(0, "test-masks"))
    total, batches = 0.0, 0
    with torch.no_grad():
        for indices in torch.arange(test_samples.shape[0]).split(BATCH_SIZE):
            batch = test_samples[indices].to(device)
            context_masks, target_masks = sample_masks(grid, grid, mask_config, mask_generator)
            loss = torch.zeros((), device=device)
            for context_mask in context_masks:
                for target_mask in target_masks:
                    loss = loss + spatial_ijepa_loss(
                        core, batch, context_mask.to(device), target_mask.to(device)
                    )
            total += (loss / (len(context_masks) * len(target_masks))).item()
            batches += 1
    return total / batches


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    records_path = run_dir / "metrics" / "diagnostics.json"
    logger = SpatialRunLogger(run_dir, enable_tensorboard=not args.no_tensorboard)

    run_config_path = run_dir / "config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"missing run config: {run_config_path}")
    run_config = json.loads(run_config_path.read_text())
    architecture = run_config.get("spatial", {}).get("architecture", "cnn")
    config = shapes3d_config_from_dict(run_config["config"])
    datasets = build_shapes3d_static_dataset_splits(config.data)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_frames = datasets.train.images
    test_frames = datasets.test.images
    train_entity = datasets.train.entities
    test_entity = datasets.test.entities
    num_entities = config.data.num_entities
    train_context = datasets.train.contexts
    test_context = datasets.test.contexts

    test_patches = patchify(test_frames, PATCH_SIZE)
    grid = 64 // PATCH_SIZE
    num_patches = test_patches.shape[1]
    patch_dim = test_patches.shape[2]
    mask_config = MaskConfig()
    # rank(S_B) = num_entities - 1; any k beyond that is a zero-eigenvalue
    # noise direction that inflates D_same and drags Q_E toward 1
    k_entity = max_useful_subspace_dim(num_entities, PATCH_LATENT_DIM)

    source = datasets.train.source
    counterfactual = build_shapes3d_counterfactual_pairs(
        config.data,
        source,
        num_pairs=config.evaluation.counterfactual_pairs,
        seed=config.data.counterfactual_seed,
    )
    ridge = config.evaluation.probe_ridge
    covariance_epsilon = config.evaluation.covariance_epsilon

    records_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if records_path.exists():
        loaded_records = json.loads(records_path.read_text())
        if not isinstance(loaded_records, list):
            raise ValueError(f"diagnostics records must be a list: {records_path}")
        records = loaded_records
    seen: set[str] = {str(record["checkpoint"]) for record in records if isinstance(record, dict)}
    print(f"watching {checkpoint_dir}", flush=True)
    try:
        while True:
            if checkpoint_dir.exists():
                for path in sorted(checkpoint_dir.glob(args.checkpoint_pattern)):
                    if path.name in seen:
                        continue
                    seen.add(path.name)
                    checkpoint = load_spatial_checkpoint(path)
                    core = build_spatial_ijepa_core(
                        architecture,
                        patch_dim=patch_dim,
                        patch_latent_dim=PATCH_LATENT_DIM,
                        num_patches=num_patches,
                    )
                    core.context_encoder.load_state_dict(checkpoint["context_encoder"])
                    core.predictor.load_state_dict(checkpoint["predictor"])
                    core.target_encoder.load_state_dict(checkpoint["target_encoder"])
                    core.context_encoder.to(device).eval()
                    core.predictor.to(device).eval()
                    core.target_encoder.to(device).eval()

                    test_loss = _held_out_loss(core, test_patches, grid, mask_config, device)

                    train_z = encode_frames_pooled_batched(
                        core,
                        train_frames,
                        patch_size=PATCH_SIZE,
                        batch_size=BATCH_SIZE,
                    ).cpu()
                    test_z = encode_frames_pooled_batched(
                        core,
                        test_frames,
                        patch_size=PATCH_SIZE,
                        batch_size=BATCH_SIZE,
                    ).cpu()
                    spectrum = compute_latent_spectrum(test_z)

                    classifier = fit_entity_classifier(
                        train_z, train_entity, num_entities, ridge=ridge
                    )
                    entity_acc = classifier_accuracy(classifier, test_z, test_entity)
                    context_acc = {}
                    for index, factor in enumerate(SHAPES3D_CONTEXT_FACTORS):
                        factor_classifier = fit_entity_classifier(
                            train_z,
                            _context_class_labels(train_context, index),
                            SHAPES3D_FACTOR_SIZES[factor],
                            ridge=ridge,
                        )
                        context_acc[factor] = classifier_accuracy(
                            factor_classifier, test_z, _context_class_labels(test_context, index)
                        )
                    context_acc_mean = float(np.mean(list(context_acc.values())))

                    # LDA entity subspace, fit on train only.
                    scatter = compute_scatter_matrices(train_z, train_entity, num_entities)
                    eigen = solve_generalized_eigenproblem(scatter, epsilon=covariance_epsilon)
                    projection = entity_subspace_projection(eigen.eigenvectors, k_entity)

                    with torch.no_grad():
                        same_1 = encode_frames_pooled(
                            core, counterfactual.same_entity_x1, patch_size=PATCH_SIZE
                        ).cpu()
                        same_2 = encode_frames_pooled(
                            core, counterfactual.same_entity_x2, patch_size=PATCH_SIZE
                        ).cpu()
                        diff_1 = encode_frames_pooled(
                            core, counterfactual.diff_entity_x1, patch_size=PATCH_SIZE
                        ).cpu()
                        diff_2 = encode_frames_pooled(
                            core, counterfactual.diff_entity_x2, patch_size=PATCH_SIZE
                        ).cpu()
                    raw_inv = counterfactual_invariance(
                        project(same_1, projection),
                        project(same_2, projection),
                        project(diff_1, projection),
                        project(diff_2, projection),
                    )
                    whitening = fit_whitening(train_z)
                    scatter_w = compute_scatter_matrices(
                        apply_whitening(whitening, train_z), train_entity, num_entities
                    )
                    eigen_w = solve_generalized_eigenproblem(scatter_w, epsilon=covariance_epsilon)
                    projection_w = entity_subspace_projection(eigen_w.eigenvectors, k_entity)
                    white_inv = counterfactual_invariance(
                        project(apply_whitening(whitening, same_1), projection_w),
                        project(apply_whitening(whitening, same_2), projection_w),
                        project(apply_whitening(whitening, diff_1), projection_w),
                        project(apply_whitening(whitening, diff_2), projection_w),
                    )

                    top_direction = torch.as_tensor(eigen.eigenvectors[:, 0])
                    mi = mutual_information_entity(
                        test_z.to(torch.float64) @ top_direction, test_entity, num_entities
                    )
                    entropy = entity_label_entropy(test_entity, num_entities)
                    step = int(checkpoint.get("global_step") or 0)

                    record = {
                        "checkpoint": path.name,
                        "epoch": checkpoint["epoch"],
                        "global_step": step,
                        "subspace_dim": k_entity,
                        "test_loss": test_loss,
                        "effective_rank": spectrum.effective_rank,
                        "trace_covariance": spectrum.trace_covariance,
                        "top_eigenvalues": spectrum.eigenvalues[:6].tolist(),
                        "entity_accuracy": entity_acc,
                        "context_accuracy": context_acc,
                        "context_accuracy_mean": context_acc_mean,
                        "raw_q_entity": raw_inv.q_entity,
                        "white_q_entity": white_inv.q_entity,
                        "mi_ratio": mi / entropy if entropy > 0 else float("nan"),
                    }
                    records.append(record)
                    records_path.write_text(json.dumps(records, indent=2))
                    logger.log(
                        step=step,
                        epoch=int(checkpoint["epoch"]),
                        event="heldout_diagnostics",
                        scalars={
                            "diag/test_loss": test_loss,
                            "diag/effective_rank": spectrum.effective_rank,
                            "diag/trace_covariance": spectrum.trace_covariance,
                            "diag/entity_accuracy": entity_acc,
                            "diag/context_accuracy_mean": context_acc_mean,
                            "diag/raw_q_entity": raw_inv.q_entity,
                            "diag/white_q_entity": white_inv.q_entity,
                            "diag/mi_ratio": record["mi_ratio"],
                        },
                        histograms={"hist/diag_eigenvalues": torch.as_tensor(spectrum.eigenvalues)},
                    )
                    print(
                        f"checkpoint={path.name} epoch={record['epoch']:4d} "
                        f"test_loss={test_loss:9.5f} eff_rank={spectrum.effective_rank:6.3f} "
                        f"entity_acc={entity_acc:.3f} context_acc={context_acc_mean:.3f} "
                        f"Q_E={raw_inv.q_entity:8.3f} Q_E_w={white_inv.q_entity:8.3f} "
                        f"MI_ratio={record['mi_ratio']:.3f}",
                        flush=True,
                    )
            if args.once:
                break
            time.sleep(args.poll_seconds)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
