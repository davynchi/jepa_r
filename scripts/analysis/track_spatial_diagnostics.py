#!/usr/bin/env python3
"""Watches /tmp/ijepa_spatial_checkpoints/ and runs the full held-out
diagnostic suite on each checkpoint as training proceeds.

The training script only reports train loss and effective_rank, which is not
enough to tell "learned something" from "found a cheap low-rank solution". Per
checkpoint this reports, all on the held-out test split:

  test_loss          held-out spatial I-JEPA prediction loss
  effective_rank     entropy of the covariance spectrum (collapse detector)
  top_eigs           leading covariance eigenvalues (is variance spread or in 1 dim?)
  entity_acc         linear probe: shape readable from the frozen latent?
  context_r2         linear probe: are the 5 context factors readable?
  Q_E raw/white      counterfactual invariance in the LDA entity subspace
  MI ratio           I(top LDA direction; shape) / H(shape), a nonlinear check

Probes are fit on train and scored on test; the LDA subspace is likewise fit on
train only, so nothing here is scored in-sample. Runs independently of the
training process; writes one JSON record per checkpoint.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from jepa.analysis.subspace import (  # noqa: E402
    apply_whitening,
    classifier_accuracy,
    compute_latent_spectrum,
    compute_scatter_matrices,
    context_r2_mean,
    counterfactual_invariance,
    entity_label_entropy,
    entity_subspace_projection,
    fit_context_regressor,
    fit_entity_classifier,
    fit_whitening,
    mutual_information_entity,
    project,
    solve_generalized_eigenproblem,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import load_shapes3d_config  # noqa: E402
from jepa.data.images.shapes3d import (  # noqa: E402
    build_shapes3d_counterfactual_pairs,
    build_shapes3d_dataset_splits,
)
from jepa.models.patches import patchify  # noqa: E402
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    MaskConfig,
    build_spatial_ijepa_core,
    encode_frames_pooled,
    load_spatial_checkpoint,
    sample_masks,
    spatial_ijepa_loss,
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "ijepa_spatial"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RECORDS_PATH = OUTPUT_DIR / "metrics" / "diagnostics.json"
PATCH_SIZE = 8
PATCH_LATENT_DIM = 16
BATCH_SIZE = 128
K_ENTITY_SUBSPACE = 2
POLL_SECONDS = 15


def _held_out_loss(core, test_patches, grid, mask_config, device) -> float:
    # Fixed mask seed so the number is comparable across checkpoints.
    mask_generator = torch.Generator().manual_seed(derive_seed(0, "test-masks"))
    total, batches = 0.0, 0
    with torch.no_grad():
        for indices in torch.arange(test_patches.shape[0]).split(BATCH_SIZE):
            batch = test_patches[indices].to(device)
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
    config = load_shapes3d_config(
        "configs/images/shapes3d/quick.yaml",
        overrides={"data.num_train_trajectories": "1000", "training.device": "cuda"},
    )
    datasets = build_shapes3d_dataset_splits(config.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_frames = datasets.train.frames.reshape(-1, 3, 64, 64)
    test_frames = datasets.test.frames.reshape(-1, 3, 64, 64)
    train_entity = datasets.train.entities.reshape(-1)
    test_entity = datasets.test.entities.reshape(-1)
    num_entities = config.data.num_entities
    context_dim = config.data.context_dim
    train_context = datasets.train.contexts.reshape(-1, context_dim)
    test_context = datasets.test.contexts.reshape(-1, context_dim)

    test_patches = patchify(test_frames, PATCH_SIZE)
    grid = 64 // PATCH_SIZE
    num_patches = test_patches.shape[1]
    patch_dim = test_patches.shape[2]
    mask_config = MaskConfig()

    source = datasets.train.source
    counterfactual = build_shapes3d_counterfactual_pairs(
        config.data,
        source,
        num_pairs=config.evaluation.counterfactual_pairs,
        seed=config.data.counterfactual_seed,
    )
    ridge = config.evaluation.probe_ridge
    covariance_epsilon = config.evaluation.covariance_epsilon

    RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seen: set[str] = set()
    print(f"watching {CHECKPOINT_DIR}", flush=True)
    while True:
        if CHECKPOINT_DIR.exists():
            for path in sorted(CHECKPOINT_DIR.glob("epoch_*.pt")):
                if path.name in seen:
                    continue
                seen.add(path.name)
                checkpoint = load_spatial_checkpoint(path)
                core = build_spatial_ijepa_core(
                    "cnn",
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

                train_z = encode_frames_pooled(core, train_frames, patch_size=PATCH_SIZE).cpu()
                test_z = encode_frames_pooled(core, test_frames, patch_size=PATCH_SIZE).cpu()
                spectrum = compute_latent_spectrum(test_z)

                classifier = fit_entity_classifier(train_z, train_entity, num_entities, ridge=ridge)
                entity_acc = classifier_accuracy(classifier, test_z, test_entity)
                regressor = fit_context_regressor(train_z, train_context, ridge=ridge)
                context_r2 = context_r2_mean(regressor, test_z, test_context)

                # LDA entity subspace, fit on train only.
                scatter = compute_scatter_matrices(train_z, train_entity, num_entities)
                eigen = solve_generalized_eigenproblem(scatter, epsilon=covariance_epsilon)
                projection = entity_subspace_projection(eigen.eigenvectors, K_ENTITY_SUBSPACE)

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
                projection_w = entity_subspace_projection(eigen_w.eigenvectors, K_ENTITY_SUBSPACE)
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

                record = {
                    "epoch": checkpoint["epoch"],
                    "test_loss": test_loss,
                    "effective_rank": spectrum.effective_rank,
                    "trace_covariance": spectrum.trace_covariance,
                    "top_eigenvalues": spectrum.eigenvalues[:6].tolist(),
                    "entity_accuracy": entity_acc,
                    "context_r2": context_r2.value,
                    "raw_q_entity": raw_inv.q_entity,
                    "white_q_entity": white_inv.q_entity,
                    "mi_ratio": mi / entropy if entropy > 0 else float("nan"),
                }
                records.append(record)
                RECORDS_PATH.write_text(json.dumps(records, indent=2))
                context_r2_text = (
                    "n/a" if record["context_r2"] is None else f"{record['context_r2']:.3f}"
                )
                print(
                    f"epoch={record['epoch']:4d} test_loss={test_loss:9.5f} "
                    f"eff_rank={spectrum.effective_rank:6.3f} "
                    f"entity_acc={entity_acc:.3f} context_r2={context_r2_text} "
                    f"Q_E={raw_inv.q_entity:8.3f} Q_E_w={white_inv.q_entity:8.3f} "
                    f"MI_ratio={record['mi_ratio']:.3f}",
                    flush=True,
                )
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
