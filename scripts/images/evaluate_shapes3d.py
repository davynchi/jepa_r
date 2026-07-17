#!/usr/bin/env python3
"""Full post-hoc analysis for one completed Shapes3D run.

    python scripts/images/evaluate_shapes3d.py --run-dir outputs/temporal_shapes3d_quick/<run-id>

Mirrors ``evaluate_temporal_image.py`` exactly (same scatter-matrix /
generalized-eigenproblem / probe / counterfactual-invariance pipeline from
:mod:`jepa.analysis.subspace`, fully generic over latent tensors) plus the
auxiliary 2D-PCA-by-entity plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jepa.analysis.metrics import compute_representation_metrics  # noqa: E402
from jepa.analysis.plots import (  # noqa: E402
    plot_counterfactual_distances,
    plot_effective_rank_over_training,
    plot_generalized_eigenvalue_spectrum,
    plot_latent_pca_scatter,
    plot_prediction_loss_curves,
    plot_probe_matrix,
)
from jepa.analysis.subspace import (  # noqa: E402
    apply_whitening,
    classifier_accuracy,
    compute_autocorrelation,
    compute_latent_spectrum,
    compute_scatter_matrices,
    context_r2_mean,
    counterfactual_invariance,
    empirical_entity_predictability,
    entity_selectivity_score,
    entity_subspace_projection,
    fit_context_regressor,
    fit_entity_classifier,
    fit_whitening,
    project,
    select_entity_subspace_dim,
    solve_generalized_eigenproblem,
)
from jepa.configs.images.shapes3d import (  # noqa: E402
    SHAPES3D_ENTITY_NAMES,
    shapes3d_config_from_dict,
)
from jepa.data.images.shapes3d import (
    build_shapes3d_counterfactual_pairs,
    build_shapes3d_dataset_splits,
)  # noqa: E402
from jepa.models.encoders import build_model_pair  # noqa: E402
from jepa.models.hierarchical import HierarchicalEncoderPredictor  # noqa: E402
from jepa.training.images.shapes3d import load_shapes3d_checkpoint  # noqa: E402


def _pca_2d(latents: torch.Tensor) -> np.ndarray:
    matrix = latents.detach().to(torch.float64).numpy()
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _build_encoder(config, checkpoint: dict[str, Any]):
    if checkpoint["model_kind"] == "hierarchical":
        model = HierarchicalEncoderPredictor(
            config.model.architecture,
            input_dim=config.data.observation_dim,
            latent_dim=config.model.latent_dim,
            entity_latent_dim=config.model.entity_latent_dim,
            context_latent_dim=config.model.context_latent_dim,
            hidden_dim=config.model.hidden_dim,
            hidden_layers=config.model.hidden_layers,
        )
        model.encoder.load_state_dict(checkpoint["online_encoder"])
        model.entity_head.load_state_dict(checkpoint["entity_head"])
        model.context_head.load_state_dict(checkpoint["context_head"])
        return model.encoder, model
    encoder, _ = build_model_pair(
        config.model.architecture,
        input_dim=config.data.observation_dim,
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
    )
    encoder.load_state_dict(checkpoint["context_encoder"])
    return encoder, None


@torch.no_grad()
def _encode_flat(encoder: torch.nn.Module, observations: torch.Tensor) -> torch.Tensor:
    encoder.eval()
    return encoder(observations)


def evaluate_shapes3d_run(run_dir: Path) -> dict[str, Any]:
    checkpoint = load_shapes3d_checkpoint(run_dir / "checkpoint.pt")
    config = shapes3d_config_from_dict(checkpoint["config"])
    encoder, hierarchical = _build_encoder(config, checkpoint)
    evaluation = config.evaluation

    splits = build_shapes3d_dataset_splits(config.data)
    train_ds, val_ds, test_ds = splits.train, splits.validation, splits.test
    obs_dim = config.data.observation_dim
    train_z = _encode_flat(encoder, train_ds.observations.reshape(-1, obs_dim))
    val_z = _encode_flat(encoder, val_ds.observations.reshape(-1, obs_dim))
    test_z = _encode_flat(encoder, test_ds.observations.reshape(-1, obs_dim))
    train_entity = train_ds.entities.reshape(-1)
    val_entity = val_ds.entities.reshape(-1)
    test_entity = test_ds.entities.reshape(-1)
    train_context = train_ds.contexts.reshape(-1, config.data.context_dim)
    val_context = val_ds.contexts.reshape(-1, config.data.context_dim)
    test_context = test_ds.contexts.reshape(-1, config.data.context_dim)

    scatter = compute_scatter_matrices(train_z, train_entity, config.data.num_entities)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=evaluation.covariance_epsilon)

    validation_accuracy: dict[int, float] = {}
    curve: dict[int, dict[str, float]] = {}
    for k in evaluation.entity_subspace_dims:
        projection = entity_subspace_projection(eigen.eigenvectors, k)
        train_proj = project(train_z, projection)
        val_proj = project(val_z, projection)
        classifier = fit_entity_classifier(
            train_proj, train_entity, config.data.num_entities, ridge=evaluation.probe_ridge
        )
        val_acc = classifier_accuracy(classifier, val_proj, val_entity)
        validation_accuracy[k] = val_acc
        context_probe = fit_context_regressor(
            train_proj, train_context, ridge=evaluation.probe_ridge
        )
        curve[k] = {
            "validation_entity_accuracy": val_acc,
            "validation_context_r2": context_r2_mean(context_probe, val_proj, val_context).value,
        }

    best_k = select_entity_subspace_dim(evaluation.entity_subspace_dims, validation_accuracy)
    projection = entity_subspace_projection(eigen.eigenvectors, best_k)
    identity = torch.eye(projection.shape[0], dtype=torch.float64).numpy()
    complement_projection = identity - projection

    representations = {
        "full_z": test_z,
        "entity_subspace": project(test_z, projection),
        "complement": project(test_z, complement_projection),
    }
    if hierarchical is not None:
        entity_dim = config.model.entity_latent_dim
        representations["explicit_zE"] = test_z[:, :entity_dim]
        representations["explicit_zC"] = test_z[:, entity_dim:]

    probe_rows: list[dict[str, Any]] = []
    for name, features in representations.items():
        if name == "full_z":
            train_features = train_z
        elif name == "entity_subspace":
            train_features = project(train_z, projection)
        elif name == "complement":
            train_features = project(train_z, complement_projection)
        elif name == "explicit_zE":
            train_features = train_z[:, : config.model.entity_latent_dim]
        else:
            train_features = train_z[:, config.model.entity_latent_dim :]
        classifier = fit_entity_classifier(
            train_features, train_entity, config.data.num_entities, ridge=evaluation.probe_ridge
        )
        context_probe = fit_context_regressor(
            train_features, train_context, ridge=evaluation.probe_ridge
        )
        entity_acc = classifier_accuracy(classifier, features, test_entity)
        context_r2 = context_r2_mean(context_probe, features, test_context)
        probe_rows.append(
            {
                "representation": name,
                "entity_accuracy": entity_acc,
                "context_r2": context_r2.value or 0.0,
            }
        )

    entity_row = next(r for r in probe_rows if r["representation"] == "entity_subspace")
    complement_row = next(r for r in probe_rows if r["representation"] == "complement")
    selectivity = entity_selectivity_score(
        entity_row["entity_accuracy"],
        complement_row["entity_accuracy"],
        entity_row["context_r2"],
        alpha=evaluation.selectivity_alpha,
    )

    # Reuse the already-open, memmap-cached source held by the train split's dataset
    # instead of reopening (and re-converting, if it somehow weren't cached) the file.
    counterfactual = build_shapes3d_counterfactual_pairs(
        config.data,
        train_ds.source,
        num_pairs=evaluation.counterfactual_pairs,
        seed=config.data.counterfactual_seed,
    )
    with torch.no_grad():
        same_1_z = encoder(counterfactual.same_entity_x1.reshape(-1, obs_dim))
        same_2_z = encoder(counterfactual.same_entity_x2.reshape(-1, obs_dim))
        diff_1_z = encoder(counterfactual.diff_entity_x1.reshape(-1, obs_dim))
        diff_2_z = encoder(counterfactual.diff_entity_x2.reshape(-1, obs_dim))
    same_1 = project(same_1_z, projection)
    same_2 = project(same_2_z, projection)
    diff_1 = project(diff_1_z, projection)
    diff_2 = project(diff_2_z, projection)
    invariance = counterfactual_invariance(same_1, same_2, diff_1, diff_2)

    # Whitened counterfactual invariance -- strips a raw-scale confound; see
    # evaluate_temporal_hierarchy.py for the full rationale.
    whitening = fit_whitening(train_z)
    train_z_white = apply_whitening(whitening, train_z)
    scatter_white = compute_scatter_matrices(train_z_white, train_entity, config.data.num_entities)
    eigen_white = solve_generalized_eigenproblem(
        scatter_white, epsilon=evaluation.covariance_epsilon
    )
    projection_white = entity_subspace_projection(eigen_white.eigenvectors, best_k)
    invariance_white = counterfactual_invariance(
        project(apply_whitening(whitening, same_1_z), projection_white),
        project(apply_whitening(whitening, same_2_z), projection_white),
        project(apply_whitening(whitening, diff_1_z), projection_white),
        project(apply_whitening(whitening, diff_2_z), projection_white),
    )

    collapse = compute_representation_metrics(test_z)
    spectrum = compute_latent_spectrum(test_z)
    mean_latent_norm = float(torch.linalg.vector_norm(test_z.double(), dim=1).mean().item())
    predictability = empirical_entity_predictability(
        train_ds.entities, horizon=config.training.prediction_horizon
    )

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    plot_generalized_eigenvalue_spectrum(
        {run_dir.name: eigen.eigenvalues.tolist()},
        analysis_dir / "generalized_eigenvalue_spectrum.png",
    )
    plot_probe_matrix(probe_rows, analysis_dir / "probe_matrix.png")
    plot_counterfactual_distances(
        [{"label": run_dir.name, **asdict(invariance)}],
        analysis_dir / "counterfactual_distances.png",
    )
    pca_points = _pca_2d(test_z)
    plot_latent_pca_scatter(
        pca_points,
        test_entity.tolist(),
        SHAPES3D_ENTITY_NAMES,
        analysis_dir / "latent_pca_by_entity.png",
    )

    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if history:
        rank_points = [
            (row["epoch"], row["effective_rank"])
            for row in history
            if row.get("split") == "validation" and row.get("effective_rank") is not None
        ]
        if rank_points:
            plot_effective_rank_over_training(
                {run_dir.name: rank_points}, analysis_dir / "effective_rank_over_training.png"
            )
        train_rows = {row["epoch"]: row for row in history if row["split"] == "train"}
        val_rows = {row["epoch"]: row for row in history if row["split"] == "validation"}
        if train_rows:
            loss_key = "loss_total" if "loss_total" in next(iter(train_rows.values())) else "loss"
            loss_points = [
                (
                    epoch,
                    train_rows[epoch].get(loss_key, float("nan")),
                    val_rows[epoch].get(loss_key, float("nan")),
                )
                for epoch in sorted(train_rows)
                if epoch in val_rows
            ]
            if loss_points:
                plot_prediction_loss_curves(
                    {run_dir.name: loss_points}, analysis_dir / "prediction_loss_curves.png"
                )

    val_z_traj = val_z.reshape(len(val_ds), config.data.trajectory_length, -1)
    autocorr = compute_autocorrelation(val_z_traj, max_lag=evaluation.autocorrelation_max_lag)

    summary = {
        "run_id": checkpoint["run_id"],
        "model_kind": checkpoint["model_kind"],
        "config": checkpoint["config"],
        "generalized_eigenvalues": eigen.eigenvalues.tolist(),
        "entity_subspace_dim_curve": curve,
        "selected_entity_subspace_dim": best_k,
        "probe_matrix": probe_rows,
        "selectivity_score": selectivity,
        "counterfactual": asdict(invariance),
        "counterfactual_whitened": asdict(invariance_white),
        "spectrum": {
            "trace_covariance": spectrum.trace_covariance,
            "effective_rank": spectrum.effective_rank,
            "top_eigenvalues": spectrum.eigenvalues[
                : min(8, spectrum.eigenvalues.shape[0])
            ].tolist(),
            "std_min": float(spectrum.per_dimension_std.min()),
            "std_max": float(spectrum.per_dimension_std.max()),
        },
        "mean_latent_norm": mean_latent_norm,
        "predictability": predictability,
        "collapse_diagnostics": {
            "effective_rank": collapse.effective_rank.value,
            "effective_rank_normalized": collapse.effective_rank_normalized.value,
            "latent_std_mean": collapse.latent_std_mean.value,
            "collapsed": collapse.collapsed,
            "collapse_reasons": list(collapse.collapse_reasons),
        },
        "autocorrelation_correlation_times": autocorr.correlation_times.tolist(),
        "test_loss": checkpoint.get("test_loss"),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = evaluate_shapes3d_run(args.run_dir)
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "selected_entity_subspace_dim": summary["selected_entity_subspace_dim"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
