#!/usr/bin/env python3
"""Full post-hoc analysis for one completed temporal-persistence run.

    python scripts/timeseries/evaluate.py --run-dir outputs/temporal/<run-id>

Loads the checkpoint, re-derives the identical train/validation/test/
counterfactual data (deterministic given the stored config), estimates the
post-hoc entity subspace from train+validation, fits evaluation probes on
train, computes every Section 7 metric on test, generates the Section 9
per-run plots, and writes ``analysis/summary.json`` inside the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataclasses import asdict  # noqa: E402

import torch  # noqa: E402

from jepa.analysis.metrics import compute_representation_metrics  # noqa: E402
from jepa.analysis.plots import (  # noqa: E402
    plot_counterfactual_distances,
    plot_effective_rank_over_training,
    plot_generalized_eigenvalue_spectrum,
    plot_latent_autocorrelation,
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
from jepa.configs.timeseries import (  # noqa: E402
    TemporalEvaluationConfig,
    temporal_config_from_dict,
)
from jepa.data.timeseries.entity_context import (  # noqa: E402
    EntityContextTrajectoryDataset,
    ObservationSystem,
    build_counterfactual_pairs,
)
from jepa.models.encoders import build_model_pair  # noqa: E402
from jepa.models.hierarchical import HierarchicalEncoderPredictor  # noqa: E402
from jepa.training.timeseries import load_temporal_checkpoint  # noqa: E402


def _rebuild_system(config, checkpoint: dict[str, Any]) -> ObservationSystem:
    payload = checkpoint["system"]
    return ObservationSystem(
        mode=payload["mode"],
        linear_map=payload["linear_map"],
        hidden_map=payload.get("hidden_map"),
        hidden_bias=payload.get("hidden_bias"),
        output_map=payload.get("output_map"),
        output_bias=payload.get("output_bias"),
    )


@torch.no_grad()
def _encode_flat(encoder: torch.nn.Module, dataset: EntityContextTrajectoryDataset) -> torch.Tensor:
    encoder.eval()
    flat = dataset.observations.reshape(-1, dataset.config.observation_dim)
    return encoder(flat)


def _build_encoder(
    config, checkpoint: dict[str, Any]
) -> tuple[torch.nn.Module, HierarchicalEncoderPredictor | None]:
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


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    checkpoint = load_temporal_checkpoint(run_dir / "checkpoint.pt")
    config = temporal_config_from_dict(checkpoint["config"])
    encoder, hierarchical = _build_encoder(config, checkpoint)

    system = _rebuild_system(config, checkpoint)
    train_ds = EntityContextTrajectoryDataset(config.data, "train", system=system)
    val_ds = EntityContextTrajectoryDataset(config.data, "validation", system=system)
    test_ds = EntityContextTrajectoryDataset(config.data, "test", system=system)

    train_z = _encode_flat(encoder, train_ds)
    val_z = _encode_flat(encoder, val_ds)
    test_z = _encode_flat(encoder, test_ds)
    train_entity = train_ds.entities.reshape(-1)
    val_entity = val_ds.entities.reshape(-1)
    test_entity = test_ds.entities.reshape(-1)
    train_context = train_ds.contexts.reshape(-1, config.data.context_dim)
    val_context = val_ds.contexts.reshape(-1, config.data.context_dim)
    test_context = test_ds.contexts.reshape(-1, config.data.context_dim)

    evaluation: TemporalEvaluationConfig = config.evaluation
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
        "full_z": (test_z, None),
        "entity_subspace": (project(test_z, projection), None),
        "complement": (project(test_z, complement_projection), None),
    }
    if hierarchical is not None:
        entity_dim = config.model.entity_latent_dim
        representations["explicit_zE"] = (test_z[:, :entity_dim], None)
        representations["explicit_zC"] = (test_z[:, entity_dim:], None)

    probe_rows: list[dict[str, Any]] = []
    train_projected_cache: dict[str, torch.Tensor] = {}
    for name, (features, _) in representations.items():
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
        train_projected_cache[name] = train_features
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

    counterfactual_config = build_counterfactual_pairs(
        config.data,
        system,
        num_pairs=evaluation.counterfactual_pairs,
        seed=config.data.counterfactual_seed,
    )
    with torch.no_grad():
        same_1_z = encoder(counterfactual_config.same_entity_x1)
        same_2_z = encoder(counterfactual_config.same_entity_x2)
        diff_1_z = encoder(counterfactual_config.diff_entity_x1)
        diff_2_z = encoder(counterfactual_config.diff_entity_x2)
    same_1 = project(same_1_z, projection)
    same_2 = project(same_2_z, projection)
    diff_1 = project(diff_1_z, projection)
    diff_2 = project(diff_2_z, projection)
    invariance = counterfactual_invariance(same_1, same_2, diff_1, diff_2)

    # Whitened counterfactual invariance: strips out a raw-scale confound (an
    # encoder can shrink/grow D_same and D_diff together just by changing its
    # overall output norm, without its relative entity/context geometry
    # changing at all -- see the P0 bottleneck-confound writeup). Re-derive the
    # entity subspace in whitened coordinates rather than reusing `projection`,
    # since whitening changes the geometry the generalized eigenproblem sees.
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
    predictability = empirical_entity_predictability(
        train_ds.entities, horizon=config.training.prediction_horizon
    )
    mean_latent_norm = float(torch.linalg.vector_norm(test_z.double(), dim=1).mean().item())
    autocorr = compute_autocorrelation(
        val_z.reshape(len(val_ds), config.data.trajectory_length, -1),
        max_lag=evaluation.autocorrelation_max_lag,
    )

    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

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
        loss_key = "loss_total" if "loss_total" in next(iter(train_rows.values()), {}) else "loss"
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
    top_dims = list(range(min(4, autocorr.per_dimension.shape[0])))
    plot_latent_autocorrelation(
        {f"dim {j}": (autocorr.lags, autocorr.per_dimension[j].tolist()) for j in top_dims},
        analysis_dir / "latent_autocorrelation.png",
    )

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
    summary = evaluate_run(args.run_dir)
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
