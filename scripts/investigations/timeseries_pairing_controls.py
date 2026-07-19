#!/usr/bin/env python3
"""Ad-hoc diagnostic-grid investigation (not part of the permanent CLI).

Bottleneck grid: d_z in {4, 8, 16} x p_E in {0.01,0.05,0.2,0.5} x h in {1,8} x
pairing in {temporal, shuffled, oracle_same_entity} x 5 seeds, plus untrained
random_encoder and random_orthogonal_projection baselines (d_z x 5 seeds,
p_E/h-independent). For every condition: D_same/D_diff/Q_E raw + whitened,
full covariance spectrum, trace, effective rank, mean latent norm, test loss,
entity/context probes on full latent, entity subspace, and complement.

NOTE: internally the oracle control is still implemented via the
``shuffled_same_entity`` target_pairing value (it uses true entity labels to
build pairs -- it's an oracle, not something JEPA could do unsupervised), but
is reported under the name ``oracle_same_entity`` per the review discussion.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
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
    empirical_entity_predictability,
    entity_subspace_projection,
    fit_context_regressor,
    fit_entity_classifier,
    fit_whitening,
    max_useful_subspace_dim,
    project,
    solve_generalized_eigenproblem,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.timeseries import TemporalExperimentConfig, load_temporal_config  # noqa: E402
from jepa.data.timeseries.entity_context import (  # noqa: E402
    EntityContextTrajectoryDataset,
    ObservationSystem,
    build_counterfactual_pairs,
)
from jepa.models.encoders import build_model_pair  # noqa: E402
from jepa.training.core import build_jepa_core  # noqa: E402
from jepa.training.timeseries import train_temporal_experiment  # noqa: E402

OUTPUT_ROOT = str(Path(__file__).resolve().parents[2] / "outputs" / "timeseries_bottleneck_grid")
SEEDS = (0, 1, 2, 3, 4)
D_Z_VALUES = (4, 8, 16)
P_E_VALUES = (0.01, 0.05, 0.20, 0.50)
HORIZONS = (1, 8)
PAIRINGS = ("temporal", "shuffled", "shuffled_same_entity")
PAIRING_LABELS = {
    "temporal": "temporal",
    "shuffled": "shuffled",
    "shuffled_same_entity": "oracle_same_entity",
}


def base_config() -> TemporalExperimentConfig:
    return load_temporal_config(
        "configs/timeseries/hierarchy_full.yaml",
        overrides={"output.root": OUTPUT_ROOT, "output.overwrite": "on"},
    )


def _rebuild_system(config: TemporalExperimentConfig, checkpoint: dict) -> ObservationSystem:
    payload = checkpoint["system"]
    return ObservationSystem(mode="linear", linear_map=payload["linear_map"])


def _full_analysis(
    encoder,
    config: TemporalExperimentConfig,
    system: ObservationSystem,
    *,
    extra: dict,
) -> dict:
    train_ds = EntityContextTrajectoryDataset(config.data, "train", system=system)
    test_ds = EntityContextTrajectoryDataset(config.data, "test", system=system)

    with torch.no_grad():
        train_z = encoder(train_ds.observations.reshape(-1, config.data.observation_dim))
        test_z = encoder(test_ds.observations.reshape(-1, config.data.observation_dim))
    train_entity = train_ds.entities.reshape(-1)
    test_entity = test_ds.entities.reshape(-1)
    train_context = train_ds.contexts.reshape(-1, config.data.context_dim)
    test_context = test_ds.contexts.reshape(-1, config.data.context_dim)

    latent_dim = train_z.shape[1]
    k = max_useful_subspace_dim(config.data.num_entities, latent_dim)

    scatter = compute_scatter_matrices(train_z, train_entity, config.data.num_entities)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=config.evaluation.covariance_epsilon)
    projection = entity_subspace_projection(eigen.eigenvectors, k)
    identity = torch.eye(latent_dim, dtype=torch.float64).numpy()
    complement = identity - projection

    cf = build_counterfactual_pairs(
        config.data,
        system,
        num_pairs=config.evaluation.counterfactual_pairs,
        seed=config.data.test_sample_seed + 999,
    )
    with torch.no_grad():
        same_1, same_2 = encoder(cf.same_entity_x1), encoder(cf.same_entity_x2)
        diff_1, diff_2 = encoder(cf.diff_entity_x1), encoder(cf.diff_entity_x2)

    raw_inv = counterfactual_invariance(
        project(same_1, projection),
        project(same_2, projection),
        project(diff_1, projection),
        project(diff_2, projection),
    )

    whitening = fit_whitening(train_z)
    train_z_w = apply_whitening(whitening, train_z)
    scatter_w = compute_scatter_matrices(train_z_w, train_entity, config.data.num_entities)
    eigen_w = solve_generalized_eigenproblem(
        scatter_w, epsilon=config.evaluation.covariance_epsilon
    )
    projection_w = entity_subspace_projection(eigen_w.eigenvectors, k)
    white_inv = counterfactual_invariance(
        project(apply_whitening(whitening, same_1), projection_w),
        project(apply_whitening(whitening, same_2), projection_w),
        project(apply_whitening(whitening, diff_1), projection_w),
        project(apply_whitening(whitening, diff_2), projection_w),
    )

    spectrum = compute_latent_spectrum(test_z)

    def _probe(features_train, features_test):
        clf = fit_entity_classifier(
            features_train,
            train_entity,
            config.data.num_entities,
            ridge=config.evaluation.probe_ridge,
        )
        ctx_probe = fit_context_regressor(
            features_train, train_context, ridge=config.evaluation.probe_ridge
        )
        acc = classifier_accuracy(clf, features_test, test_entity)
        r2 = context_r2_mean(ctx_probe, features_test, test_context)
        return acc, (r2.value if r2.value is not None else float("nan"))

    full_acc, full_r2 = _probe(train_z, test_z)
    comp_acc, comp_r2 = _probe(project(train_z, complement), project(test_z, complement))

    predictability = empirical_entity_predictability(
        train_ds.entities, horizon=config.training.prediction_horizon
    )
    mean_norm = float(torch.linalg.vector_norm(test_z.double(), dim=1).mean().item())

    return {
        **extra,
        "latent_dim": latent_dim,
        "predictability": predictability,
        "raw_d_same": raw_inv.d_same,
        "raw_d_diff": raw_inv.d_diff,
        "raw_q_entity": raw_inv.q_entity,
        "white_d_same": white_inv.d_same,
        "white_d_diff": white_inv.d_diff,
        "white_q_entity": white_inv.q_entity,
        "trace_covariance": spectrum.trace_covariance,
        "effective_rank": spectrum.effective_rank,
        "top_eigenvalues": spectrum.eigenvalues[: min(8, latent_dim)].tolist(),
        "std_min": float(spectrum.per_dimension_std.min()),
        "std_max": float(spectrum.per_dimension_std.max()),
        "mean_latent_norm": mean_norm,
        "full_entity_accuracy": full_acc,
        "full_context_r2": full_r2,
        "complement_entity_accuracy": comp_acc,
        "complement_context_r2": comp_r2,
    }


def run_trained(seed: int, d_z: int, p_e: float, horizon: int, pairing: str) -> dict:
    config = base_config()
    config = replace(
        config,
        model=replace(config.model, kind="standard", latent_dim=d_z),
        data=replace(config.data, entity_switch_probability=p_e),
        training=replace(
            config.training, seed=seed, target_pairing=pairing, prediction_horizon=horizon
        ),
    )
    result = train_temporal_experiment(
        config, seed_label=derive_seed(seed, "dz", d_z, "pe", p_e, "h", horizon, "pairing", pairing)
    )
    checkpoint = torch.load(
        result.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    encoder, _ = build_model_pair(
        config.model.architecture,
        input_dim=config.data.observation_dim,
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
    )
    encoder.load_state_dict(checkpoint["context_encoder"])
    encoder.eval()
    system = _rebuild_system(config, checkpoint)
    metrics = _full_analysis(
        encoder,
        config,
        system,
        extra={
            "condition": PAIRING_LABELS[pairing],
            "seed": seed,
            "d_z": d_z,
            "p_entity_switch": p_e,
            "horizon": horizon,
            "test_loss": result.metrics["test_loss"],
        },
    )
    return metrics


def run_random_encoder(seed: int, d_z: int) -> dict:
    config = base_config()
    config = replace(
        config,
        model=replace(config.model, kind="random", latent_dim=d_z),
        training=replace(config.training, seed=seed),
    )
    from jepa.configs.base import derive_seed as _derive
    from jepa.training.core import _seed_everything

    _seed_everything(_derive(seed, config.model.architecture, "model"))
    core = build_jepa_core(
        config.model.architecture,
        input_dim=config.data.observation_dim,
        latent_dim=config.model.latent_dim,
        stop_gradient=True,
        ema_enabled=True,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
    )
    from jepa.data.timeseries.entity_context import generate_observation_system

    system = generate_observation_system(config.data)
    metrics = _full_analysis(
        core.context_encoder,
        config,
        system,
        extra={
            "condition": "random_encoder",
            "seed": seed,
            "d_z": d_z,
            "p_entity_switch": None,
            "horizon": None,
            "test_loss": None,
        },
    )
    return metrics


def run_random_orthogonal(seed: int, d_z: int) -> dict:
    config = base_config()
    config = replace(config, model=replace(config.model, latent_dim=d_z))
    from jepa.data.timeseries.entity_context import generate_observation_system

    system = generate_observation_system(config.data)
    generator = torch.Generator().manual_seed(derive_seed(seed, "orthogonal", d_z))
    raw = torch.randn(config.data.observation_dim, d_z, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")  # observation_dim x d_z, orthonormal columns

    class _OrthogonalEncoder:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return (x.double() @ q).float()

        def eval(self):
            return self

    metrics = _full_analysis(
        _OrthogonalEncoder(),
        config,
        system,
        extra={
            "condition": "random_orthogonal",
            "seed": seed,
            "d_z": d_z,
            "p_entity_switch": None,
            "horizon": None,
            "test_loss": None,
        },
    )
    return metrics


def main() -> None:
    records = []
    start = time.time()
    total = len(D_Z_VALUES) * len(P_E_VALUES) * len(HORIZONS) * len(PAIRINGS) * len(SEEDS)
    done = 0

    for d_z in D_Z_VALUES:
        for seed in SEEDS:
            records.append(run_random_encoder(seed, d_z))
            records.append(run_random_orthogonal(seed, d_z))

    for d_z in D_Z_VALUES:
        for p_e in P_E_VALUES:
            for horizon in HORIZONS:
                for pairing in PAIRINGS:
                    for seed in SEEDS:
                        metrics = run_trained(seed, d_z, p_e, horizon, pairing)
                        records.append(metrics)
                        done += 1
                        if done % 20 == 0:
                            elapsed = time.time() - start
                            print(
                                f"[{done}/{total}] d_z={d_z} p_E={p_e} h={horizon} "
                                f"pairing={PAIRING_LABELS[pairing]} seed={seed} "
                                f"Q_E(raw)={metrics['raw_q_entity']:.3g} elapsed={elapsed:.0f}s",
                                flush=True,
                            )

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    with open(f"{OUTPUT_ROOT}/records.json", "w") as f:
        json.dump(records, f, indent=2)
    elapsed_total = time.time() - start
    print(f"\nwrote {len(records)} records to {OUTPUT_ROOT}/records.json in {elapsed_total:.0f}s")


if __name__ == "__main__":
    main()
