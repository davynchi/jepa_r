#!/usr/bin/env python3
"""Ad-hoc diagnostic investigation for the Shapes3D image world.

Mirrors ``_investigate_pairing_controls.py`` (vector world) and
exactly: same
bottleneck question, same controls ({temporal, shuffled, oracle_same_entity}
plus random_encoder / random_orthogonal baselines), same metrics (D_same/D_diff
raw + whitened, full spectrum, entity/context probes on full latent and
complement). Requires ``data/3dshapes.h5`` (see README).

NOTE: internally the oracle control is implemented via the
``shuffled_same_entity`` target_pairing value (true entity labels are used to
build pairs -- it is an oracle, not something JEPA could do unsupervised),
reported here as ``oracle_same_entity``.
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
    entity_subspace_projection,
    fit_context_regressor,
    fit_entity_classifier,
    fit_whitening,
    project,
    solve_generalized_eigenproblem,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import (  # noqa: E402
    Shapes3DExperimentConfig,
    load_shapes3d_config,
)
from jepa.data.images.shapes3d import (  # noqa: E402
    Shapes3DDatasetSplits,
    Shapes3DSource,
    build_shapes3d_counterfactual_pairs,
    build_shapes3d_dataset_splits,
)
from jepa.models.encoders import build_model_pair  # noqa: E402
from jepa.training.core import build_jepa_core  # noqa: E402
from jepa.training.images.shapes3d import train_shapes3d_experiment  # noqa: E402

OUTPUT_ROOT = str(Path(__file__).resolve().parents[2] / "outputs" / "shapes3d_bottleneck_grid")
SEEDS = (0, 1, 2)
D_Z_VALUES = (8, 32)
PAIRINGS = ("temporal", "shuffled", "shuffled_same_entity")
PAIRING_LABELS = {
    "temporal": "temporal",
    "shuffled": "shuffled",
    "shuffled_same_entity": "oracle_same_entity",
}
K_ENTITY_SUBSPACE = 2

# config.data (and evaluation.counterfactual_pairs) never varies across a grid run --
# only model/training fields do (d_z, architecture, pairing, seed) -- so the full
# train/validation/test trajectory splits and the counterfactual pairs are
# bit-identical for every one of the 27-39 cells in a run. Rebuilding them per cell
# (both here for analysis AND again inside train_shapes3d_experiment for training)
# was pure CPU waste -- each split is a multi-GB fancy-index gather out of the
# Shapes3D memmap plus a per-trajectory Python generation loop (~20s alone) -- and
# was the main reason the GPU sat idle for minutes before/between cells.
_DATASET_CACHE: dict[tuple, tuple[Shapes3DDatasetSplits, object]] = {}


def base_config() -> Shapes3DExperimentConfig:
    return load_shapes3d_config(
        "configs/images/shapes3d/quick.yaml",
        overrides={"output.root": OUTPUT_ROOT, "output.overwrite": "on"},
    )


def _cached_datasets(
    config: Shapes3DExperimentConfig, source: Shapes3DSource
) -> tuple[Shapes3DDatasetSplits, object]:
    key = (config.data, config.evaluation.counterfactual_pairs)
    cached = _DATASET_CACHE.get(key)
    if cached is None:
        splits = build_shapes3d_dataset_splits(config.data)
        cf = build_shapes3d_counterfactual_pairs(
            config.data,
            source,
            num_pairs=config.evaluation.counterfactual_pairs,
            seed=config.data.counterfactual_seed,
        )
        cached = (splits, cf)
        _DATASET_CACHE[key] = cached
    return cached


def _full_analysis(
    encoder,
    config: Shapes3DExperimentConfig,
    source: Shapes3DSource,
    *,
    extra: dict,
) -> dict:
    splits, cf = _cached_datasets(config, source)
    train_ds, test_ds = splits.train, splits.test
    obs_dim = config.data.observation_dim

    with torch.no_grad():
        train_z = encoder(train_ds.observations.reshape(-1, obs_dim))
        test_z = encoder(test_ds.observations.reshape(-1, obs_dim))
    train_entity = train_ds.entities.reshape(-1)
    test_entity = test_ds.entities.reshape(-1)
    train_context = train_ds.contexts.reshape(-1, config.data.context_dim)
    test_context = test_ds.contexts.reshape(-1, config.data.context_dim)

    latent_dim = train_z.shape[1]
    k = min(K_ENTITY_SUBSPACE, latent_dim - 1) if latent_dim > 1 else latent_dim

    scatter = compute_scatter_matrices(train_z, train_entity, config.data.num_entities)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=config.evaluation.covariance_epsilon)
    projection = entity_subspace_projection(eigen.eigenvectors, k)
    identity = torch.eye(latent_dim, dtype=torch.float64).numpy()
    complement = identity - projection

    with torch.no_grad():
        same_1 = encoder(cf.same_entity_x1.reshape(-1, obs_dim))
        same_2 = encoder(cf.same_entity_x2.reshape(-1, obs_dim))
        diff_1 = encoder(cf.diff_entity_x1.reshape(-1, obs_dim))
        diff_2 = encoder(cf.diff_entity_x2.reshape(-1, obs_dim))

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
    mean_norm = float(torch.linalg.vector_norm(test_z.double(), dim=1).mean().item())

    return {
        **extra,
        "latent_dim": latent_dim,
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


def run_trained(seed: int, d_z: int, pairing: str, source: Shapes3DSource) -> dict:
    config = base_config()
    config = replace(
        config,
        model=replace(config.model, kind="standard", latent_dim=d_z),
        training=replace(config.training, seed=seed, target_pairing=pairing),
    )
    splits, _ = _cached_datasets(config, source)
    result = train_shapes3d_experiment(
        config, seed_label=derive_seed(seed, "dz", d_z, "pairing", pairing), datasets=splits
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
    metrics = _full_analysis(
        encoder,
        config,
        source,
        extra={
            "condition": PAIRING_LABELS[pairing],
            "seed": seed,
            "d_z": d_z,
            "test_loss": result.metrics["test_loss"],
        },
    )
    return metrics


def run_random_encoder(seed: int, d_z: int, source: Shapes3DSource) -> dict:
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
    metrics = _full_analysis(
        core.context_encoder,
        config,
        source,
        extra={"condition": "random_encoder", "seed": seed, "d_z": d_z, "test_loss": None},
    )
    return metrics


def run_random_orthogonal(seed: int, d_z: int, source: Shapes3DSource) -> dict:
    config = base_config()
    config = replace(config, model=replace(config.model, latent_dim=d_z))
    generator = torch.Generator().manual_seed(derive_seed(seed, "orthogonal", d_z))
    raw = torch.randn(config.data.observation_dim, d_z, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")

    class _OrthogonalEncoder:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return (x.double() @ q).float()

        def eval(self):
            return self

    metrics = _full_analysis(
        _OrthogonalEncoder(),
        config,
        source,
        extra={"condition": "random_orthogonal", "seed": seed, "d_z": d_z, "test_loss": None},
    )
    return metrics


def main() -> None:
    source = Shapes3DSource(base_config().data.h5_path)
    records = []
    start = time.time()
    total = len(D_Z_VALUES) * len(PAIRINGS) * len(SEEDS)
    done = 0

    for d_z in D_Z_VALUES:
        for seed in SEEDS:
            records.append(run_random_encoder(seed, d_z, source))
            records.append(run_random_orthogonal(seed, d_z, source))

    for d_z in D_Z_VALUES:
        for pairing in PAIRINGS:
            for seed in SEEDS:
                metrics = run_trained(seed, d_z, pairing, source)
                records.append(metrics)
                done += 1
                elapsed = time.time() - start
                print(
                    f"[{done}/{total}] d_z={d_z} pairing={PAIRING_LABELS[pairing]} seed={seed} "
                    f"Q_E(raw)={metrics['raw_q_entity']:.3g} "
                    f"Q_E(white)={metrics['white_q_entity']:.3g} elapsed={elapsed:.0f}s",
                    flush=True,
                )

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    with open(f"{OUTPUT_ROOT}/records.json", "w") as f:
        json.dump(records, f, indent=2)
    elapsed_total = time.time() - start
    print(f"\nwrote {len(records)} records to {OUTPUT_ROOT}/records.json in {elapsed_total:.0f}s")


if __name__ == "__main__":
    main()
