#!/usr/bin/env python3
"""Explicit launcher for the full Section 8 experimental grid.

Never invoked implicitly by ``train_temporal.py``. Each cell is trained in
isolation (a failure is recorded, not fatal to its peers), matching the base
repository's sweep semantics in :mod:`jepa.reporting`.

    python scripts/timeseries/run_grid.py --preset quick
    python scripts/timeseries/run_grid.py --preset full
    python scripts/timeseries/run_grid.py --preset full_bottleneck

``full_bottleneck`` is the official-pipeline version of the ad-hoc
``scripts/investigations/timeseries_pairing_controls.py`` bottleneck check: it sweeps
``model.latent_dim`` across (and past) the world's true signal dimensionality,
plus ``context_rho`` and the SG/EMA policy, so those checks no longer live
only in a throwaway script.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import yaml  # noqa: E402

from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.timeseries import load_temporal_config  # noqa: E402
from jepa.training.timeseries import train_temporal_experiment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATE_ENTITY_SUBSPACE_DIMS = (1, 2, 4, 8, 16, 32)


def _replicate_seed_overrides(seed: int) -> dict[str, str]:
    """Vary every stochastic owner (data + training) with one replicate seed."""
    return {
        "training.seed": str(derive_seed(seed, "temporal-training")),
        "data.system_seed": str(derive_seed(seed, "temporal-system")),
        "data.train_sample_seed": str(derive_seed(seed, "temporal-samples", "train")),
        "data.validation_sample_seed": str(derive_seed(seed, "temporal-samples", "validation")),
        "data.test_sample_seed": str(derive_seed(seed, "temporal-samples", "test")),
        "data.counterfactual_seed": str(derive_seed(seed, "temporal-samples", "counterfactual")),
    }


def _load_grid_spec(preset: str) -> dict[str, Any]:
    path = REPO_ROOT / "configs" / "temporal_grids" / f"{preset}.yaml"
    return yaml.safe_load(path.read_text())


def _standard_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    # These three axes are optional and backward compatible: omitting them from
    # a grid yaml reproduces the exact old behavior (no latent_dim/context_rho
    # override, single default SG=on/EMA=on cell). Add them to actually sweep
    # the bottleneck dimension, the entity/context timescale ratio, or the
    # SG/EMA policy -- each was previously only exercised in ad-hoc
    # `_investigate_*.py` scripts, never in the official grid.
    latent_dims = spec.get("latent_dims", [None])
    context_rhos = spec.get("context_rhos", [None])
    sg_ema_combinations = spec.get("sg_ema_combinations", [[True, True]])

    axes = list(
        itertools.product(
            spec["seeds"],
            spec["model_types"],
            spec["observation_modes"],
            spec["entity_switch_probabilities"],
            spec["prediction_horizons"],
            spec["target_pairings"],
            latent_dims,
            context_rhos,
            sg_ema_combinations,
        )
    )
    cells = []
    for (
        seed,
        architecture,
        observation_mode,
        p_switch,
        horizon,
        pairing,
        latent_dim,
        context_rho,
        sg_ema,
    ) in axes:
        stop_gradient, ema_enabled = sg_ema
        overrides = {
            **_replicate_seed_overrides(seed),
            "model.architecture": architecture,
            "model.kind": "standard",
            "data.observation_mode": observation_mode,
            "data.entity_switch_probability": str(p_switch),
            "training.prediction_horizon": str(horizon),
            "training.target_pairing": pairing,
            "training.stop_gradient": "on" if stop_gradient else "off",
            "training.ema.enabled": "on" if ema_enabled else "off",
        }
        if latent_dim is not None:
            overrides["model.latent_dim"] = str(latent_dim)
            # entity_subspace_dims defaults to values tuned for the base config's
            # own latent_dim (e.g. [1, 2, 4, 8]); sweeping latent_dim smaller than
            # the largest of those breaks the "dims must not exceed latent_dim"
            # config check, so clip to what's still valid for this cell.
            valid_dims = [d for d in _CANDIDATE_ENTITY_SUBSPACE_DIMS if d <= latent_dim]
            overrides["evaluation.entity_subspace_dims"] = str(valid_dims or [latent_dim])
        if context_rho is not None:
            overrides["data.context_rho"] = str(context_rho)
        cells.append({"kind": "standard", "seed": seed, "overrides": overrides})
    return cells


def _random_baseline_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if not spec.get("include_random_baseline", False):
        return []
    latent_dims = spec.get("latent_dims", [None])
    cells = []
    for seed, architecture, observation_mode, latent_dim in itertools.product(
        spec["seeds"], spec["model_types"], spec["observation_modes"], latent_dims
    ):
        overrides = {
            **_replicate_seed_overrides(seed),
            "model.architecture": architecture,
            "model.kind": "random",
            "data.observation_mode": observation_mode,
        }
        if latent_dim is not None:
            overrides["model.latent_dim"] = str(latent_dim)
            valid_dims = [d for d in _CANDIDATE_ENTITY_SUBSPACE_DIMS if d <= latent_dim]
            overrides["evaluation.entity_subspace_dims"] = str(valid_dims or [latent_dim])
        cells.append({"kind": "random", "seed": seed, "overrides": overrides})
    return cells


def _hierarchical_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    hierarchical_spec = spec.get("hierarchical")
    if not hierarchical_spec:
        return []
    cells = []
    for seed, architecture, p_switch in itertools.product(
        spec["seeds"], spec["model_types"], hierarchical_spec["entity_switch_probabilities"]
    ):
        cells.append(
            {
                "kind": "hierarchical",
                "seed": seed,
                "base_config": hierarchical_spec["base_config"],
                "overrides": {
                    **_replicate_seed_overrides(seed),
                    "model.architecture": architecture,
                    "model.kind": "hierarchical",
                    "data.entity_switch_probability": str(p_switch),
                    "hierarchical.short_horizon": str(hierarchical_spec["short_horizon"]),
                    "hierarchical.long_horizon": str(hierarchical_spec["long_horizon"]),
                },
            }
        )
    return cells


def run_grid(preset: str, *, output_root: str | None, overwrite: bool) -> dict[str, Any]:
    spec = _load_grid_spec(preset)
    base_config_path = REPO_ROOT / spec["base_config"]
    scale_overrides = {
        key: spec[key]
        for key in (
            "num_train_trajectories",
            "num_val_trajectories",
            "num_test_trajectories",
            "epochs",
        )
        if key in spec
    }

    grid_root = Path(output_root or f"outputs/timeseries/grids/{preset}").expanduser().resolve()
    grid_root.mkdir(parents=True, exist_ok=True)

    cells = _standard_cells(spec) + _random_baseline_cells(spec) + _hierarchical_cells(spec)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        config_path = REPO_ROOT / cell.get("base_config", spec["base_config"])
        overrides = dict(cell["overrides"])
        if config_path == base_config_path:
            for key, value in scale_overrides.items():
                dotted = f"data.{key}" if key != "epochs" else "training.epochs"
                overrides.setdefault(dotted, str(value))
        overrides["output.root"] = str(grid_root)
        overrides["output.overwrite"] = "on" if overwrite else "off"
        try:
            config = load_temporal_config(config_path, overrides=overrides)
            result = train_temporal_experiment(config, seed_label=cell["seed"])
            rows.append(
                {
                    "cell_kind": cell["kind"],
                    "seed": cell["seed"],
                    "state": "complete",
                    "run_id": result.run_id,
                    "run_dir": str(result.run_dir),
                    "overrides": cell["overrides"],
                }
            )
        except Exception as error:  # noqa: BLE001 - isolate cell failures, matching jepa.reporting
            rows.append(
                {
                    "cell_kind": cell["kind"],
                    "seed": cell["seed"],
                    "state": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "overrides": cell["overrides"],
                }
            )

    completed = sum(row["state"] == "complete" for row in rows)
    summary = {
        "preset": preset,
        "cell_count": len(rows),
        "completed_count": completed,
        "failed_count": len(rows) - completed,
        "rows": rows,
    }
    (grid_root / "grid_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "full", "full_bottleneck"), required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = run_grid(args.preset, output_root=args.output_root, overwrite=args.overwrite)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
