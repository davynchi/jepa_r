#!/usr/bin/env python3
"""Explicit launcher for the full Section 8 experimental grid.

Never invoked implicitly by ``train_temporal.py``. Each cell is trained in
isolation (a failure is recorded, not fatal to its peers), matching the base
repository's sweep semantics in :mod:`jepa.reporting`.

    python scripts/run_temporal_grid.py --preset quick
    python scripts/run_temporal_grid.py --preset full
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from jepa.config import derive_seed  # noqa: E402
from jepa.temporal_config import load_temporal_config  # noqa: E402
from jepa.temporal_training import train_temporal_experiment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    axes = list(
        itertools.product(
            spec["seeds"],
            spec["model_types"],
            spec["observation_modes"],
            spec["entity_switch_probabilities"],
            spec["prediction_horizons"],
            spec["target_pairings"],
        )
    )
    cells = []
    for seed, architecture, observation_mode, p_switch, horizon, pairing in axes:
        cells.append(
            {
                "kind": "standard",
                "seed": seed,
                "overrides": {
                    **_replicate_seed_overrides(seed),
                    "model.architecture": architecture,
                    "model.kind": "standard",
                    "data.observation_mode": observation_mode,
                    "data.entity_switch_probability": str(p_switch),
                    "training.prediction_horizon": str(horizon),
                    "training.target_pairing": pairing,
                },
            }
        )
    return cells


def _random_baseline_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if not spec.get("include_random_baseline", False):
        return []
    cells = []
    for seed, architecture, observation_mode in itertools.product(
        spec["seeds"], spec["model_types"], spec["observation_modes"]
    ):
        cells.append(
            {
                "kind": "random",
                "seed": seed,
                "overrides": {
                    **_replicate_seed_overrides(seed),
                    "model.architecture": architecture,
                    "model.kind": "random",
                    "data.observation_mode": observation_mode,
                },
            }
        )
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

    grid_root = Path(output_root or f"outputs/temporal_grid_{preset}").expanduser().resolve()
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
    parser.add_argument("--preset", choices=("quick", "full"), required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = run_grid(args.preset, output_root=args.output_root, overwrite=args.overwrite)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
