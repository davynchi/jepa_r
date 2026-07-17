#!/usr/bin/env python3
"""Aggregate evaluated runs under a temporal-persistence grid directory.

    python scripts/analysis/aggregate_temporal.py --root outputs/temporal_grid_full

Walks every run directory under ``--root`` (as produced by
``timeseries/run_grid.py``), runs ``timeseries/evaluate.py`` on any run
that has not been analyzed yet, aggregates across seeds, writes
``aggregate_summary.csv`` / ``aggregate_summary.json``, and produces the two
cross-run plots (``entity_selectivity_vs_timescale.png``,
``temporal_vs_shuffled.png``) plus a text answer sheet for Section 14.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# evaluate_run lives in the sibling timeseries/ script directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "timeseries"))

from evaluate import evaluate_run  # noqa: E402

from jepa.analysis.plots import (  # noqa: E402
    plot_entity_selectivity_vs_timescale,
    plot_temporal_vs_shuffled,
)

# Fields that identify "the same condition modulo target_pairing" -- used to
# build matched temporal/shuffled pairs rather than pooling every run of each
# pairing regardless of seed, p_E, horizon, architecture, etc.
_MATCH_FIELDS = (
    "seed",
    "model_kind",
    "architecture",
    "observation_mode",
    "p_entity_switch",
    "prediction_horizon",
    "latent_dim",
    "context_rho",
    "stop_gradient",
    "ema_enabled",
)


def _iter_run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/checkpoint.pt"))


def _load_or_compute_summary(run_dir: Path, *, force: bool) -> dict[str, Any]:
    summary_path = run_dir / "analysis" / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())
    return evaluate_run(run_dir)


def _probe_row(summary: dict[str, Any], representation: str) -> dict[str, Any]:
    return next(r for r in summary["probe_matrix"] if r["representation"] == representation)


def aggregate(root: Path, *, force: bool) -> dict[str, Any]:
    run_dirs = _iter_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(f"no run directories with a checkpoint.pt found under {root}")

    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = _load_or_compute_summary(run_dir, force=force)
        config = summary["config"]
        full_row = _probe_row(summary, "full_z")
        entity_row = _probe_row(summary, "entity_subspace")
        complement_row = _probe_row(summary, "complement")
        whitened = summary.get("counterfactual_whitened", {})
        spectrum = summary.get("spectrum", {})
        record = {
            "run_id": summary["run_id"],
            "run_dir": str(run_dir),
            "model_kind": summary["model_kind"],
            "architecture": config["model"]["architecture"],
            "observation_mode": config["data"]["observation_mode"],
            "p_entity_switch": config["data"]["entity_switch_probability"],
            "prediction_horizon": config["training"]["prediction_horizon"],
            "target_pairing": config["training"]["target_pairing"],
            "seed": config["training"]["seed"],
            "latent_dim": config["model"]["latent_dim"],
            "context_rho": config["data"]["context_rho"],
            "stop_gradient": config["training"]["stop_gradient"],
            "ema_enabled": config["training"]["ema"]["enabled"],
            "predictability": summary.get("predictability"),
            "q_entity": summary["counterfactual"]["q_entity"],
            "d_same": summary["counterfactual"]["d_same"],
            "d_diff": summary["counterfactual"]["d_diff"],
            "q_entity_whitened": whitened.get("q_entity"),
            "d_same_whitened": whitened.get("d_same"),
            "d_diff_whitened": whitened.get("d_diff"),
            "trace_covariance": spectrum.get("trace_covariance"),
            "mean_latent_norm": summary.get("mean_latent_norm"),
            # NOTE: "entity_accuracy"/"context_leakage" here are the
            # *entity-subspace* (P_E z) probe, used for the selectivity plots.
            # Do not read these as "full-latent" accuracy -- that is
            # `full_z_entity_accuracy` below. Mixing the two up was a real bug
            # in an earlier version of this script's answer sheet.
            "entity_accuracy": entity_row["entity_accuracy"],
            "context_leakage": entity_row["context_r2"],
            "full_z_entity_accuracy": full_row["entity_accuracy"],
            "full_z_context_r2": full_row["context_r2"],
            "complement_entity_accuracy": complement_row["entity_accuracy"],
            "complement_context_r2": complement_row["context_r2"],
            "selectivity": summary["selectivity_score"],
            "selected_entity_subspace_dim": summary["selected_entity_subspace_dim"],
            "effective_rank": summary["collapse_diagnostics"]["effective_rank"],
            "collapsed": summary["collapse_diagnostics"]["collapsed"],
            "test_loss": summary.get("test_loss"),
        }
        records.append(record)

    root.mkdir(parents=True, exist_ok=True)
    columns = list(records[0].keys())
    with (root / "aggregate_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    (root / "aggregate_summary.json").write_text(json.dumps(records, indent=2, sort_keys=True))

    selectivity_records = [
        {
            "p_entity_switch": r["p_entity_switch"],
            "seed": r["seed"],
            "q_entity": r["q_entity"],
            "entity_accuracy": r["entity_accuracy"],
            "context_leakage": r["context_leakage"],
            "selectivity": r["selectivity"],
        }
        for r in records
        if r["model_kind"] == "standard" and r["target_pairing"] == "temporal"
    ]
    if selectivity_records:
        plot_entity_selectivity_vs_timescale(
            selectivity_records, root / "entity_selectivity_vs_timescale.png"
        )

    paired = _paired_temporal_shuffled(records)
    if paired:
        import numpy as np

        control_records = []
        for metric in ("q_entity", "q_entity_whitened", "entity_accuracy", "selectivity"):
            temporal_values = [
                p["temporal"][metric] for p in paired if p["temporal"][metric] is not None
            ]
            shuffled_values = [
                p["shuffled"][metric] for p in paired if p["shuffled"][metric] is not None
            ]
            if not temporal_values or not shuffled_values:
                continue
            control_records.append(
                {
                    "metric": metric,
                    "temporal_mean": float(np.mean(temporal_values)),
                    "temporal_std": float(np.std(temporal_values)),
                    "shuffled_mean": float(np.mean(shuffled_values)),
                    "shuffled_std": float(np.std(shuffled_values)),
                }
            )
        if control_records:
            plot_temporal_vs_shuffled(control_records, root / "temporal_vs_shuffled.png")

    answers = _answer_sheet(records, paired)
    (root / "scientific_summary.md").write_text(answers)
    return {"run_count": len(records), "paired_count": len(paired), "root": str(root)}


def _paired_temporal_shuffled(records: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    """Match each temporal run to its shuffled counterpart on every axis except
    ``target_pairing`` itself, so temporal-vs-shuffled comparisons are always
    apples-to-apples (same seed, p_E, horizon, architecture, ...) instead of
    pooling means across mismatched conditions."""
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for r in records:
        if r["target_pairing"] not in ("temporal", "shuffled"):
            continue
        key = tuple(r[field] for field in _MATCH_FIELDS)
        by_key.setdefault(key, {})[r["target_pairing"]] = r
    return [pair for pair in by_key.values() if "temporal" in pair and "shuffled" in pair]


def _answer_sheet(records: list[dict[str, Any]], paired: list[dict[str, dict[str, Any]]]) -> str:
    import numpy as np

    def _mean(key: str, predicate) -> float | None:
        values = [r[key] for r in records if predicate(r) and r[key] is not None]
        return float(np.mean(values)) if values else None

    lines = ["# Section 14 scientific comparisons\n"]

    full_entity_acc = _mean("full_z_entity_accuracy", lambda r: r["target_pairing"] == "temporal")
    lines.append(
        f"1. Full-latent (z) entity accuracy (temporal, mean over runs): {full_entity_acc}"
    )

    subspace_entity_acc = _mean("entity_accuracy", lambda r: r["target_pairing"] == "temporal")
    complement_entity_acc = _mean(
        "complement_entity_accuracy", lambda r: r["target_pairing"] == "temporal"
    )
    lines.append(
        f"1b. Entity-subspace vs. complement accuracy (temporal): "
        f"P_E z={subspace_entity_acc}, (I-P_E) z={complement_entity_acc} "
        f"(concentration evidence requires the subspace value to clearly exceed the complement, "
        f"not just exceed chance)"
    )

    by_p = sorted({r["p_entity_switch"] for r in records})
    q_by_p = {
        p: _mean(
            "q_entity_whitened",
            lambda r, p=p: r["p_entity_switch"] == p and r["target_pairing"] == "temporal",
        )
        for p in by_p
    }
    lines.append(
        f"2/3. Whitened Q_E by entity switch probability (higher p_E = faster switching): {q_by_p}"
    )
    monotone = all(
        q_by_p[by_p[i]] is None
        or q_by_p[by_p[i + 1]] is None
        or q_by_p[by_p[i]] >= q_by_p[by_p[i + 1]]
        for i in range(len(by_p) - 1)
    )
    lines.append(
        f"4. Selectivity increases as entity timescale lengthens (p_E decreases): {monotone} "
        f"(judged on whitened Q_E, which controls for raw-scale confounds)"
    )

    by_h = sorted({r["prediction_horizon"] for r in records})
    q_by_h = {
        h: _mean(
            "selectivity",
            lambda r, h=h: r["prediction_horizon"] == h and r["target_pairing"] == "temporal",
        )
        for h in by_h
    }
    lines.append(f"5. Selectivity by prediction horizon (temporal only): {q_by_h}")

    if paired:
        deltas = {
            metric: [
                p["temporal"][metric] - p["shuffled"][metric]
                for p in paired
                if p["temporal"][metric] is not None and p["shuffled"][metric] is not None
            ]
            for metric in ("q_entity_whitened", "entity_accuracy", "selectivity")
        }
        delta_summary = {
            metric: (float(np.mean(vals)), float(np.std(vals))) if vals else None
            for metric, vals in deltas.items()
        }
        lines.append(
            f"6. Paired temporal-minus-shuffled deltas over {len(paired)} matched "
            f"(seed, architecture, observation_mode, p_E, horizon) cells "
            f"(mean, std): {delta_summary}"
        )
    else:
        temporal_q = _mean("q_entity_whitened", lambda r: r["target_pairing"] == "temporal")
        shuffled_q = _mean("q_entity_whitened", lambda r: r["target_pairing"] == "shuffled")
        lines.append(
            f"6. Shuffled-target control whitened Q_E (UNPAIRED -- no matching cells found): "
            f"temporal={temporal_q}, shuffled={shuffled_q}"
        )

    linear_sel = _mean(
        "selectivity", lambda r: r["architecture"] == "linear" and r["target_pairing"] == "temporal"
    )
    nonlinear_sel = _mean(
        "selectivity",
        lambda r: r["architecture"] == "nonlinear" and r["target_pairing"] == "temporal",
    )
    lines.append(
        f"7. Linear vs nonlinear JEPA selectivity (temporal only): "
        f"linear={linear_sel}, nonlinear={nonlinear_sel}"
    )

    hier_sel = _mean("selectivity", lambda r: r["model_kind"] == "hierarchical")
    std_sel = _mean(
        "selectivity", lambda r: r["model_kind"] == "standard" and r["target_pairing"] == "temporal"
    )
    lines.append(
        f"8. Explicit entity/context model selectivity vs standard: "
        f"hierarchical={hier_sel}, standard={std_sel}"
    )

    collapsed_fraction = float(np.mean([1.0 if r["collapsed"] else 0.0 for r in records]))
    lines.append(
        f"9. Fraction of runs flagged as collapsed by representation diagnostics: "
        f"{collapsed_fraction}"
    )

    seed_counts: dict[tuple[Any, ...], int] = {}
    for r in records:
        key = (r["model_kind"], r["architecture"], r["p_entity_switch"], r["target_pairing"])
        seed_counts[key] = seed_counts.get(key, 0) + 1
    lines.append(f"10. Seeds per condition (condition -> count): {seed_counts}")

    return "\n".join(str(line) for line in lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="re-run analysis even if cached")
    args = parser.parse_args(argv)
    result = aggregate(args.root, force=args.force)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
