#!/usr/bin/env python3
"""Aggregate evaluated runs under a temporal-persistence grid directory.

    python scripts/aggregate_temporal_hierarchy.py --root outputs/temporal_grid_full

Walks every run directory under ``--root`` (as produced by
``run_temporal_grid.py``), runs ``evaluate_temporal_hierarchy`` on any run
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_temporal_hierarchy import evaluate_run  # noqa: E402

from jepa.temporal_plots import (  # noqa: E402
    plot_entity_selectivity_vs_timescale,
    plot_temporal_vs_shuffled,
)


def _iter_run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/checkpoint.pt"))


def _load_or_compute_summary(run_dir: Path, *, force: bool) -> dict[str, Any]:
    summary_path = run_dir / "analysis" / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())
    return evaluate_run(run_dir)


def aggregate(root: Path, *, force: bool) -> dict[str, Any]:
    run_dirs = _iter_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(f"no run directories with a checkpoint.pt found under {root}")

    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = _load_or_compute_summary(run_dir, force=force)
        config = summary["config"]
        entity_row = next(
            r for r in summary["probe_matrix"] if r["representation"] == "entity_subspace"
        )
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
            "q_entity": summary["counterfactual"]["q_entity"],
            "d_same": summary["counterfactual"]["d_same"],
            "d_diff": summary["counterfactual"]["d_diff"],
            "entity_accuracy": entity_row["entity_accuracy"],
            "context_leakage": entity_row["context_r2"],
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

    shuffled_rows = [r for r in records if r["target_pairing"] == "shuffled"]
    if shuffled_rows:
        import numpy as np

        control_records = []
        for metric in ("q_entity", "entity_accuracy", "selectivity"):
            temporal_values = [r[metric] for r in records if r["target_pairing"] == "temporal"]
            shuffled_values = [r[metric] for r in records if r["target_pairing"] == "shuffled"]
            control_records.append(
                {
                    "metric": metric,
                    "temporal_mean": float(np.mean(temporal_values)),
                    "temporal_std": float(np.std(temporal_values)),
                    "shuffled_mean": float(np.mean(shuffled_values)),
                    "shuffled_std": float(np.std(shuffled_values)),
                }
            )
        plot_temporal_vs_shuffled(control_records, root / "temporal_vs_shuffled.png")

    answers = _answer_sheet(records)
    (root / "scientific_summary.md").write_text(answers)
    return {"run_count": len(records), "root": str(root)}


def _answer_sheet(records: list[dict[str, Any]]) -> str:
    import numpy as np

    def _mean(key: str, predicate) -> float | None:
        values = [r[key] for r in records if predicate(r)]
        return float(np.mean(values)) if values else None

    lines = ["# Section 14 scientific comparisons\n"]
    full_entity_acc = _mean("entity_accuracy", lambda r: r["target_pairing"] == "temporal")
    lines.append(f"1. Full-latent entity accuracy (temporal, mean over runs): {full_entity_acc}")
    by_p = sorted({r["p_entity_switch"] for r in records})
    q_by_p = {
        p: _mean(
            "q_entity",
            lambda r, p=p: r["p_entity_switch"] == p and r["target_pairing"] == "temporal",
        )
        for p in by_p
    }
    lines.append(
        f"2/3. Q_E by entity switch probability (higher p_E = faster entity switching): {q_by_p}"
    )
    monotone = all(
        q_by_p[by_p[i]] is None
        or q_by_p[by_p[i + 1]] is None
        or q_by_p[by_p[i]] >= q_by_p[by_p[i + 1]]
        for i in range(len(by_p) - 1)
    )
    lines.append(
        f"4. Selectivity increases as entity timescale lengthens (p_E decreases): {monotone}"
    )
    by_h = sorted({r["prediction_horizon"] for r in records})
    q_by_h = {h: _mean("selectivity", lambda r, h=h: r["prediction_horizon"] == h) for h in by_h}
    lines.append(f"5. Selectivity by prediction horizon: {q_by_h}")
    temporal_q = _mean("q_entity", lambda r: r["target_pairing"] == "temporal")
    shuffled_q = _mean("q_entity", lambda r: r["target_pairing"] == "shuffled")
    lines.append(f"6. Shuffled-target control Q_E: temporal={temporal_q}, shuffled={shuffled_q}")
    linear_sel = _mean("selectivity", lambda r: r["architecture"] == "linear")
    nonlinear_sel = _mean("selectivity", lambda r: r["architecture"] == "nonlinear")
    lines.append(
        f"7. Linear vs nonlinear JEPA selectivity: linear={linear_sel}, nonlinear={nonlinear_sel}"
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
    seed_counts = {}
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
