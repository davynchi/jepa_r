#!/usr/bin/env python3
"""Analyze terminal Shapes3D quality/accuracy associations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from jepa.analysis.quality_metrics import METRIC_SPEC_BY_NAME  # noqa: E402
from jepa.analysis.quality_statistics import (  # noqa: E402
    TerminalObservation,
    benjamini_hochberg,
    block_bootstrap_interval,
    block_permutation_pvalue,
    complete_seed_blocks,
    curriculum_partial_spearman,
    pearson,
    seed_block_partial_spearman,
)

CURRICULA = (
    "uniform",
    "loss",
    "ras-logdet",
    "coord-covariance",
    "coord-transformation",
    "coord-dynamics",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=91_117)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _terminal_rows(root: Path) -> dict[tuple[str, str, str], list[TerminalObservation]]:
    accuracy = {
        (row["run_id"], row["checkpoint_id"]): float(row["accuracy"])
        for row in _jsonl(root / "accuracy.jsonl")
    }
    candidates: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for row in _jsonl(root / "records.jsonl"):
        if row["state"] != "complete" or row["value"] is None:
            continue
        diagnostics = json.loads(row["diagnostics_json"])
        key = (str(diagnostics.get("phase", "unspecified")), row["panel"], row["metric_name"])
        candidates[key].append((int(diagnostics["global_step"]), {**row, **diagnostics}))
    grouped: dict[tuple[str, str, str], list[TerminalObservation]] = defaultdict(list)
    for key, rows in candidates.items():
        by_run: dict[str, tuple[int, dict]] = {}
        for step, row in rows:
            if row["run_id"] not in by_run or step > by_run[row["run_id"]][0]:
                by_run[row["run_id"]] = (step, row)
        for _, row in by_run.values():
            accuracy_value = accuracy.get((row["run_id"], row["checkpoint_id"]))
            if accuracy_value is None:
                continue
            grouped[key].append(
                TerminalObservation(
                    row["run_id"],
                    int(row["seed_block"]),
                    str(row["curriculum"]),
                    float(row["value"]),
                    accuracy_value,
                )
            )
    return grouped


def analyze(root: Path, *, bootstrap: int, permutations: int, seed: int) -> list[dict]:
    results: list[dict] = []
    for index, ((phase, panel, metric), observations) in enumerate(
        sorted(_terminal_rows(root).items())
    ):
        complete = complete_seed_blocks(observations, expected_curricula=CURRICULA)
        if len({row.seed_block for row in complete}) < 4:
            results.append(
                {
                    "panel": panel,
                    "phase": phase,
                    "metric_name": metric,
                    "n": len(complete),
                    "null_reason": "not_testable_data",
                    "p_value": 1.0,
                }
            )
            continue
        primary = seed_block_partial_spearman(complete)
        secondary = curriculum_partial_spearman(complete)
        interval = block_bootstrap_interval(complete, iterations=bootstrap, seed=seed + index)
        pvalue = block_permutation_pvalue(
            complete, iterations=permutations, seed=seed + 10_000 + index
        )
        results.append(
            {
                "panel": panel,
                "phase": phase,
                "metric_name": metric,
                "n": len(complete),
                "complete_blocks": len({row.seed_block for row in complete}),
                "seed_block_partial_spearman": primary,
                "curriculum_partial_spearman": secondary,
                "pearson": pearson(
                    [row.metric for row in complete], [row.accuracy for row in complete]
                ),
                "ci_low": interval[0],
                "ci_high": interval[1],
                "p_value": pvalue,
                "null_reason": None,
            }
        )
    present_phases = {
        str(json.loads(row["diagnostics_json"]).get("phase", "unspecified"))
        for row in _jsonl(root / "records.jsonl")
    }
    primary_names = {name for name, spec in METRIC_SPEC_BY_NAME.items() if spec.primary_eligible}
    existing = {(result["phase"], result["panel"], result["metric_name"]) for result in results}
    for phase in present_phases.intersection({"discovery", "replication"}):
        for metric in sorted(primary_names):
            if (phase, "label_free", metric) not in existing:
                results.append(
                    {
                        "phase": phase,
                        "panel": "label_free",
                        "metric_name": metric,
                        "n": 0,
                        "null_reason": "not_testable_data",
                        "p_value": 1.0,
                    }
                )
    for phase in ("discovery", "replication"):
        label_free_indices = [
            index
            for index, result in enumerate(results)
            if result["phase"] == phase
            and result["panel"] == "label_free"
            and METRIC_SPEC_BY_NAME[result["metric_name"]].primary_eligible
        ]
        adjusted = benjamini_hochberg(
            [float(results[index].get("p_value", 1.0)) for index in label_free_indices]
        )
        for index, value in zip(label_free_indices, adjusted, strict=True):
            results[index]["q_value"] = value
    return results


def _write(root: Path, results: list[dict]) -> None:
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "correlations.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in results for key in row})
    with (output / "correlations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    valid = [row for row in results if row.get("seed_block_partial_spearman") is not None]
    if valid:
        figure, axis = plt.subplots(figsize=(10, max(4, len(valid) * 0.35)))
        axis.barh(
            [f"{row['panel']}:{row['metric_name']}" for row in valid],
            [row["seed_block_partial_spearman"] for row in valid],
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("seed-block-partial Spearman rho")
        figure.tight_layout()
        figure.savefig(output / "terminal_correlations.png", dpi=160)
        plt.close(figure)


def main() -> None:
    arguments = _args()
    root = Path(arguments.quality_root).expanduser().resolve()
    results = analyze(
        root,
        bootstrap=arguments.bootstrap_iterations,
        permutations=arguments.permutation_iterations,
        seed=arguments.seed,
    )
    _write(root, results)
    print(f"metrics={len(results)} analysis={root / 'analysis'}")


if __name__ == "__main__":
    main()
