#!/usr/bin/env python3
"""Aggregate matched RAS utility audits across virtual step multipliers."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-utility-key",
        default="negative_delta_probe_loss",
    )
    parser.add_argument(
        "--ras-key",
        default="ras_barlow_adamw_cosine",
    )
    parser.add_argument("--max-cosine-drift", type=float, default=0.1)
    return parser.parse_args()


def rankdata(values: list[float]) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64)
    order = torch.argsort(tensor)
    ranks = torch.empty_like(tensor)
    sorted_values = tensor[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def correlation(left: list[float], right: list[float], *, ranks: bool = False) -> float:
    if ranks:
        x = rankdata(left)
        y = rankdata(right)
    else:
        x = torch.tensor(left, dtype=torch.float64)
        y = torch.tensor(right, dtype=torch.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = x.norm() * y.norm()
    if denominator <= torch.finfo(torch.float64).eps:
        return float("nan")
    return float((x @ y / denominator).item())


def mean_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def load_records(paths: list[Path]) -> list[dict[str, float]]:
    records = []
    for path in paths:
        records_path = path.expanduser().resolve() / "records.jsonl"
        if not records_path.exists():
            raise FileNotFoundError(records_path)
        for line in records_path.read_text().splitlines():
            record = json.loads(line)
            if "step_multiplier" not in record:
                raise ValueError(f"{records_path} was produced without --step-multiplier")
            records.append(record)
    return records


def main() -> None:
    args = parse_args()
    if args.max_cosine_drift <= 0:
        raise ValueError("--max-cosine-drift must be positive")
    records = load_records(args.audit_dirs)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    by_trial: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    by_multiplier: dict[float, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        key = (int(record["candidate"]), int(record.get("mask_repeat", 0)))
        by_trial[key].append(record)
        by_multiplier[float(record["step_multiplier"])].append(record)

    multipliers = sorted(by_multiplier)
    expected = set(multipliers)
    for key, rows in by_trial.items():
        observed = {float(row["step_multiplier"]) for row in rows}
        if observed != expected:
            raise ValueError(
                f"trial {key} has multipliers {sorted(observed)}, expected {multipliers}"
            )

    metric_keys = (
        args.probe_utility_key,
        "jepa_loss_improvement",
        "representation_cosine_drift",
    )
    richness_keys = sorted(
        key for key in records[0] if key.startswith("delta_richness_")
    )
    curves = {}
    for multiplier in multipliers:
        curves[str(multiplier)] = {}
        for key in (*metric_keys, *richness_keys):
            values = [float(row[key]) for row in by_multiplier[multiplier]]
            mean, std = mean_std(values)
            curves[str(multiplier)][key] = {"mean": mean, "std": std}

    trial_summaries = []
    for (candidate, mask_repeat), rows in sorted(by_trial.items()):
        best = max(rows, key=lambda row: float(row[args.probe_utility_key]))
        jepa_stable = [
            row for row in rows if float(row["jepa_loss_improvement"]) >= 0
        ]
        trust_region_stable = [
            row
            for row in jepa_stable
            if float(row["representation_cosine_drift"]) <= args.max_cosine_drift
        ]
        baseline = min(rows, key=lambda row: abs(float(row["step_multiplier"]) - 1.0))
        trial_summaries.append(
            {
                "candidate": candidate,
                "mask_repeat": mask_repeat,
                "ras": float(baseline[args.ras_key]),
                "best_multiplier": float(best["step_multiplier"]),
                "best_probe_utility": float(best[args.probe_utility_key]),
                "oracle_gap_over_1x": float(best[args.probe_utility_key])
                - float(baseline[args.probe_utility_key]),
                "largest_jepa_stable_multiplier": (
                    max(float(row["step_multiplier"]) for row in jepa_stable)
                    if jepa_stable
                    else 0.0
                ),
                "largest_trust_region_multiplier": (
                    max(float(row["step_multiplier"]) for row in trust_region_stable)
                    if trust_region_stable
                    else 0.0
                ),
            }
        )

    ras = [row["ras"] for row in trial_summaries]
    best_multiplier = [math.log2(row["best_multiplier"]) for row in trial_summaries]
    stable_multiplier = [
        math.log2(max(row["largest_trust_region_multiplier"], min(multipliers)))
        for row in trial_summaries
    ]
    summary = {
        "multipliers": multipliers,
        "num_trials": len(trial_summaries),
        "probe_utility_key": args.probe_utility_key,
        "ras_key": args.ras_key,
        "max_cosine_drift": args.max_cosine_drift,
        "curves": curves,
        "ras_correlations": {
            "pearson_log2_best_multiplier": correlation(ras, best_multiplier),
            "spearman_best_multiplier": correlation(ras, best_multiplier, ranks=True),
            "pearson_log2_largest_trust_region_multiplier": correlation(
                ras, stable_multiplier
            ),
            "spearman_largest_trust_region_multiplier": correlation(
                ras, stable_multiplier, ranks=True
            ),
            "pearson_oracle_gap": correlation(
                ras,
                [row["oracle_gap_over_1x"] for row in trial_summaries],
            ),
            "spearman_oracle_gap": correlation(
                ras,
                [row["oracle_gap_over_1x"] for row in trial_summaries],
                ranks=True,
            ),
        },
        "trials": trial_summaries,
    }
    (output_dir / "step_size_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = (
        (axes[0, 0], args.probe_utility_key, "Frozen-probe utility"),
        (axes[0, 1], "jepa_loss_improvement", "JEPA loss improvement"),
        (axes[1, 0], "representation_cosine_drift", "Representation cosine drift"),
        (
            axes[1, 1],
            richness_keys[0],
            f"Realized {richness_keys[0].removeprefix('delta_richness_')} ΔR",
        ),
    )
    for axis, key, title in panels:
        means = [curves[str(multiplier)][key]["mean"] for multiplier in multipliers]
        stds = [curves[str(multiplier)][key]["std"] for multiplier in multipliers]
        axis.errorbar(multipliers, means, yerr=stds, marker="o", capsize=3)
        axis.axhline(0, color="black", linewidth=1, alpha=0.4)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("AdamW step multiplier")
        axis.set_ylabel(key)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "step_size_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary["ras_correlations"], indent=2))
    print(f"wrote {output_dir / 'step_size_summary.json'}")
    print(f"wrote {output_dir / 'step_size_grid.png'}")


if __name__ == "__main__":
    main()
