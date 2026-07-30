#!/usr/bin/env python3
"""Correlate factorization-v2 metrics with aligned Tiny ImageNet probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorization-records", type=Path, required=True)
    parser.add_argument(
        "--probe-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Probe results root (epoch_*/results.json) or training metrics.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--minimum-points", type=int, default=4)
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, *, spearman: bool) -> float:
    if spearman:
        left, right = rankdata(left), rankdata(right)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    spearman: bool,
    samples: int,
    generator: np.random.Generator,
) -> tuple[float, float]:
    if len(left) < 3:
        return float("nan"), float("nan")
    estimates = []
    for _ in range(samples):
        indices = generator.integers(0, len(left), size=len(left))
        value = correlation(left[indices], right[indices], spearman=spearman)
        if math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def load_factorization(path: Path) -> dict[str, dict[int, float]]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        series: dict[str, dict[int, float]] = {}
        for line in text.splitlines():
            record = json.loads(line)
            epoch = int(record["epoch"])
            for name, value in record.get("scalars", {}).items():
                if (
                    name.startswith("factorization/")
                    and value is not None
                    and math.isfinite(float(value))
                ):
                    series.setdefault(name, {})[epoch] = float(value)
        return series
    payload = json.loads(text)
    series: dict[str, dict[int, float]] = {}
    for record in payload:
        epoch = int(record["epoch"])
        for name, metric in record["metrics"].items():
            value = metric.get("mean")
            if value is not None and math.isfinite(float(value)):
                series.setdefault(name, {})[epoch] = float(value)
    return series


def load_probe_root(path: Path) -> dict[str, dict[int, float]]:
    series: dict[str, dict[int, float]] = {}
    for result_path in sorted(path.glob("epoch_*/results.json")):
        payload = json.loads(result_path.read_text())
        epoch = int(payload["checkpoint_epoch"])
        for probe, values in payload.get("probes", {}).items():
            for score in ("top1", "top5", "loss"):
                if score in values:
                    series.setdefault(f"{probe}/{score}", {})[epoch] = float(values[score])
    return series


def load_metrics_jsonl(path: Path) -> dict[str, dict[int, float]]:
    series: dict[str, dict[int, float]] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") != "linear_probe":
            continue
        epoch = int(record["epoch"])
        scalars = record.get("scalars", {})
        mapping = {
            "context/ridge/top1": "diag/class_accuracy",
            "context/ridge/top5": "diag/class_top5_accuracy",
        }
        for output_name, scalar_name in mapping.items():
            if scalar_name in scalars:
                series.setdefault(output_name, {})[epoch] = float(scalars[scalar_name])
    return series


def parse_probe_sources(values: list[str]) -> dict[str, dict[int, float]]:
    combined: dict[str, dict[int, float]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"probe source must be NAME=PATH: {value}")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        loaded = load_metrics_jsonl(path) if path.is_file() else load_probe_root(path)
        for name, points in loaded.items():
            key = f"{label}/{name}"
            if key in combined:
                overlap = set(combined[key]) & set(points)
                if overlap:
                    raise ValueError(f"duplicate probe epochs for {key}: {sorted(overlap)}")
            combined.setdefault(key, {}).update(points)
    return combined


def aligned(
    left: dict[int, float], right: dict[int, float], *, differences: bool
) -> tuple[list[int], np.ndarray, np.ndarray]:
    epochs = sorted(set(left) & set(right))
    x = np.asarray([left[epoch] for epoch in epochs], dtype=np.float64)
    y = np.asarray([right[epoch] for epoch in epochs], dtype=np.float64)
    if differences:
        return epochs[1:], np.diff(x), np.diff(y)
    return epochs, x, y


def main() -> None:
    args = parse_args()
    if not args.probe_source:
        raise ValueError("at least one --probe-source is required")
    factors = load_factorization(args.factorization_records.expanduser().resolve())
    probes = parse_probe_sources(args.probe_source)
    generator = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for factor_name, factor_points in sorted(factors.items()):
        for probe_name, probe_points in sorted(probes.items()):
            for mode, differences in (("levels", False), ("first_differences", True)):
                epochs, left, right = aligned(
                    factor_points, probe_points, differences=differences
                )
                if len(epochs) < args.minimum_points:
                    continue
                row: dict[str, Any] = {
                    "factorization_metric": factor_name,
                    "probe": probe_name,
                    "mode": mode,
                    "n": len(epochs),
                    "epochs": epochs,
                }
                for statistic, spearman in (("pearson", False), ("spearman", True)):
                    estimate = correlation(left, right, spearman=spearman)
                    low, high = bootstrap_interval(
                        left,
                        right,
                        spearman=spearman,
                        samples=args.bootstrap_samples,
                        generator=generator,
                    )
                    row[statistic] = estimate
                    row[f"{statistic}_ci95"] = [low, high]
                rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "correlations.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True)
    )
    for mode in ("levels", "first_differences"):
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            continue
        metric_names = sorted({row["factorization_metric"] for row in selected})
        probe_names = sorted({row["probe"] for row in selected})
        lookup = {
            (row["factorization_metric"], row["probe"]): row for row in selected
        }
        figure, axes = plt.subplots(1, 2, figsize=(max(12, len(probe_names) * 1.1), 7))
        for axis, statistic in zip(axes, ("pearson", "spearman"), strict=True):
            matrix = np.full((len(metric_names), len(probe_names)), np.nan)
            for row_index, metric in enumerate(metric_names):
                for column_index, probe in enumerate(probe_names):
                    row = lookup.get((metric, probe))
                    if row:
                        matrix[row_index, column_index] = row[statistic]
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            axis.set_title(statistic.title())
            axis.set_xticks(range(len(probe_names)), probe_names, rotation=55, ha="right")
            axis.set_yticks(range(len(metric_names)), metric_names)
            for row_index in range(len(metric_names)):
                for column_index in range(len(probe_names)):
                    value = matrix[row_index, column_index]
                    if math.isfinite(value):
                        axis.text(
                            column_index, row_index, f"{value:.2f}",
                            ha="center", va="center", fontsize=7,
                        )
            figure.colorbar(image, ax=axis, fraction=0.035)
        figure.suptitle(f"Factorization vs probing: {mode.replace('_', ' ')}")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"correlations_{mode}.png", dpi=180)
        plt.close(figure)
    print(args.output_dir)


if __name__ == "__main__":
    main()
