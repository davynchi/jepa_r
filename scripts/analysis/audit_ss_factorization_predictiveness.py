#!/usr/bin/env python3
"""Run trend, partial, cross-model, and lead tests for SS factorization."""

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

from correlate_tiny_factorization_v2 import correlation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME,FACTOR_JSON,PROBE_ROOT",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lead-epochs", type=int, default=50)
    parser.add_argument("--minimum-points", type=int, default=4)
    return parser.parse_args()


def load_factorization_display_epochs(path: Path) -> dict[str, dict[int, float]]:
    payload = json.loads(path.read_text())
    series: dict[str, dict[int, float]] = {}
    for record in payload:
        epoch = int(record.get("display_epoch", record["epoch"]))
        for name, metric in record["metrics"].items():
            value = metric.get("mean")
            if value is not None and math.isfinite(float(value)):
                series.setdefault(name, {})[epoch] = float(value)
    return series


def load_probe_display_epochs(path: Path) -> dict[str, dict[int, float]]:
    series: dict[str, dict[int, float]] = {}
    for result_path in sorted(path.glob("epoch_*/results.json")):
        epoch = int(result_path.parent.name.removeprefix("epoch_"))
        payload = json.loads(result_path.read_text())
        for probe, values in payload.get("probes", {}).items():
            for score in ("top1", "top5", "loss"):
                if score in values:
                    series.setdefault(f"{probe}/{score}", {})[epoch] = float(values[score])
    return series


def parse_runs(values: list[str]) -> dict[str, tuple[dict, dict]]:
    runs = {}
    for value in values:
        fields = value.split(",", maxsplit=2)
        if len(fields) != 3:
            raise ValueError(f"run must be NAME,FACTOR_JSON,PROBE_ROOT: {value}")
        name, factor_path, probe_path = fields
        runs[name] = (
            load_factorization_display_epochs(
                Path(factor_path).expanduser().resolve()
            ),
            load_probe_display_epochs(Path(probe_path).expanduser().resolve()),
        )
    return runs


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def corr_payload(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    return {
        "pearson": _finite_or_none(correlation(left, right, spearman=False)),
        "spearman": _finite_or_none(correlation(left, right, spearman=True)),
    }


def residualize_polynomial(
    epochs: np.ndarray, values: np.ndarray, *, degree: int
) -> np.ndarray:
    degree = min(degree, max(len(values) - 2, 1))
    scaled = (epochs - epochs.mean()) / max(epochs.std(), 1.0)
    design = np.column_stack([scaled**power for power in range(degree + 1)])
    fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    return values - fitted


def aligned(
    left: dict[int, float], right: dict[int, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs = sorted(set(left) & set(right))
    return (
        np.asarray(epochs, dtype=np.float64),
        np.asarray([left[e] for e in epochs], dtype=np.float64),
        np.asarray([right[e] for e in epochs], dtype=np.float64),
    )


def within_run_tests(
    runs: dict[str, tuple[dict, dict]],
    *,
    minimum_points: int,
    lead_epochs: int,
) -> list[dict[str, Any]]:
    rows = []
    for run_name, (factors, probes) in runs.items():
        for metric_name, metric_points in factors.items():
            for probe_name, probe_points in probes.items():
                epochs, metric, probe = aligned(metric_points, probe_points)
                if len(epochs) >= minimum_points:
                    for name, degree in (("partial_linear_epoch", 1), ("detrended_quadratic", 2)):
                        rows.append(
                            {
                                "test": name,
                                "run": run_name,
                                "metric": metric_name,
                                "probe": probe_name,
                                "n": len(epochs),
                                "epochs": epochs.astype(int).tolist(),
                                **corr_payload(
                                    residualize_polynomial(epochs, metric, degree=degree),
                                    residualize_polynomial(epochs, probe, degree=degree),
                                ),
                            }
                        )
                lead_epochs_available = sorted(
                    epoch
                    for epoch in metric_points
                    if epoch + lead_epochs in probe_points
                )
                if len(lead_epochs_available) >= minimum_points:
                    metric_now = np.asarray(
                        [metric_points[e] for e in lead_epochs_available]
                    )
                    probe_future = np.asarray(
                        [probe_points[e + lead_epochs] for e in lead_epochs_available]
                    )
                    row = {
                        "test": "lead",
                        "run": run_name,
                        "metric": metric_name,
                        "probe": probe_name,
                        "lead_epochs": lead_epochs,
                        "n": len(lead_epochs_available),
                        "epochs": lead_epochs_available,
                        **corr_payload(metric_now, probe_future),
                    }
                    if all(epoch in probe_points for epoch in lead_epochs_available):
                        future_gain = probe_future - np.asarray(
                            [probe_points[e] for e in lead_epochs_available]
                        )
                        gain = corr_payload(metric_now, future_gain)
                        row["future_gain_pearson"] = gain["pearson"]
                        row["future_gain_spearman"] = gain["spearman"]
                    rows.append(row)
    return rows


def cross_model_tests(
    runs: dict[str, tuple[dict, dict]],
    *,
    minimum_points: int,
) -> list[dict[str, Any]]:
    metric_names = sorted(set.intersection(*(set(value[0]) for value in runs.values())))
    probe_names = sorted(set.intersection(*(set(value[1]) for value in runs.values())))
    rows = []
    for metric_name in metric_names:
        for probe_name in probe_names:
            shared_epochs = sorted(
                set.intersection(
                    *(
                        set(factors[metric_name]) & set(probes[probe_name])
                        for factors, probes in runs.values()
                    )
                )
            )
            centered_metric: list[float] = []
            centered_probe: list[float] = []
            concordant = 0
            compared = 0
            for epoch in shared_epochs:
                metric_values = np.asarray(
                    [value[0][metric_name][epoch] for value in runs.values()]
                )
                probe_values = np.asarray(
                    [value[1][probe_name][epoch] for value in runs.values()]
                )
                centered_metric.extend(metric_values - metric_values.mean())
                centered_probe.extend(probe_values - probe_values.mean())
                for left in range(len(metric_values)):
                    for right in range(left + 1, len(metric_values)):
                        metric_difference = metric_values[left] - metric_values[right]
                        probe_difference = probe_values[left] - probe_values[right]
                        if metric_difference == 0 or probe_difference == 0:
                            continue
                        compared += 1
                        concordant += int(metric_difference * probe_difference > 0)
            if len(centered_metric) < minimum_points:
                continue
            rows.append(
                {
                    "test": "cross_model_epoch_fixed_effects",
                    "run": "all",
                    "metric": metric_name,
                    "probe": probe_name,
                    "n": len(centered_metric),
                    "epochs": shared_epochs,
                    "models": list(runs),
                    "ranking_concordance": concordant / compared if compared else None,
                    "ranking_comparisons": compared,
                    **corr_payload(
                        np.asarray(centered_metric),
                        np.asarray(centered_probe),
                    ),
                }
            )
    return rows


def plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    tests = (
        "partial_linear_epoch",
        "detrended_quadratic",
        "cross_model_epoch_fixed_effects",
        "lead",
    )
    metrics = sorted(
        {
            row["metric"]
            for row in rows
            if row["metric"]
            in {
                "ss_crop_invariance_mean",
                "ss_crop_latent_effective_rank",
                "ss_crop_signal_to_perturbation_trace",
                "ss_crop_perturbation_effective_rank",
            }
        }
    )
    probes = sorted(
        {
            row["probe"]
            for row in rows
            if row["probe"].startswith("target/") and row["probe"].endswith("/top1")
        }
    )
    figure, axes = plt.subplots(len(tests), 1, figsize=(max(12, len(probes) * 1.4), 13))
    for axis, test in zip(axes, tests, strict=True):
        matrix = np.full((len(metrics), len(probes)), np.nan)
        for row_index, metric in enumerate(metrics):
            for column_index, probe in enumerate(probes):
                selected = [
                    row["pearson"]
                    for row in rows
                    if row["test"] == test
                    and row["metric"] == metric
                    and row["probe"] == probe
                    and row["pearson"] is not None
                    and math.isfinite(row["pearson"])
                ]
                if selected:
                    matrix[row_index, column_index] = float(np.mean(selected))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        axis.set_title(f"{test}: mean Pearson across eligible runs")
        axis.set_xticks(range(len(probes)), probes, rotation=35, ha="right")
        axis.set_yticks(range(len(metrics)), metrics)
        for row_index in range(len(metrics)):
            for column_index in range(len(probes)):
                value = matrix[row_index, column_index]
                if math.isfinite(value):
                    axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.02)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    rows = within_run_tests(
        runs,
        minimum_points=args.minimum_points,
        lead_epochs=args.lead_epochs,
    )
    rows.extend(cross_model_tests(runs, minimum_points=args.minimum_points))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False)
    )
    plot_summary(rows, args.output_dir / "audit_summary.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()
