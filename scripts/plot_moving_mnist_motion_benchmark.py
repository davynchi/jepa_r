#!/usr/bin/env python3
"""Create a compact low-shot motion dashboard for Moving-MNIST JEPA runs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="include raw, last-token, and shuffled-input controls in budget panels",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def find_rows(root: Path, filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob(filename)):
        rows.extend(read_rows(path))
    return rows


def float_value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def int_value(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def latest_checkpoint_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    maximum: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (
            row.get("run_label", ""),
            row.get("feature_variant", ""),
            row.get("training_seed", ""),
        )
        maximum[key] = max(maximum.get(key, -1), int_value(row, "clips_seen"))
    return [
        row
        for row in rows
        if int_value(row, "clips_seen")
        == maximum[
            (
                row.get("run_label", ""),
                row.get("feature_variant", ""),
                row.get("training_seed", ""),
            )
        ]
    ]


def allowed_budget_row(row: dict[str, str], include_controls: bool) -> bool:
    variant = row.get("feature_variant", "")
    if variant in {"temporal_linear", "random_temporal_linear"}:
        return True
    return include_controls and variant in {
        "last_token_linear",
        "shuffled_input_temporal_linear",
        "raw_temporal_linear",
        "raw_last_frame_linear",
        "raw_difference_linear",
    }


def display_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("run_label", row.get("strategy", "run")),
        row.get("display_name", row.get("run_label", "run")),
        row.get("feature_variant", ""),
    )


def budget_curves(
    rows: list[dict[str, str]],
    metric: str,
    *,
    include_controls: bool,
) -> dict[tuple[str, str, str], list[tuple[int, float, float]]]:
    latest = latest_checkpoint_rows(rows)
    grouped: dict[tuple[tuple[str, str, str], int], list[float]] = defaultdict(list)
    for row in latest:
        if row.get("split") != "probe_evaluation" or row.get("metric") != metric:
            continue
        if not allowed_budget_row(row, include_controls):
            continue
        grouped[(display_key(row), int_value(row, "budget"))].append(
            float_value(row, "value")
        )
    curves: dict[tuple[str, str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for (key, budget), values in grouped.items():
        mean, std = mean_std(values)
        curves[key].append((budget, mean, std))
    for values in curves.values():
        values.sort(key=lambda item: item[0])
    return curves


def checkpoint_curve(
    rows: list[dict[str, str]],
    metric: str,
    budget: int,
) -> dict[tuple[str, str], list[tuple[int, float, float]]]:
    grouped: dict[tuple[tuple[str, str], int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "probe_evaluation" or row.get("metric") != metric:
            continue
        if row.get("feature_variant") != "temporal_linear":
            continue
        if int_value(row, "budget") != budget:
            continue
        key = (row.get("run_label", "run"), row.get("display_name", "run"))
        grouped[(key, int_value(row, "clips_seen"))].append(float_value(row, "value"))
    curves: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for (key, clips), values in grouped.items():
        mean, std = mean_std(values)
        curves[key].append((clips, mean, std))
    for values in curves.values():
        values.sort(key=lambda item: item[0])
    return curves


def rank_curves(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[tuple[int, float, float]]]:
    grouped: dict[tuple[tuple[str, str], int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "validation":
            continue
        if row.get("metric") != "context_token_effective_rank":
            continue
        key = (row.get("run_label", row.get("strategy", "run")), row.get("display_name", "run"))
        grouped[(key, int_value(row, "clips_seen"))].append(float_value(row, "value"))
    curves: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for (key, clips), values in grouped.items():
        mean, std = mean_std(values)
        curves[key].append((clips, mean, std))
    for values in curves.values():
        values.sort(key=lambda item: item[0])
    return curves


def plot_curve(ax: Any, points: list[tuple[int, float, float]], label: str) -> None:
    x = [value[0] for value in points]
    y = [value[1] for value in points]
    std = [value[2] for value in points]
    line = ax.plot(x, y, marker="o", label=label)[0]
    if any(value > 0 for value in std):
        ax.fill_between(
            x,
            [mean - deviation for mean, deviation in zip(y, std, strict=True)],
            [mean + deviation for mean, deviation in zip(y, std, strict=True)],
            alpha=0.15,
            color=line.get_color(),
        )


def friendly_label(key: tuple[str, str, str]) -> str:
    _, display_name, variant = key
    if variant == "temporal_linear" or variant == "random_temporal_linear":
        return display_name
    suffix = {
        "last_token_linear": "last token",
        "shuffled_input_temporal_linear": "shuffled input",
        "raw_temporal_linear": "raw temporal",
        "raw_last_frame_linear": "raw last frame",
        "raw_difference_linear": "raw differences",
    }.get(variant, variant)
    return f"{display_name} · {suffix}"


def write_summary(
    path: Path,
    rows: list[dict[str, str]],
    budget: int,
) -> None:
    latest = latest_checkpoint_rows(rows)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    wanted = {"direction_accuracy", "velocity_vector_rmse", "future_fde"}
    for row in latest:
        if row.get("feature_variant") != "temporal_linear":
            continue
        if row.get("split") != "probe_evaluation" or row.get("metric") not in wanted:
            continue
        if int_value(row, "budget") != budget:
            continue
        key = (row.get("run_label", "run"), row.get("display_name", "run"), row["metric"])
        grouped[key].append(float_value(row, "value"))
    records: list[dict[str, object]] = []
    for (run_label, display_name, metric), values in sorted(grouped.items()):
        mean, std = mean_std(values)
        records.append(
            {
                "run_label": run_label,
                "display_name": display_name,
                "budget": budget,
                "metric": metric,
                "mean": mean,
                "std": std,
                "observations": len(values),
            }
        )
    if not records:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    lowshot_rows = find_rows(root, "lowshot_metrics.csv")
    pretrain_rows = find_rows(root, "pretrain_metrics.csv")
    if not lowshot_rows:
        raise SystemExit(f"no lowshot_metrics.csv files found below {root}")
    output_dir = Path(args.output_dir) if args.output_dir else root / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    figure.suptitle("Low-shot motion quality on Moving-MNIST", fontsize=18, fontweight="bold")

    for key, points in budget_curves(
        lowshot_rows,
        "velocity_vector_rmse",
        include_controls=args.include_controls,
    ).items():
        plot_curve(axes[0, 0], points, friendly_label(key))
    axes[0, 0].set_title("Velocity-vector RMSE ↓")
    axes[0, 0].set_xlabel("labeled clips")
    axes[0, 0].set_ylabel("RMSE")

    for key, points in budget_curves(
        lowshot_rows,
        "direction_accuracy",
        include_controls=args.include_controls,
    ).items():
        plot_curve(axes[0, 1], points, friendly_label(key))
    axes[0, 1].axhline(0.125, linestyle=":", linewidth=1.5, label="Chance")
    axes[0, 1].set_title("Direction accuracy ↑")
    axes[0, 1].set_xlabel("labeled clips")
    axes[0, 1].set_ylabel("accuracy")

    for (_, display_name), points in checkpoint_curve(
        lowshot_rows, "future_fde", args.budget
    ).items():
        plot_curve(axes[1, 0], points, display_name)
    axes[1, 0].set_title(f"Future FDE at {args.budget} labels ↓")
    axes[1, 0].set_xlabel("clips seen in pretraining")
    axes[1, 0].set_ylabel("pixels")

    for (_, display_name), points in rank_curves(pretrain_rows).items():
        plot_curve(axes[1, 1], points, display_name)
    axes[1, 1].set_title("Context token effective rank ↑")
    axes[1, 1].set_xlabel("clips seen in pretraining")
    axes[1, 1].set_ylabel("effective rank")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    if by_label:
        figure.legend(
            by_label.values(),
            by_label.keys(),
            loc="outside lower center",
            ncol=min(4, len(by_label)),
            frameon=False,
        )

    for suffix in ["png", "svg"]:
        figure.savefig(output_dir / f"motion_quality_dashboard.{suffix}", dpi=180)
    plt.close(figure)
    write_summary(output_dir / "lowshot_final_summary.csv", lowshot_rows, args.budget)
    print(output_dir / "motion_quality_dashboard.png")


if __name__ == "__main__":
    main()
