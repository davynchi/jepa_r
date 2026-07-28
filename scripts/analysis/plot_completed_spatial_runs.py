#!/usr/bin/env python3
"""Plot comparable training diagnostics for completed spatial-grid runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "uniform": "#2563EB",
    "loss": "#D97706",
    "ras-logdet": "#7C3AED",
    "coord-covariance": "#0F766E",
    "coord-transformation": "#DB2777",
    "coord-dynamics": "#64748B",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", default="pilot")
    return parser.parse_args()


def _completed_runs(grid_root: Path, phase: str) -> list[dict]:
    manifest = json.loads((grid_root / "grid_manifest.json").read_text())
    return [
        cell
        for cell in manifest["cells"]
        if cell["phase"] == phase and cell["state"] == "complete"
    ]


def _series(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        epoch = int(row["epoch"])
        scalars = row.get("scalars", {})
        if row["event"] == "epoch":
            values["train_loss"].append((epoch, float(scalars["train/epoch_loss"])))
        elif row["event"] == "eval_spectrum":
            values["test_loss"].append((epoch, float(scalars["eval/test_loss"])))
            values["effective_rank"].append(
                (epoch, float(scalars["repr/effective_rank"]))
            )
            values["trace_covariance"].append(
                (epoch, float(scalars["repr/trace_covariance"]))
            )
    return values


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#475569")
    axis.xaxis.label.set_color("#334155")
    axis.yaxis.label.set_color("#334155")
    axis.title.set_color("#0F172A")


def _draw_metric(
    axis: plt.Axes,
    runs: list[tuple[str, dict[str, list[tuple[int, float]]]]],
    metric: str,
    title: str,
    ylabel: str,
    *,
    log_scale: bool = False,
    legend: bool = False,
) -> None:
    for curriculum, data in runs:
        points = data[metric]
        axis.plot(
            [epoch for epoch, _ in points],
            [value for _, value in points],
            color=COLORS[curriculum],
            label=curriculum,
            linewidth=1.8,
            marker="o" if len(points) <= 12 else None,
            markersize=3.5,
        )
    axis.set_title(title, loc="left", fontsize=11, fontweight="semibold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    if log_scale:
        axis.set_yscale("log")
    _style_axis(axis)
    if legend:
        axis.legend(frameon=False, fontsize=8, ncols=2)


def _overview(
    runs: list[tuple[str, dict[str, list[tuple[int, float]]]]], destination: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    _draw_metric(
        axes[0, 0],
        runs,
        "train_loss",
        "Training loss",
        "Mean epoch loss (log scale)",
        log_scale=True,
        legend=True,
    )
    _draw_metric(
        axes[0, 1],
        runs,
        "test_loss",
        "Held-out JEPA loss",
        "Test loss (log scale)",
        log_scale=True,
    )
    _draw_metric(
        axes[1, 0],
        runs,
        "effective_rank",
        "Representation effective rank",
        "Effective rank",
    )
    _draw_metric(
        axes[1, 1],
        runs,
        "trace_covariance",
        "Representation covariance",
        "Trace of covariance",
    )
    figure.suptitle(
        "Shapes3D completed pilot runs",
        x=0.01,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _per_run(
    curriculum: str,
    data: dict[str, list[tuple[int, float]]],
    destination: Path,
) -> None:
    runs = [(curriculum, data)]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    _draw_metric(
        axes[0, 0],
        runs,
        "train_loss",
        "Training loss",
        "Mean epoch loss (log scale)",
        log_scale=True,
    )
    _draw_metric(
        axes[0, 1],
        runs,
        "test_loss",
        "Held-out JEPA loss",
        "Test loss (log scale)",
        log_scale=True,
    )
    _draw_metric(
        axes[1, 0],
        runs,
        "effective_rank",
        "Representation effective rank",
        "Effective rank",
    )
    _draw_metric(
        axes[1, 1],
        runs,
        "trace_covariance",
        "Representation covariance",
        "Trace of covariance",
    )
    figure.suptitle(
        f"Shapes3D pilot — {curriculum}",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    arguments = _arguments()
    grid_root = arguments.grid_root.expanduser().resolve()
    output = arguments.output_dir.expanduser().resolve()
    cells = _completed_runs(grid_root, arguments.phase)
    if not cells:
        raise ValueError(f"no completed runs in phase {arguments.phase!r}")
    output.mkdir(parents=True, exist_ok=True)
    runs = [
        (str(cell["curriculum"]), _series(Path(cell["run_dir"]))) for cell in cells
    ]
    _overview(runs, output / "completed_runs_overview.png")
    for curriculum, data in runs:
        _per_run(curriculum, data, output / f"{curriculum}.png")
    print(f"runs={len(runs)} output={output}")


if __name__ == "__main__":
    sys.exit(main())
