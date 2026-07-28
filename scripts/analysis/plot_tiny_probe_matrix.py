#!/usr/bin/env python3
"""Plot uniform/RAS frozen-probe curves as one 2x2 comparison figure."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--uniform-ridge-root", type=Path, required=True)
    parser.add_argument("--uniform-metrics", type=Path)
    parser.add_argument("--ras-metrics", type=Path, required=True)
    parser.add_argument(
        "--finetune-results-root",
        type=Path,
        help="Directory containing per-checkpoint Uniform/RAS fine-tune result folders.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=("matrix", "overlay"),
        default="matrix",
        help="Use four separate panels or overlay Uniform/RAS in Top-1 and Top-5 panels.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        help="Hide points after this epoch.",
    )
    return parser.parse_args()


def load_matrix(root: Path, run: str) -> dict[str, dict[int, tuple[float, float]]]:
    series: dict[str, dict[int, tuple[float, float]]] = {}
    for path in sorted((root / run).glob("epoch_*/results.json")):
        payload = json.loads(path.read_text())
        epoch = int(payload["checkpoint_epoch"])
        for name, metrics in payload["probes"].items():
            series.setdefault(name, {})[epoch] = (
                float(metrics["top1"]),
                float(metrics["top5"]),
            )
    return series


def add_uniform_ridge(
    series: dict[str, dict[int, tuple[float, float]]],
    root: Path,
) -> None:
    points = series.setdefault("context/ridge", {})
    for path in sorted(root.glob("epoch_*/results.json")):
        payload = json.loads(path.read_text())
        metrics = payload["probes"]["context/ridge"]
        points[int(payload["checkpoint_epoch"])] = (
            float(metrics["top1"]),
            float(metrics["top5"]),
        )


def add_ras_ridge(
    series: dict[str, dict[int, tuple[float, float]]],
    path: Path,
) -> None:
    points = series.setdefault("context/ridge", {})
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") != "linear_probe":
            continue
        scalars = record["scalars"]
        points[int(record["epoch"])] = (
            float(scalars["diag/class_accuracy"]),
            float(scalars["diag/class_top5_accuracy"]),
        )


def merge_ridge_metrics(
    series: dict[str, dict[int, tuple[float, float]]],
    path: Path,
) -> None:
    points = series.setdefault("context/ridge", {})
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") != "linear_probe":
            continue
        scalars = record["scalars"]
        points[int(record["epoch"])] = (
            float(scalars["diag/class_accuracy"]),
            float(scalars["diag/class_top5_accuracy"]),
        )


def add_finetune_results(
    uniform: dict[str, dict[int, tuple[float, float]]],
    ras: dict[str, dict[int, tuple[float, float]]],
    root: Path,
) -> None:
    pattern = re.compile(r".*_(ras|uniform)_e(\d+)_(context|target)_finetune$")
    for path in sorted(root.glob("*/results.json")):
        match = pattern.fullmatch(path.parent.name)
        if match is None:
            continue
        run, epoch_text, encoder = match.groups()
        payload = json.loads(path.read_text())
        metrics = payload.get("probes", {}).get(f"{encoder}/finetune")
        if metrics is None:
            continue
        destination = ras if run == "ras" else uniform
        destination.setdefault(f"{encoder}/finetune", {})[int(epoch_text)] = (
            float(metrics["top1"]),
            float(metrics["top5"]),
        )


def plot_panel(
    axis,
    series: dict[str, dict[int, tuple[float, float]]],
    *,
    metric_index: int,
    title: str,
) -> None:
    for name, points in sorted(series.items()):
        epochs = sorted(points)
        axis.plot(
            epochs,
            [100 * points[epoch][metric_index] for epoch in epochs],
            marker="o",
            linewidth=1.8,
            label=name,
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy, %")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)


def main() -> None:
    args = parse_args()
    uniform = load_matrix(args.matrix_root, "uniform")
    ras = load_matrix(args.matrix_root, "ras")
    add_uniform_ridge(uniform, args.uniform_ridge_root)
    if args.uniform_metrics is not None:
        merge_ridge_metrics(uniform, args.uniform_metrics)
    add_ras_ridge(ras, args.ras_metrics)
    if args.finetune_results_root is not None:
        add_finetune_results(uniform, ras, args.finetune_results_root)
    if args.max_epoch is not None:
        for run_series in (uniform, ras):
            for name, points in run_series.items():
                run_series[name] = {
                    epoch: metrics
                    for epoch, metrics in points.items()
                    if epoch <= args.max_epoch
                }

    if args.layout == "matrix":
        figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        plot_panel(axes[0, 0], uniform, metric_index=0, title="Uniform: Top-1")
        plot_panel(axes[0, 1], uniform, metric_index=1, title="Uniform: Top-5")
        plot_panel(axes[1, 0], ras, metric_index=0, title="Predictive-Barlow RAS: Top-1")
        plot_panel(axes[1, 1], ras, metric_index=1, title="Predictive-Barlow RAS: Top-5")
    else:
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        probe_names = sorted(set(uniform) | set(ras))
        colors = {
            name: plt.get_cmap("tab10")(index % 10)
            for index, name in enumerate(probe_names)
        }
        for axis, metric_index, title in zip(
            axes,
            (0, 1),
            ("Top-1 accuracy", "Top-5 accuracy"),
            strict=True,
        ):
            for name in probe_names:
                for run_name, run_series, linestyle in (
                    ("Uniform", uniform, "--"),
                    ("RAS", ras, "-"),
                ):
                    points = run_series.get(name, {})
                    if not points:
                        continue
                    epochs = sorted(points)
                    axis.plot(
                        epochs,
                        [100 * points[epoch][metric_index] for epoch in epochs],
                        color=colors[name],
                        linestyle=linestyle,
                        marker="o",
                        markersize=3.5,
                        linewidth=1.8,
                    )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Accuracy, %")
            axis.grid(alpha=0.25)
            probe_handles = [
                Line2D([0], [0], color=colors[name], linewidth=2, label=name)
                for name in probe_names
            ]
            style_handles = [
                Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Uniform"),
                Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="RAS"),
            ]
            probe_legend = axis.legend(
                handles=probe_handles,
                title="Probe",
                fontsize=8,
                title_fontsize=8,
                ncol=2,
                loc="upper left",
            )
            axis.add_artist(probe_legend)
            axis.legend(
                handles=style_handles,
                title="Sampling",
                fontsize=8,
                title_fontsize=8,
                loc="lower right",
            )
    figure.suptitle("Tiny ImageNet ViT-Small: probes and full fine-tuning")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
