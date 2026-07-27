#!/usr/bin/env python3
"""Plot matching context-ridge probe curves for uniform and RAS runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-probe-root", type=Path, required=True)
    parser.add_argument("--uniform-metrics", type=Path)
    parser.add_argument("--ras-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_uniform(root: Path) -> dict[int, tuple[float, float]]:
    points = {}
    for path in sorted(root.glob("epoch_*/results.json")):
        payload = json.loads(path.read_text())
        metrics = payload["probes"]["context/ridge"]
        points[int(payload["checkpoint_epoch"])] = (
            float(metrics["top1"]),
            float(metrics["top5"]),
        )
    return points


def load_ras(path: Path) -> dict[int, tuple[float, float]]:
    points = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") != "linear_probe":
            continue
        scalars = record["scalars"]
        points[int(record["epoch"])] = (
            float(scalars["diag/class_accuracy"]),
            float(scalars["diag/class_top5_accuracy"]),
        )
    return points


def main() -> None:
    args = parse_args()
    uniform = load_uniform(args.uniform_probe_root)
    if args.uniform_metrics is not None:
        uniform.update(load_ras(args.uniform_metrics))
    series = {
        "Uniform": uniform,
        "Predictive-Barlow RAS": load_ras(args.ras_metrics),
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for label, points in series.items():
        epochs = sorted(points)
        axes[0].plot(
            epochs,
            [100 * points[epoch][0] for epoch in epochs],
            marker="o",
            linewidth=2,
            label=label,
        )
        axes[1].plot(
            epochs,
            [100 * points[epoch][1] for epoch in epochs],
            marker="o",
            linewidth=2,
            label=label,
        )
    for axis, title in zip(axes, ("Top-1 accuracy", "Top-5 accuracy"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy, %")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Tiny ImageNet ViT-Small: context ridge probe")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
