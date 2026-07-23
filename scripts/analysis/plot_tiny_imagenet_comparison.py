#!/usr/bin/env python3
"""Compare the useful Tiny ImageNet spatial I-JEPA runs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    source: str
    color: str
    linestyle: str = "-"


RUNS = (
    RunSpec("cnn_uniform", "CNN · uniform · b2048", "legacy CNN", "#2673b8"),
    RunSpec("cnn_loss", "CNN · loss · b2048", "legacy CNN", "#e07a28"),
    RunSpec("cnn_ras_pr", "CNN · RAS-PR · b2048", "legacy CNN", "#2e9364"),
    RunSpec(
        "vit_tiny_uniform",
        "ViT-T/8 · uniform · b512",
        "upstream architecture",
        "#7655b5",
    ),
    RunSpec(
        "vit_tiny_loss",
        "ViT-T/8 · loss · b512",
        "upstream architecture",
        "#b54e76",
    ),
)

PANELS = (
    ("Linear probe top-1", ("diag/class_accuracy",), False, None),
    ("Linear probe top-5", ("diag/class_top5_accuracy",), False, None),
    (
        "Effective rank",
        ("diag/effective_rank", "repr/effective_rank"),
        False,
        None,
    ),
    ("Held-out JEPA loss", ("diag/test_loss", "eval/test_loss"), True, None),
    ("Training loss", ("train/epoch_loss",), True, None),
    (
        "Sampling support",
        ("weighting/effective_sample_size",),
        False,
        100_000.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def load_metrics(path: Path) -> tuple[dict[str, dict[float, float]], int]:
    series: dict[str, dict[float, float]] = {}
    max_epoch = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch = row.get("epoch")
        scalars = row.get("scalars")
        if not isinstance(epoch, int | float) or not isinstance(scalars, dict):
            continue
        max_epoch = max(max_epoch, int(epoch))
        for key, value in scalars.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            x = scalars.get("train/epoch_fraction", epoch) if key == "train/loss" else epoch
            if math.isfinite(float(x)) and math.isfinite(float(value)):
                series.setdefault(key, {})[float(x)] = float(value)
    return series, max_epoch


def merged_points(
    series: dict[str, dict[float, float]],
    keys: tuple[str, ...],
    *,
    denominator: float | None,
) -> list[tuple[float, float]]:
    merged: dict[float, float] = {}
    for key in reversed(keys):
        merged.update(series.get(key, {}))
    points = sorted(merged.items())
    if denominator is not None:
        points = [(x, y / denominator) for x, y in points]
    return points


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, tuple[dict[str, dict[float, float]], int]] = {}
    manifest_runs = []
    for spec in RUNS:
        metrics_path = input_dir / f"{spec.key}.metrics.jsonl"
        if not metrics_path.exists():
            continue
        loaded[spec.key] = load_metrics(metrics_path)
        manifest_runs.append(
            {
                "key": spec.key,
                "label": spec.label,
                "source_group": spec.source,
                "max_epoch": loaded[spec.key][1],
                "config": load_config(input_dir / f"{spec.key}.config.json"),
            }
        )

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for axis, (title, keys, log_scale, denominator) in zip(
        axes.flat, PANELS, strict=True
    ):
        plotted = False
        for spec in RUNS:
            if spec.key not in loaded:
                continue
            points = merged_points(loaded[spec.key][0], keys, denominator=denominator)
            if not points:
                continue
            x, y = zip(*points, strict=True)
            axis.plot(
                x,
                y,
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=1.8,
                marker="o",
                markersize=2.8,
                markevery=max(len(x) // 18, 1),
                label=spec.label,
            )
            plotted = True
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        if denominator is not None:
            axis.set_ylabel("ESS / dataset size")
            axis.set_ylim(0, 1.03)
        if log_scale:
            axis.set_yscale("log")
        if not plotted:
            axis.text(
                0.5,
                0.5,
                "metric not available",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="gray",
            )

    legend_entries: dict[str, Any] = {}
    for axis in axes.flat:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=True):
            legend_entries.setdefault(label, handle)
    labels = list(legend_entries)
    handles = [legend_entries[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Tiny ImageNet · spatial I-JEPA experiment comparison",
        fontsize=17,
        y=0.985,
    )
    fig.text(
        0.5,
        0.947,
        "Completed and intentionally paused historical runs",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.935))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "figure": output.name,
        "included_runs": manifest_runs,
        "excluded_runs": [
            {
                "pattern": "tiny_vit_small_p4_*_opt",
                "reason": "currently running; omitted until comparable metrics are complete",
            },
            {
                "pattern": "tiny_resnet_*",
                "reason": "aborted after two completed epochs",
            },
            {
                "pattern": "tiny_*_b32000_*",
                "reason": "large-batch collapse control; omitted from the readable main figure",
            },
        ],
    }
    manifest_path = args.manifest or output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
