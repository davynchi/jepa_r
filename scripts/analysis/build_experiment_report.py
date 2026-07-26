#!/usr/bin/env python3
"""Build compact figures for the spatial I-JEPA sampling experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]


def jsonl_series(path: Path, event: str, keys: tuple[str, ...]) -> dict[str, list[tuple[float, float]]]:
    result = {key: [] for key in keys}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != event:
            continue
        epoch = float(row["epoch"])
        scalars = row.get("scalars", {})
        for key in keys:
            if key in scalars:
                result[key].append((epoch, float(scalars[key])))
    return result


def json_array_series(path: Path, keys: tuple[str, ...]) -> dict[str, list[tuple[float, float]]]:
    rows = json.loads(path.read_text())
    return {
        key: sorted(
            (float(row["epoch"]), float(row[key])) for row in rows if key in row
        )
        for key in keys
    }


def draw(axis, points, label, color, *, linestyle="-"):
    if not points:
        return
    x, y = zip(*points, strict=True)
    axis.plot(x, y, label=label, color=color, linewidth=2, linestyle=linestyle)


def finish_figure(fig, axes, output: Path, title: str):
    handles, labels = [], []
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.set_xlabel("epoch")
        h, labels_now = axis.get_legend_handles_labels()
        for handle, label in zip(h, labels_now, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle(title, fontsize=17)
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_clean_shapes(output_dir: Path):
    source = ROOT / "outputs/ijepa_spatial/plots_134_pr_rbar_20260723"
    specs = (
        ("uniform_diagnostics.json", "Uniform", "#2878b5"),
        ("loss_diagnostics.json", "Loss sampling", "#e17c27"),
        ("ras_logdet_diagnostics.json", "RAS logdet", "#31945f"),
        ("rbar_diagnostics.json", "RAS Rbar", "#ca3f3f"),
        ("pr_diagnostics.json", "RAS PR", "#7857b5"),
    )
    keys = ("entity_accuracy", "context_accuracy_mean", "effective_rank", "test_loss")
    fig, grid = plt.subplots(2, 2, figsize=(13.5, 9))
    panels = (
        ("Entity probe accuracy", "entity_accuracy"),
        ("Context probe accuracy", "context_accuracy_mean"),
        ("Effective rank", "effective_rank"),
        ("Held-out JEPA loss", "test_loss"),
    )
    for filename, label, color in specs:
        series = json_array_series(source / filename, keys)
        for axis, (_, key) in zip(grid.flat, panels, strict=True):
            draw(axis, series[key], label, color)
    for axis, (title, _) in zip(grid.flat, panels, strict=True):
        axis.set_title(title)
    finish_figure(
        fig,
        list(grid.flat),
        output_dir / "01_shapes3d_clean.png",
        "Shapes3D, legacy CNN: richness and sampling ablations",
    )


def build_corrupted_shapes(output_dir: Path):
    source = Path("/tmp")
    specs = (
        ("corrupted_uniform.jsonl", "Uniform", "#2878b5"),
        ("corrupted_loss.jsonl", "Loss sampling", "#e17c27"),
        ("corrupted_ras_pr.jsonl", "RAS PR", "#ca3f3f"),
        ("corrupted_oracle_clean.jsonl", "Oracle clean", "#31945f"),
    )
    diag_keys = ("diag/entity_accuracy", "diag/test_loss")
    epoch_keys = ("corruption/sampled_fraction", "weighting/effective_sample_size")
    fig, grid = plt.subplots(2, 2, figsize=(13.5, 9))
    for filename, label, color in specs:
        path = source / filename
        diagnostics = jsonl_series(path, "heldout_diagnostics", diag_keys)
        weights = jsonl_series(path, "epoch", epoch_keys)
        draw(grid[0, 0], diagnostics["diag/entity_accuracy"], label, color)
        draw(grid[0, 1], diagnostics["diag/test_loss"], label, color)
        draw(grid[1, 0], weights["corruption/sampled_fraction"], label, color)
        ess = [(x, y / 16_000) for x, y in weights["weighting/effective_sample_size"]]
        draw(grid[1, 1], ess, label, color)
    grid[0, 0].set_title("Entity probe accuracy")
    grid[0, 1].set_title("Held-out JEPA loss")
    grid[1, 0].set_title("Corrupted samples actually drawn")
    grid[1, 1].set_title("Sampling support")
    grid[1, 0].axhline(0.2, color="#777777", linestyle=":", linewidth=1.5)
    grid[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    grid[1, 1].set_ylabel("ESS / 16k")
    grid[1, 1].set_ylim(0, 1.05)
    finish_figure(
        fig,
        list(grid.flat),
        output_dir / "02_shapes3d_corrupted20.png",
        "Shapes3D with 20% Gaussian noise: can sampling reject bad data?",
    )


def build_tiny(output_dir: Path):
    source = Path("/tmp/tiny_imagenet_comparison")
    specs = (
        ("cnn_uniform.metrics.jsonl", "Legacy CNN uniform", "#2878b5"),
        ("cnn_loss.metrics.jsonl", "Legacy CNN loss", "#e17c27"),
        ("cnn_ras_pr.metrics.jsonl", "Legacy CNN RAS PR", "#31945f"),
    )
    keys = ("diag/class_accuracy", "diag/class_top5_accuracy", "diag/effective_rank")
    fig, grid = plt.subplots(2, 2, figsize=(13.5, 9))
    for filename, label, color in specs:
        path = source / filename
        diagnostics = jsonl_series(path, "heldout_diagnostics", keys)
        weights = jsonl_series(path, "weighting_update", ("weighting/effective_sample_size",))
        draw(grid[0, 0], diagnostics["diag/class_accuracy"], label, color)
        draw(grid[0, 1], diagnostics["diag/class_top5_accuracy"], label, color)
        draw(grid[1, 0], diagnostics["diag/effective_rank"], label, color)
        ess = [(x, y / 100_000) for x, y in weights["weighting/effective_sample_size"]]
        draw(grid[1, 1], ess, label, color)

    vit_uniform_path = Path("/tmp/tiny_vits4_uniform_e700_diagnostics.json")
    if vit_uniform_path.exists() and vit_uniform_path.stat().st_size:
        vit_uniform = json_array_series(
            vit_uniform_path,
            ("class_accuracy", "class_top5_accuracy", "effective_rank"),
        )
    else:
        # Exact watcher output retained locally; epoch 50 was unavailable during
        # the interrupted artifact transfer and is intentionally left out.
        rows = [
            (25, .153, .324, 11.930),
            (75, .231, .488, 52.004),
            (100, .245, .488, 56.83557541967138),
            (125, .248, .515, 59.916481078879464),
            (150, .262, .527, 67.24589302414867),
            (175, .259, .525, 68.04979668901717),
            (200, .269, .532, 72.62640538812036),
            (225, .256, .522, 77.278),
            (250, .257, .514, 80.07861276951114),
            (275, .257, .528, 80.818),
            (300, .260, .524, 82.04351464465151),
            (325, .259, .527, 86.018),
            (350, .272, .517, 87.14306330866827),
            (375, .273, .512, 88.272),
            (400, .260, .509, 90.611786141142),
            (425, .258, .512, 90.32952680868064),
            (450, .259, .502, 88.465),
            (475, .268, .504, 86.063),
            (500, .262, .508, 85.143),
            (525, .249, .508, 82.060),
            (550, .254, .504, 78.414),
            (600, .255, .502, 70.315),
            (625, .258, .500, 65.810),
            (650, .252, .501, 62.266),
            (675, .256, .501, 60.395),
            (700, .259, .499, 60.11125336030425),
        ]
        vit_uniform = {
            "class_accuracy": [(e, top1) for e, top1, _, _ in rows],
            "class_top5_accuracy": [(e, top5) for e, _, top5, _ in rows],
            "effective_rank": [(e, rank) for e, _, _, rank in rows],
        }

    if vit_uniform:
        draw(
            grid[0, 0],
            vit_uniform["class_accuracy"],
            "ViT-S/4 uniform",
            "#7655b5",
        )
        draw(
            grid[0, 1],
            vit_uniform["class_top5_accuracy"],
            "ViT-S/4 uniform",
            "#7655b5",
        )
        draw(
            grid[1, 0],
            vit_uniform["effective_rank"],
            "ViT-S/4 uniform",
            "#7655b5",
        )

    continuation_epochs = [700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
    continuation = {
        "ViT-S/4 uniform continuation": {
            "color": "#7655b5",
            "top1": [.259, .262, .258, .251, .240, .246, .247, .248, .241, .243, .244, .245, .247],
            "top5": [.499, .528, .499, .496, .504, .497, .517, .508, .506, .509, .507, .511, .506],
            "rank": [60.111, 62.0, 64.565, 62.648, 61.763, 57.138, 54.374, 51.821, 48.680, 47.385, 45.174, 44.579, 44.076],
        },
        "RAS predictive-spectral": {
            "color": "#ca5a2a",
            "top1": [.259, .253, .253, .250, .241, .235, .240, .237, .233, .234, .233, .234, .232],
            "top5": [.499, .494, .492, .499, .491, .493, .500, .497, .511, .498, .498, .500, .503],
            "rank": [60.111, 63.3, 63.914, 62.325, 60.045, 59.093, 55.020, 52.156, 49.460, 46.697, 44.916, 44.149, 43.637],
        },
    }
    for label, values in continuation.items():
        color = values["color"]
        draw(grid[0, 0], list(zip(continuation_epochs, values["top1"])), label, color, linestyle="--")
        draw(grid[0, 1], list(zip(continuation_epochs, values["top5"])), label, color, linestyle="--")
        draw(grid[1, 0], list(zip(continuation_epochs, values["rank"])), label, color, linestyle="--")

    spectral_ess = [
        (700, 46023.49),
        (750, 53962.81),
        (800, 46097.01),
        (850, 54300.91),
        (900, 55137.57),
        (950, 47839.84),
        (1000, 39150.73),
    ]
    draw(
        grid[1, 1],
        [(x, y / 100_000) for x, y in spectral_ess],
        "RAS predictive-spectral",
        "#ca5a2a",
        linestyle="--",
    )
    grid[0, 0].set_title("Linear probe top-1")
    grid[0, 1].set_title("Linear probe top-5")
    grid[1, 0].set_title("Effective rank")
    grid[1, 1].set_title("Sampling support")
    grid[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
    grid[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    grid[1, 1].set_ylabel("ESS / 100k")
    grid[1, 1].set_ylim(0, 1.05)
    finish_figure(
        fig,
        list(grid.flat),
        output_dir / "03_tiny_imagenet.png",
        "Tiny ImageNet: legacy samplers and ViT-S/4 continuation",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/experiment_report_20260726",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_clean_shapes(args.output_dir)
    build_corrupted_shapes(args.output_dir)
    build_tiny(args.output_dir)


if __name__ == "__main__":
    main()
