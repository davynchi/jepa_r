#!/usr/bin/env python3
"""Plot target-encoder probes for I-JEPA, RAS, and regularized runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/probe_benchmarks"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_curve(
    directory: Path,
    probe: str,
    *,
    maximum_epoch: int = 250,
) -> tuple[list[int], list[float], list[float]]:
    rows: list[tuple[int, float, float]] = []
    for path in sorted(directory.glob("epoch_*/results.json")):
        payload = json.loads(path.read_text())
        result = payload.get("probes", {}).get(probe)
        if result is not None:
            epoch = int(path.parent.name.removeprefix("epoch_"))
            if epoch <= maximum_epoch:
                rows.append((epoch, float(result["top1"]), float(result["top5"])))
    return (
        [row[0] for row in rows],
        [100 * row[1] for row in rows],
        [100 * row[2] for row in rows],
    )


def load_single(path: Path, probe: str) -> tuple[list[int], list[float], list[float]]:
    if not path.exists():
        return [], [], []
    payload = json.loads(path.read_text())
    result = payload.get("probes", {}).get(probe)
    if result is None:
        return [], [], []
    epoch = int(payload["checkpoint_epoch"])
    return [epoch], [100 * float(result["top1"])], [100 * float(result["top5"])]


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    sources = {
        "Weighthning RAS": root / "vit_small_probe_matrix" / "ras",
        "Barlow regularization": root / "official_barlow_reg_curve",
    }
    styles = {
        "Weighthning RAS": (":", "^"),
        "Barlow regularization": ("-", "s"),
    }
    probes = {
        "target/ridge": "#7b61a8",
        "target/mlp": "#d62728",
        "target/finetune": "#1f77b4",
    }
    curves: dict[tuple[str, str], tuple[list[int], list[float], list[float]]] = {}
    for model, directory in sources.items():
        for probe in probes:
            curves[(model, probe)] = load_curve(directory, probe)
    curves[("Weighthning RAS", "target/finetune")] = load_single(
        root / "tiny_vit_small_ras_e250_target_finetune" / "results.json",
        "target/finetune",
    )

    figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    for axis, score_index, title in zip(
        axes, (1, 2), ("Top-1 accuracy", "Top-5 accuracy"), strict=True
    ):
        for model, (line_style, marker) in styles.items():
            for probe, color in probes.items():
                values = curves[(model, probe)]
                epochs = values[0]
                scores = values[score_index]
                if not epochs:
                    continue
                axis.plot(
                    epochs,
                    scores,
                    color=color,
                    linestyle=line_style,
                    marker=marker,
                    linewidth=2.2,
                    markersize=6,
                )
        axis.set_title(title)
        axis.set_xlabel("Pretraining epoch")
        axis.set_ylabel("Accuracy, %")
        axis.set_xticks((50, 100, 150, 200, 250))
        axis.grid(alpha=0.25)

    probe_handles = [
        Line2D([0], [0], color=color, linewidth=2.5, label=probe)
        for probe, color in probes.items()
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=line_style,
            marker=marker,
            linewidth=2.2,
            label=model,
        )
        for model, (line_style, marker) in styles.items()
    ]
    axes[0].legend(handles=probe_handles, title="Probe", loc="upper left")
    axes[1].legend(handles=model_handles, title="Pretraining", loc="lower right")
    figure.suptitle("Tiny ImageNet ViT-Small: target-encoder downstream quality")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
