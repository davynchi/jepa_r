#!/usr/bin/env python3
"""Plot the spatial I-JEPA diagnostic trajectory written by
_test_loss_tracker_spatial.py (/tmp/ijepa_spatial_diagnostics.json).

Usage:
    python scripts/plot_spatial_diagnostics.py diagnostics.json out.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Untrained (random-init) CNN encoder, measured with the same pipeline --
# the reference every trained number has to beat to mean anything.
UNTRAINED = {
    "effective_rank": 3.788,
    "entity_accuracy": 0.421,
    "context_accuracy_mean": 0.439,
    "mi_ratio": 0.035,
}


def main() -> None:
    records = json.loads(Path(sys.argv[1]).read_text())
    out_path = sys.argv[2]
    epochs = [r["epoch"] for r in records]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    def baseline(axis, key: str) -> None:
        if key in UNTRAINED:
            axis.axhline(
                UNTRAINED[key],
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="untrained",
            )

    ax = axes[0, 0]
    ax.plot(epochs, [r["test_loss"] for r in records], marker="o", markersize=4)
    ax.set_title("Held-out test loss")
    ax.set_ylabel("smooth L1")

    ax = axes[0, 1]
    ax.plot(
        epochs, [r["effective_rank"] for r in records], marker="o", markersize=4, color="darkorange"
    )
    baseline(ax, "effective_rank")
    ax.axhline(6, color="green", linestyle=":", linewidth=1, alpha=0.7, label="6 = #factors")
    ax.set_title("Effective rank (collapse detector)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    # Selectivity: does entity information outpace context information, or do
    # they rise together (representation encodes everything, no entity subspace)?
    ax = axes[0, 2]
    ax.plot(
        epochs,
        [r["entity_accuracy"] for r in records],
        marker="o",
        markersize=4,
        label="entity acc",
    )
    ax.plot(
        epochs,
        [r["context_accuracy_mean"] for r in records],
        marker="s",
        markersize=4,
        label="context acc (mean)",
    )
    ax.axhline(UNTRAINED["entity_accuracy"], color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(
        UNTRAINED["context_accuracy_mean"], color="red", linestyle=":", linewidth=1, alpha=0.5
    )
    ax.set_title("Linear probes, classification (dashed/dotted = untrained)")
    ax.legend(fontsize=8)

    # raw vs whitened Q_E: a growing gap means raw is tracking output scale,
    # not geometry -- whitened is the trustworthy one.
    ax = axes[1, 0]
    ax.plot(epochs, [r["raw_q_entity"] for r in records], marker="o", markersize=4, label="Q_E raw")
    ax.plot(
        epochs,
        [r["white_q_entity"] for r in records],
        marker="s",
        markersize=4,
        label="Q_E whitened",
    )
    ax.set_title("Counterfactual invariance Q_E (k = rank S_B = 3)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(epochs, [r["mi_ratio"] for r in records], marker="o", markersize=4, color="purple")
    baseline(ax, "mi_ratio")
    ax.set_title("MI(top LDA dir; shape) / H(shape)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    for index in range(len(records[0]["top_eigenvalues"])):
        ax.plot(
            epochs,
            [r["top_eigenvalues"][index] for r in records],
            marker=".",
            markersize=3,
            label=f"λ{index + 1}",
        )
    ax.set_title("Top covariance eigenvalues")
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)

    for row in axes:
        for axis in row:
            axis.grid(alpha=0.3)

    fig.suptitle("Spatial I-JEPA (CNN) on Shapes3D — held-out diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path} ({len(records)} checkpoints)")


if __name__ == "__main__":
    main()
