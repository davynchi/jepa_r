"""Publication-style PNG plots for the entity/context temporal experiment (Section 9).

The base repository's own reporting is a dependency-free SVG renderer
(:mod:`jepa.reporting`); the spec here explicitly asks for PNG figures, which
requires a raster backend, so this module adds ``matplotlib`` (declared in
``pyproject.toml``). All aggregate plots show mean +/- standard deviation
across seeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_FIGSIZE = (7.0, 4.5)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std())


def _savefig(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_entity_selectivity_vs_timescale(
    records: Sequence[Mapping[str, Any]], out_path: str | Path
) -> None:
    """records: {p_entity_switch, seed, q_entity, entity_accuracy, context_leakage, selectivity}."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics = [
        ("q_entity", "Counterfactual invariance Q_E"),
        ("entity_accuracy", "Entity probe accuracy"),
        ("context_leakage", "Context leakage (R^2 in P_E z)"),
        ("selectivity", "Selectivity score"),
    ]
    p_values = sorted({record["p_entity_switch"] for record in records})
    x = [-np.log(p) for p in p_values]
    for axis, (key, title) in zip(axes.flat, metrics, strict=True):
        means, stds = [], []
        for p in p_values:
            values = [record[key] for record in records if record["p_entity_switch"] == p]
            m, s = _mean_std(values)
            means.append(m)
            stds.append(s)
        axis.errorbar(x, means, yerr=stds, marker="o", capsize=3)
        axis.set_xlabel(r"$-\log p_E$ (entity/context timescale proxy)")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.3)
    _savefig(fig, out_path)


def plot_generalized_eigenvalue_spectrum(
    spectra: Mapping[str, Sequence[float]], out_path: str | Path
) -> None:
    fig, axis = plt.subplots(figsize=_FIGSIZE)
    for label, eigenvalues in spectra.items():
        values = np.asarray(eigenvalues, dtype=float)
        values = np.clip(values, 1e-12, None)
        axis.plot(np.arange(1, len(values) + 1), values, marker="o", label=label, alpha=0.8)
    axis.set_yscale("log")
    axis.set_xlabel("Eigenvector rank")
    axis.set_ylabel("Generalized eigenvalue (log scale)")
    axis.set_title("Generalized eigenvalue spectrum: S_B v = lambda (S_W + eps I) v")
    axis.legend(fontsize=7, loc="best")
    axis.grid(alpha=0.3)
    _savefig(fig, out_path)


def plot_probe_matrix(rows: Sequence[Mapping[str, Any]], out_path: str | Path) -> None:
    labels = [row["representation"] for row in rows]
    entity_accuracy = [row["entity_accuracy"] for row in rows]
    context_r2 = [row["context_r2"] for row in rows]
    x = np.arange(len(labels))
    width = 0.35
    fig, axis = plt.subplots(figsize=_FIGSIZE)
    axis.bar(x - width / 2, entity_accuracy, width, label="Entity accuracy")
    axis.bar(x + width / 2, context_r2, width, label="Context R^2")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("Score")
    axis.set_title("Leakage / selectivity matrix")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    _savefig(fig, out_path)


def plot_counterfactual_distances(
    records: Sequence[Mapping[str, Any]], out_path: str | Path
) -> None:
    labels = [record["label"] for record in records]
    d_same = [record["d_same"] for record in records]
    d_diff = [record["d_diff"] for record in records]
    q_entity = [record["q_entity"] for record in records]
    x = np.arange(len(labels))
    width = 0.3
    fig, axis = plt.subplots(figsize=_FIGSIZE)
    axis.bar(x - width, d_same, width, label="D_same")
    axis.bar(x, d_diff, width, label="D_diff")
    twin = axis.twinx()
    twin.plot(x, q_entity, "ko-", label="Q_E")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("Squared distance in P_E z")
    twin.set_ylabel("Q_E = D_diff / D_same")
    axis.set_title("Counterfactual invariance")
    lines, line_labels = axis.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    axis.legend(lines + twin_lines, line_labels + twin_labels, fontsize=8)
    _savefig(fig, out_path)


def plot_effective_rank_over_training(
    curves: Mapping[str, Sequence[tuple[int, float]]], out_path: str | Path
) -> None:
    fig, axis = plt.subplots(figsize=_FIGSIZE)
    for label, points in curves.items():
        epochs = [p[0] for p in points]
        ranks = [p[1] for p in points]
        axis.plot(epochs, ranks, marker=".", label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Effective rank")
    axis.set_title("Effective rank over training")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    _savefig(fig, out_path)


def plot_prediction_loss_curves(
    curves: Mapping[str, Sequence[tuple[int, float, float]]], out_path: str | Path
) -> None:
    fig, axis = plt.subplots(figsize=_FIGSIZE)
    for label, points in curves.items():
        epochs = [p[0] for p in points]
        train_loss = [p[1] for p in points]
        val_loss = [p[2] for p in points]
        axis.plot(epochs, train_loss, linestyle="--", label=f"{label} (train)")
        axis.plot(epochs, val_loss, linestyle="-", label=f"{label} (val)")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("JEPA loss")
    axis.set_yscale("log")
    axis.set_title("Prediction loss curves")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    _savefig(fig, out_path)


def plot_latent_autocorrelation(
    curves: Mapping[str, tuple[Sequence[int], Sequence[float]]], out_path: str | Path
) -> None:
    fig, axis = plt.subplots(figsize=_FIGSIZE)
    for label, (lags, values) in curves.items():
        axis.plot(lags, values, marker=".", label=label)
    axis.axhline(1 / np.e, color="gray", linestyle=":", label="1/e threshold")
    axis.set_xlabel("Lag (steps)")
    axis.set_ylabel("Normalized autocorrelation")
    axis.set_title("Latent autocorrelation")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    _savefig(fig, out_path)


def plot_latent_pca_scatter(
    latents_2d: Any,
    entity_labels: Sequence[int],
    entity_names: Sequence[str],
    out_path: str | Path,
) -> None:
    """Auxiliary-only 2D PCA visualization of frozen latents, colored by entity
    (Section 2.4.10). Not a primary metric -- entity/context separability is
    assessed by the probes and counterfactual invariance metrics instead."""
    points = np.asarray(latents_2d, dtype=float)
    labels = np.asarray(entity_labels)
    fig, axis = plt.subplots(figsize=_FIGSIZE)
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        name = entity_names[label] if label < len(entity_names) else str(label)
        axis.scatter(points[mask, 0], points[mask, 1], s=8, alpha=0.6, label=name)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("PCA of frozen latents, colored by entity (auxiliary only)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    _savefig(fig, out_path)


def plot_temporal_vs_shuffled(records: Sequence[Mapping[str, Any]], out_path: str | Path) -> None:
    """records: {metric, temporal_mean, temporal_std, shuffled_mean, shuffled_std}."""
    labels = [record["metric"] for record in records]
    x = np.arange(len(labels))
    width = 0.35
    fig, axis = plt.subplots(figsize=_FIGSIZE)
    axis.bar(
        x - width / 2,
        [r["temporal_mean"] for r in records],
        width,
        yerr=[r["temporal_std"] for r in records],
        capsize=3,
        label="temporal",
    )
    axis.bar(
        x + width / 2,
        [r["shuffled_mean"] for r in records],
        width,
        yerr=[r["shuffled_std"] for r in records],
        capsize=3,
        label="shuffled",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_title("Temporal vs. shuffled-target control")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    _savefig(fig, out_path)
