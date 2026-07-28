#!/usr/bin/env python3
"""Analyze live-epoch Q metrics against Shapes3D accuracy, loss, and runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from jepa.analysis.quality_metrics import METRIC_SPECS  # noqa: E402
from scripts.analysis.plot_quality_vs_loss import FORMULAS  # noqa: E402

MIN_CORRELATION_POINTS = 8


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", required=True, type=Path)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray, *, ranks: bool = False) -> float | None:
    if len(x) < MIN_CORRELATION_POINTS or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    if ranks:
        x, y = _ranks(x), _ranks(y)
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _rows(root: Path) -> list[dict]:
    losses = {
        (row["run_id"], row["checkpoint_id"]): row
        for row in _jsonl(root / "checkpoint_losses.jsonl")
    }
    accuracies = {
        (row["run_id"], row["checkpoint_id"]): row
        for row in _jsonl(root / "accuracy.jsonl")
    }
    q_by_name = {spec.name: spec.q_number for spec in METRIC_SPECS}
    output: list[dict] = []
    for record in _jsonl(root / "records.jsonl"):
        expected_panel = (
            "supervised_lda"
            if record["metric_name"] == "q16_entity_consistency"
            else "label_free"
        )
        if record["panel"] != expected_panel:
            continue
        key = (record["run_id"], record["checkpoint_id"])
        loss = losses.get(key)
        accuracy = accuracies.get(key)
        if loss is None or accuracy is None:
            continue
        diagnostics = json.loads(record["diagnostics_json"])
        output.append(
            {
                "run_id": record["run_id"],
                "observation_id": record["checkpoint_id"],
                "epoch": int(loss["epoch"]),
                "global_step": int(loss["global_step"]),
                "q_number": q_by_name[record["metric_name"]],
                "metric_name": record["metric_name"],
                "q_value": record["value"],
                "null_reason": record["null_reason"],
                "classification_accuracy": float(accuracy["accuracy"]),
                "balanced_accuracy": float(accuracy["balanced_accuracy"]),
                "heldout_jepa_loss": float(loss["loss"]),
                "curriculum": diagnostics["curriculum"],
            }
        )
    return sorted(output, key=lambda row: (row["metric_name"], row["epoch"]))


def _correlations(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["q_value"] is not None:
            grouped[row["metric_name"]].append(row)
    q_by_name = {spec.name: spec.q_number for spec in METRIC_SPECS}
    results: list[dict] = []
    for metric_name, metric_rows in sorted(grouped.items()):
        x = np.asarray([row["q_value"] for row in metric_rows], dtype=float)
        epochs = np.asarray([row["epoch"] for row in metric_rows], dtype=int)
        order = np.argsort(epochs)
        x = x[order]
        for outcome in ("classification_accuracy", "heldout_jepa_loss"):
            y = np.asarray([row[outcome] for row in metric_rows], dtype=float)[order]
            delta_valid = len(x) >= MIN_CORRELATION_POINTS + 1
            results.append(
                {
                    "q_number": q_by_name[metric_name],
                    "metric_name": metric_name,
                    "outcome": outcome,
                    "n": len(x),
                    "pearson": _correlation(x, y),
                    "spearman": _correlation(x, y, ranks=True),
                    "delta_n": len(x) - 1 if delta_valid else 0,
                    "delta_pearson": (
                        _correlation(np.diff(x), np.diff(y)) if delta_valid else None
                    ),
                    "delta_spearman": (
                        _correlation(np.diff(x), np.diff(y), ranks=True)
                        if delta_valid
                        else None
                    ),
                }
            )
    return results


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(color="#E2E8F0", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#475569", labelsize=8)


def _scatter_pages(
    output: Path,
    rows: list[dict],
    correlations: list[dict],
    *,
    outcome: str,
) -> None:
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["q_value"] is not None:
            by_metric[row["metric_name"]].append(row)
    correlation_map = {
        row["metric_name"]: row for row in correlations if row["outcome"] == outcome
    }
    cmap = plt.get_cmap("viridis")
    normalize = Normalize(vmin=1, vmax=max((row["epoch"] for row in rows), default=100))
    for page, start in enumerate(range(0, len(METRIC_SPECS), 7), start=1):
        specs = list(METRIC_SPECS)[start : start + 7]
        figure, axes = plt.subplots(2, 4, figsize=(15, 10.2))
        for axis, spec in zip(axes.flat, specs, strict=False):
            points = sorted(by_metric.get(spec.name, []), key=lambda row: row["epoch"])
            x = np.asarray([row["q_value"] for row in points], dtype=float)
            y = np.asarray([row[outcome] for row in points], dtype=float)
            epochs = np.asarray([row["epoch"] for row in points], dtype=int)
            if len(points):
                axis.scatter(
                    x,
                    y,
                    c=epochs,
                    cmap=cmap,
                    norm=normalize,
                    s=24,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
                if len(points) >= MIN_CORRELATION_POINTS and np.ptp(x) > 1e-12:
                    coefficients = np.polyfit(x, y, 1)
                    grid = np.linspace(float(x.min()), float(x.max()), 100)
                    axis.plot(
                        grid,
                        np.polyval(coefficients, grid),
                        color="#64748B",
                        linewidth=1,
                    )
                for point in points:
                    if point["epoch"] == 1 or point["epoch"] % 10 == 0:
                        axis.annotate(
                            str(point["epoch"]),
                            (point["q_value"], point[outcome]),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=5.8,
                            color="#475569",
                        )
            result = correlation_map.get(spec.name, {})
            r = result.get("pearson")
            rho = result.get("spearman")
            axis.set_title(f"Q{spec.q_number}", loc="left", fontsize=10, fontweight="bold")
            axis.text(
                0.98,
                0.96,
                f"n={len(points)}  r={'NA' if r is None else f'{r:+.2f}'}  "
                f"ρ={'NA' if rho is None else f'{rho:+.2f}'}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="#475569",
            )
            axis.set_xlabel("Q value", fontsize=8)
            axis.set_ylabel(
                "Classification accuracy"
                if outcome == "classification_accuracy"
                else "Held-out JEPA loss",
                fontsize=8,
            )
            if outcome == "heldout_jepa_loss":
                axis.set_yscale("log")
            display_name, formula = FORMULAS[spec.name]
            axis.text(
                0,
                -0.25,
                f"{display_name}\n{formula}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="#334155",
            )
            axis.xaxis.set_major_locator(MaxNLocator(5))
            _style_axis(axis)
        for axis in list(axes.flat)[len(specs) :]:
            axis.remove()
        title_outcome = (
            "classification accuracy"
            if outcome == "classification_accuracy"
            else "held-out JEPA loss"
        )
        figure.suptitle(
            f"Uniform live-epoch Q metrics vs {title_outcome} · page {page}",
            x=0.025,
            y=0.985,
            ha="left",
            fontsize=15,
            fontweight="bold",
        )
        figure.text(
            0.025,
            0.945,
            "Each point is one epoch; color encodes epoch. "
            "Labels mark epoch 1 and multiples of 10.",
            fontsize=8,
            color="#64748B",
        )
        colorbar_axis = figure.add_axes((0.86, 0.94, 0.12, 0.012))
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=normalize, cmap=cmap),
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_label("Epoch", fontsize=7)
        colorbar.ax.tick_params(labelsize=7)
        figure.subplots_adjust(
            left=0.07, right=0.985, top=0.88, bottom=0.11, hspace=0.72, wspace=0.34
        )
        short = "accuracy" if outcome == "classification_accuracy" else "loss"
        figure.savefig(
            output / f"quality_vs_{short}_page_{page}.png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def _correlation_summary(
    output: Path, correlations: list[dict], *, outcome: str
) -> None:
    valid = [
        row
        for row in correlations
        if row["outcome"] == outcome
        and row["pearson"] is not None
        and row["spearman"] is not None
    ]
    valid.sort(key=lambda row: max(abs(row["pearson"]), abs(row["spearman"])))
    figure, axis = plt.subplots(figsize=(11, max(5, 0.38 * len(valid))))
    positions = np.arange(len(valid))
    axis.barh(
        positions - 0.18,
        [row["pearson"] for row in valid],
        height=0.36,
        color="#2563EB",
        label="Pearson r",
    )
    axis.barh(
        positions + 0.18,
        [row["spearman"] for row in valid],
        height=0.36,
        color="#D97706",
        label="Spearman ρ",
    )
    axis.set_yticks(
        positions,
        [f"Q{row['q_number']} · {row['metric_name']}" for row in valid],
    )
    axis.set_xlim(-1, 1)
    axis.axvline(0, color="#0F172A", linewidth=0.8)
    axis.set_xlabel("Correlation")
    axis.set_title(
        "Live-epoch correlations with "
        + ("classification accuracy" if outcome == "classification_accuracy" else "JEPA loss"),
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False)
    _style_axis(axis)
    figure.tight_layout()
    short = "accuracy" if outcome == "classification_accuracy" else "loss"
    figure.savefig(
        output / f"correlation_summary_{short}.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _timing_outputs(root: Path, output: Path) -> list[dict]:
    rows = _jsonl(root / "timings.jsonl")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if int(row["epoch"]) == 1:
            continue
        grouped[(row["kind"], row["name"])].append(float(row["seconds"]))
    summary = [
        {
            "kind": kind,
            "name": name,
            "epochs": len(values),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95)),
        }
        for (kind, name), values in sorted(grouped.items())
    ]
    with (output / "timing_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else ["kind"])
        writer.writeheader()
        writer.writerows(summary)
    stage_summary = sorted(
        (row for row in summary if row["kind"] == "stage"),
        key=lambda row: row["median_seconds"],
    )
    if stage_summary:
        figure, axis = plt.subplots(figsize=(11, max(5, 0.35 * len(stage_summary))))
        positions = np.arange(len(stage_summary))
        axis.barh(
            positions,
            [row["median_seconds"] for row in stage_summary],
            color="#2563EB",
            label="median",
        )
        axis.scatter(
            [row["p95_seconds"] for row in stage_summary],
            positions,
            color="#D97706",
            s=24,
            label="p95",
            zorder=3,
        )
        axis.set_yticks(positions, [row["name"] for row in stage_summary])
        axis.set_xlabel("Seconds")
        axis.set_title("Online quality timing by stage · epochs 2–100", loc="left")
        axis.legend(frameon=False)
        _style_axis(axis)
        figure.tight_layout()
        figure.savefig(
            output / "timing_summary.png", dpi=180, bbox_inches="tight", facecolor="white"
        )
        plt.close(figure)
    by_stage: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["kind"] == "stage" and row["name"] in {
            "train_epoch",
            "quality_total",
            "factorization_label_free",
            "factorization_supervised",
        }:
            by_stage[row["name"]].append((int(row["epoch"]), float(row["seconds"])))
    if by_stage:
        figure, axis = plt.subplots(figsize=(11, 5.5))
        for name, points in sorted(by_stage.items()):
            points.sort()
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker=".",
                linewidth=1,
                label=name,
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Seconds")
        axis.set_title("Training and diagnostic time by epoch", loc="left")
        axis.legend(frameon=False)
        _style_axis(axis)
        figure.tight_layout()
        figure.savefig(
            output / "timing_by_epoch.png", dpi=180, bbox_inches="tight", facecolor="white"
        )
        plt.close(figure)
    return summary


def _write_tables(output: Path, rows: list[dict], correlations: list[dict]) -> None:
    with (output / "epoch_quality_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "correlations.json").write_text(
        json.dumps(correlations, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    fields = sorted({key for row in correlations for key in row})
    with (output / "correlations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(correlations)


def _write_interpretation(
    output: Path, rows: list[dict], correlations: list[dict], timings: list[dict]
) -> None:
    epochs = sorted({row["epoch"] for row in rows})
    missing: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["q_value"] is None:
            missing[str(row["null_reason"])] += 1

    def strongest(outcome: str, coefficient: str) -> str:
        candidates = [
            row
            for row in correlations
            if row["outcome"] == outcome and row[coefficient] is not None
        ]
        candidates.sort(key=lambda row: abs(row[coefficient]), reverse=True)
        symbol = "r" if coefficient == "pearson" else "rho"
        return "\n".join(
            f"  Q{row['q_number']}: {symbol}={row[coefficient]:+.3f}, n={row['n']}"
            for row in candidates[:8]
        )

    stage = {
        row["name"]: row
        for row in timings
        if row["kind"] == "stage"
    }
    train_median = stage.get("train_epoch", {}).get("median_seconds", "NA")
    quality_median = stage.get("quality_total", {}).get("median_seconds", "NA")
    label_free_median = stage.get("factorization_label_free", {}).get(
        "median_seconds", "NA"
    )
    supervised_median = stage.get("factorization_supervised", {}).get(
        "median_seconds", "NA"
    )
    text = f"""ИНТЕРПРЕТАЦИЯ ONLINE Q-МЕТРИК SHAPES3D

Наблюдения: {len(epochs)} эпох ({epochs[0] if epochs else 'NA'}–{epochs[-1] if epochs else 'NA'}).
Sampling: uniform. Каждая точка соответствует одной эпохе одной training trajectory.

Пирсон r измеряет линейную связь. Спирмен rho измеряет монотонную ранговую связь.
Эпохи не являются независимыми наблюдениями, поэтому коэффициенты описательные:
обычные iid p-values для этого запуска не интерпретируются.

Сильнейшие связи с accuracy по Пирсону:
{strongest('classification_accuracy', 'pearson')}

Сильнейшие связи с accuracy по Спирмену:
{strongest('classification_accuracy', 'spearman')}

Сильнейшие связи с held-out JEPA loss по Пирсону:
{strongest('heldout_jepa_loss', 'pearson')}

Сильнейшие связи с held-out JEPA loss по Спирмену:
{strongest('heldout_jepa_loss', 'spearman')}

Пропуски:
{chr(10).join(f'  {reason}: {count}' for reason, count in sorted(missing.items())) or '  нет'}

Timing исключает эпоху 1 как warm-up. GPU/MPS стадии синхронизируются до и после
измерения. Общие зависимости не приписываются повторно каждому Q.

Median train epoch: {train_median} s
Median all-Q diagnostics: {quality_median} s
Median label-free factorization: {label_free_median} s
Median supervised factorization: {supervised_median} s

Файлы:
  correlations.csv/json
  epoch_quality_values.csv
  timing_summary.csv/png
  timing_by_epoch.png
  quality_vs_accuracy_page_1..3.png
  quality_vs_loss_page_1..3.png
"""
    (output / "INTERPRETATION_RU.txt").write_text(text)


def main() -> None:
    arguments = _args()
    root = arguments.quality_root.expanduser().resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows = _rows(root)
    correlations = _correlations(rows)
    _write_tables(output, rows, correlations)
    _scatter_pages(
        output, rows, correlations, outcome="classification_accuracy"
    )
    _scatter_pages(output, rows, correlations, outcome="heldout_jepa_loss")
    _correlation_summary(
        output, correlations, outcome="classification_accuracy"
    )
    _correlation_summary(output, correlations, outcome="heldout_jepa_loss")
    timings = _timing_outputs(root, output)
    _write_interpretation(output, rows, correlations, timings)
    print(
        f"epochs={len({row['epoch'] for row in rows})} "
        f"metrics={len({row['metric_name'] for row in rows})} output={output}"
    )


if __name__ == "__main__":
    main()
