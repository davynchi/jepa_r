#!/usr/bin/env python3
"""Build compact or full reports for Moving-MNIST JEPA experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from jepa.video_utils import load_yaml

COMPACT_QUALITY = [
    ("attentive_direction_accuracy", "Motion direction accuracy ↑"),
    ("speed_rmse", "Speed RMSE ↓"),
    ("future_fde", "Future position FDE ↓"),
    ("digit_accuracy", "Digit identity accuracy ↑"),
]
FULL_QUALITY = [
    ("attentive_direction_accuracy", "Motion direction accuracy ↑"),
    ("speed_rmse", "Speed RMSE ↓"),
    ("future_ade", "Future position ADE ↓"),
    ("future_fde", "Future position FDE ↓"),
    ("bounce_macro_f1", "Bounce macro-F1 ↑"),
    ("digit_accuracy", "Digit identity accuracy ↑"),
]
SAMPLER_METRICS = [
    ("sampler_ess_fraction", "Sampler ESS / dataset ↑"),
    ("dataset_coverage", "Epoch dataset coverage ↑"),
    ("repeat_fraction", "Epoch repeat fraction ↓"),
    ("selected_bounce_fraction", "Selected bounce fraction"),
    ("selected_speed_mean", "Selected mean speed"),
    ("next_sampler_ess_fraction", "Next-epoch sampler ESS ↑"),
]
KNOWN_DISPLAY_NAMES = {
    "uniform_shuffle": "Uniform shuffle",
    "uniform_replacement": "Uniform replacement",
    "soft_surprise": "Soft surprise",
    "warmup_learnable_surprise": "Learnable surprise",
    "progress_quarantine_surprise": "Progress + quarantine",
    "random_init": "Random encoder",
}
LOWER_IS_BETTER = {
    "speed_rmse",
    "future_ade",
    "future_fde",
    "jepa_loss",
    "unweighted_jepa_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--x-axis",
        choices=["clips_seen", "step", "total_forward_clips"],
        default="clips_seen",
    )
    parser.add_argument("--view", choices=["compact", "full"], default="compact")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--show-mean-probe", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_runs(root: Path, include_incomplete: bool) -> list[Path]:
    runs: list[Path] = []
    for status_path in sorted(root.glob("*/status.json")):
        status = json.loads(status_path.read_text())
        if status.get("state") == "completed" or include_incomplete:
            runs.append(status_path.parent)
    if not runs:
        raise SystemExit(f"no usable runs found below {root}")
    return runs


def series_id(row: dict[str, Any]) -> str:
    value = row.get("run_label")
    if value not in {None, ""}:
        return str(value)
    return str(row.get("strategy", "unknown"))


def series_display(row: dict[str, Any]) -> str:
    value = row.get("display_name")
    if value not in {None, ""}:
        return str(value)
    series = series_id(row)
    return KNOWN_DISPLAY_NAMES.get(series, series)


def x_value(row: dict[str, Any], mode: str) -> float:
    if mode == "total_forward_clips":
        return float(row.get("clips_seen", 0.0)) + float(
            row.get("scoring_clips", 0.0)
        )
    return float(row.get(mode, 0.0))


def collect_long_rows(run_dirs: Iterable[Path], filename: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for row in read_csv(run_dir / filename):
            converted: dict[str, Any] = dict(row)
            for key in [
                "epoch",
                "step",
                "clips_seen",
                "scoring_clips",
                "seed",
                "value",
            ]:
                if key in converted and converted[key] != "":
                    converted[key] = float(converted[key])
            converted["run_dir"] = str(run_dir)
            converted["series"] = series_id(converted)
            converted["series_display"] = series_display(converted)
            result.append(converted)
    return result


def display_lookup(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        result[series_id(row)] = series_display(row)
    return result


def aggregate_curves(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    split: str,
    x_axis: str,
) -> dict[str, list[tuple[float, float, float]]]:
    by_series_seed_x: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("metric") != metric or row.get("split") != split:
            continue
        series = series_id(row)
        seed = int(float(row.get("seed", 0)))
        x = x_value(row, x_axis)
        by_series_seed_x[(series, seed, x)].append(float(row["value"]))

    by_series_x: dict[tuple[str, float], list[float]] = defaultdict(list)
    for (series, _seed, x), values in by_series_seed_x.items():
        by_series_x[(series, x)].append(float(np.mean(values)))

    result: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (series, x), values in by_series_x.items():
        result[series].append((x, float(np.mean(values)), float(np.std(values))))
    return {series: sorted(points) for series, points in result.items()}


def selection_x_by_series(run_dirs: Iterable[Path], x_axis: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for run_dir in run_dirs:
        path = run_dir / "checkpoint_selection.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        series = str(payload.get("run_label", run_dir.name))
        if x_axis == "total_forward_clips":
            x = float(payload.get("clips_seen", 0.0)) + float(
                payload.get("scoring_clips", 0.0)
            )
        else:
            x = float(payload.get(x_axis, 0.0))
        values[series].append(x)
    return {series: float(np.mean(xs)) for series, xs in values.items()}


def strategy_colors(series: Iterable[str]) -> dict[str, str]:
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ordered = sorted(item for item in set(series) if item != "random_init")
    return {item: palette[index % len(palette)] for index, item in enumerate(ordered)}


def _random_reference(
    axis: plt.Axes,
    points: list[tuple[float, float, float]],
    *,
    label: str,
) -> None:
    if not points:
        return
    mean = float(np.mean([point[1] for point in points]))
    std = float(np.mean([point[2] for point in points]))
    line = axis.axhline(mean, linestyle=":", linewidth=1.8, label=label)
    if std > 0.0:
        axis.axhspan(mean - std, mean + std, alpha=0.10, color=line.get_color())


def plot_curves(
    axis: plt.Axes,
    curves: dict[str, list[tuple[float, float, float]]],
    *,
    colors: dict[str, str],
    displays: dict[str, str],
    selected_x: dict[str, float] | None = None,
    linestyle: str = "-",
    label_suffix: str = "",
    show_final: bool = True,
    show_selected: bool = True,
) -> None:
    selected_x = selected_x or {}
    for series, points in sorted(curves.items()):
        if not points:
            continue
        label = displays.get(series, KNOWN_DISPLAY_NAMES.get(series, series)) + label_suffix
        if series == "random_init":
            _random_reference(axis, points, label=label)
            continue
        x = np.asarray([point[0] for point in points])
        mean = np.asarray([point[1] for point in points])
        std = np.asarray([point[2] for point in points])
        line = axis.plot(
            x,
            mean,
            label=label,
            linewidth=2.2,
            linestyle=linestyle,
            color=colors.get(series),
        )[0]
        if np.any(std > 0.0):
            axis.fill_between(x, mean - std, mean + std, alpha=0.14, color=line.get_color())
        if show_final:
            axis.scatter(x[-1], mean[-1], marker="o", s=28, color=line.get_color(), zorder=4)
        if show_selected and series in selected_x:
            index = int(np.argmin(np.abs(x - selected_x[series])))
            axis.scatter(
                x[index],
                mean[index],
                marker="D",
                s=48,
                facecolor="white",
                edgecolor=line.get_color(),
                linewidth=1.8,
                zorder=5,
            )


def _style_axis(axis: plt.Axes, title: str, x_axis: str) -> None:
    axis.set_title(title, fontsize=11)
    axis.set_xlabel(x_axis.replace("_", " "), fontsize=9)
    axis.grid(alpha=0.22)
    axis.tick_params(labelsize=9)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def deduplicated_legend(axes: Iterable[plt.Axes]) -> tuple[list[Any], list[str]]:
    handles: list[Any] = []
    labels: list[str] = []
    for axis in axes:
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels, strict=True):
            if label and label != "_nolegend_" and label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels


def quality_dashboard(
    rows: list[dict[str, Any]],
    *,
    run_dirs: list[Path],
    x_axis: str,
    num_directions: int,
    output: Path,
    view: str,
    show_mean_probe: bool,
) -> None:
    metrics = COMPACT_QUALITY if view == "compact" else FULL_QUALITY
    shape = (2, 2) if view == "compact" else (2, 3)
    fig, axes = plt.subplots(*shape, figsize=(12.5, 7.3) if view == "compact" else (14.5, 7.8), constrained_layout=True)
    displays = display_lookup(rows)
    colors = strategy_colors(displays)
    selected = selection_x_by_series(run_dirs, x_axis)

    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        curves = aggregate_curves(rows, metric=metric, split="validation", x_axis=x_axis)
        if not curves and metric == "attentive_direction_accuracy":
            curves = aggregate_curves(rows, metric="direction_accuracy", split="validation", x_axis=x_axis)
        plot_curves(
            axis,
            curves,
            colors=colors,
            displays=displays,
            selected_x=selected,
        )
        if metric == "attentive_direction_accuracy" and show_mean_probe:
            mean_curves = aggregate_curves(
                rows,
                metric="mean_direction_accuracy",
                split="validation",
                x_axis=x_axis,
            )
            plot_curves(
                axis,
                mean_curves,
                colors=colors,
                displays=displays,
                selected_x={},
                linestyle="--",
                label_suffix=" · mean pool",
                show_final=False,
                show_selected=False,
            )
        if metric == "attentive_direction_accuracy":
            axis.axhline(
                1.0 / num_directions,
                linestyle=":",
                linewidth=1.1,
                label=f"Chance ({1.0 / num_directions:.3f})",
            )
        if metric == "digit_accuracy":
            axis.axhline(0.1, linestyle=":", linewidth=1.1, label="Chance (0.10)")
        _style_axis(axis, title, x_axis)

    handles, labels = deduplicated_legend(axes.flat)
    handles.extend(
        [
            Line2D([0], [0], marker="D", markerfacecolor="white", linestyle="None"),
            Line2D([0], [0], marker="o", linestyle="None"),
        ]
    )
    labels.extend(["Selected checkpoint", "Final checkpoint"])
    fig.suptitle("Frozen downstream quality on Moving-MNIST", fontsize=15, fontweight="bold")
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(labels), 5),
        frameon=False,
    )
    save_figure(fig, output)


def warmup_x(run_dirs: list[Path], x_axis: str) -> float | None:
    if not run_dirs:
        return None
    config = load_yaml(run_dirs[0] / "config.yaml")
    training = config["training"]
    data = config["data"]
    epochs = int(training.get("lr_warmup_epochs", 0))
    if epochs <= 0:
        return None
    steps_per_epoch = int(data["train_samples"]) // int(training["batch_size"])
    if x_axis == "step":
        return float(epochs * steps_per_epoch)
    return float(epochs * steps_per_epoch * int(training["batch_size"]))


def add_warmup_marker(axis: plt.Axes, value: float | None) -> None:
    if value is not None:
        axis.axvline(value, linestyle=":", linewidth=1.0, alpha=0.55)


def plot_metric_variants(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    variants: list[tuple[str, str, str]],
    *,
    split: str,
    x_axis: str,
    colors: dict[str, str],
    displays: dict[str, str],
) -> None:
    for metric, suffix, linestyle in variants:
        curves = aggregate_curves(rows, metric=metric, split=split, x_axis=x_axis)
        plot_curves(
            axis,
            curves,
            colors=colors,
            displays=displays,
            selected_x={},
            linestyle=linestyle,
            label_suffix=suffix,
            show_final=False,
            show_selected=False,
        )


def training_dashboard(
    rows: list[dict[str, Any]],
    *,
    x_axis: str,
    warmup: float | None,
    output: Path,
    view: str,
) -> None:
    displays = display_lookup(rows)
    colors = strategy_colors(displays)
    if view == "compact":
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.3), constrained_layout=True)
        loss_metric = (
            "unweighted_jepa_loss"
            if any(row.get("metric") == "unweighted_jepa_loss" for row in rows)
            else "jepa_loss"
        )
        plot_metric_variants(
            axes.flat[0],
            rows,
            [(loss_metric, " train", "--")],
            split="train",
            x_axis=x_axis,
            colors=colors,
            displays=displays,
        )
        plot_metric_variants(
            axes.flat[0],
            rows,
            [(loss_metric, " validation", "-")],
            split="validation",
            x_axis=x_axis,
            colors=colors,
            displays=displays,
        )
        _style_axis(axes.flat[0], "Comparable unweighted JEPA loss ↓", x_axis)

        plot_metric_variants(
            axes.flat[1],
            rows,
            [("context_token_effective_rank", "", "-")],
            split="validation",
            x_axis=x_axis,
            colors=colors,
            displays=displays,
        )
        _style_axis(axes.flat[1], "Context token effective rank ↑", x_axis)

        plot_metric_variants(
            axes.flat[2],
            rows,
            [
                ("pairing_gap_shuffled_context", " shuffled context", "-"),
                ("pairing_gap_reversed_context", " reversed time", "--"),
            ],
            split="validation",
            x_axis=x_axis,
            colors=colors,
            displays=displays,
        )
        axes.flat[2].axhline(0.0, linestyle=":", linewidth=1.0)
        _style_axis(axes.flat[2], "Context-sensitivity gap ↑", x_axis)

        lr_curves = aggregate_curves(rows, metric="learning_rate", split="train", x_axis=x_axis)
        plot_curves(
            axes.flat[3],
            lr_curves,
            colors=colors,
            displays=displays,
            selected_x={},
            label_suffix=" · LR",
            show_final=False,
            show_selected=False,
        )
        momentum_axis = axes.flat[3].twinx()
        ema_curves = aggregate_curves(rows, metric="ema_momentum", split="train", x_axis=x_axis)
        plot_curves(
            momentum_axis,
            ema_curves,
            colors=colors,
            displays=displays,
            selected_x={},
            linestyle="--",
            label_suffix=" · EMA",
            show_final=False,
            show_selected=False,
        )
        axes.flat[3].set_ylabel("learning rate", fontsize=8)
        momentum_axis.set_ylabel("EMA momentum", fontsize=8)
        momentum_axis.tick_params(labelsize=8)
        _style_axis(axes.flat[3], "Optimization schedule", x_axis)
        extra_axes = [momentum_axis]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), constrained_layout=True)
        plot_metric_variants(
            axes.flat[0],
            rows,
            [("jepa_loss", " weighted train", "--"), ("unweighted_jepa_loss", " unweighted train", ":")],
            split="train",
            x_axis=x_axis,
            colors=colors,
            displays=displays,
        )
        plot_metric_variants(
            axes.flat[0],
            rows,
            [("jepa_loss", " weighted val", "-"), ("unweighted_jepa_loss", " unweighted val", "-.")],
            split="validation",
            x_axis=x_axis,
            colors=colors,
            displays=displays,
        )
        _style_axis(axes.flat[0], "JEPA objective diagnostics", x_axis)

        plot_metric_variants(
            axes.flat[1],
            rows,
            [("context_pooled_effective_rank", " pooled", "--"), ("context_token_effective_rank", " token", "-")],
            split="validation",
            x_axis=x_axis,
            colors=colors,
            displays=displays,
        )
        _style_axis(axes.flat[1], "Context effective rank ↑", x_axis)

        plot_metric_variants(
            axes.flat[2],
            rows,
            [("foreground_jepa_loss", " foreground", "-"), ("background_jepa_loss", " background", "--")],
            split="validation",
            x_axis=x_axis,
            colors=colors,
            displays=displays,
        )
        _style_axis(axes.flat[2], "Foreground/background prediction loss", x_axis)

        plot_metric_variants(
            axes.flat[3],
            rows,
            [
                ("pairing_gap_shuffled_context", " shuffled context", "-"),
                ("pairing_gap_zero_context", " zero context", "--"),
                ("pairing_gap_reversed_context", " reversed time", "-."),
                ("pairing_gap_shuffled_target", " shuffled target", ":"),
            ],
            split="validation",
            x_axis=x_axis,
            colors=colors,
            displays=displays,
        )
        axes.flat[3].axhline(0.0, linestyle=":", linewidth=1.0)
        _style_axis(axes.flat[3], "Counterfactual loss gap ↑", x_axis)

        lr_curves = aggregate_curves(rows, metric="learning_rate", split="train", x_axis=x_axis)
        plot_curves(
            axes.flat[4], lr_curves, colors=colors, displays=displays, selected_x={}, label_suffix=" · LR", show_final=False, show_selected=False
        )
        momentum_axis = axes.flat[4].twinx()
        ema_curves = aggregate_curves(rows, metric="ema_momentum", split="train", x_axis=x_axis)
        plot_curves(
            momentum_axis, ema_curves, colors=colors, displays=displays, selected_x={}, linestyle="--", label_suffix=" · EMA", show_final=False, show_selected=False
        )
        axes.flat[4].set_ylabel("learning rate", fontsize=8)
        momentum_axis.set_ylabel("EMA momentum", fontsize=8)
        _style_axis(axes.flat[4], "Optimization schedule", x_axis)

        gradient_curves = aggregate_curves(rows, metric="gradient_norm", split="train", x_axis=x_axis)
        plot_curves(
            axes.flat[5], gradient_curves, colors=colors, displays=displays, selected_x={}, show_final=False, show_selected=False
        )
        _style_axis(axes.flat[5], "Gradient norm", x_axis)
        extra_axes = [momentum_axis]

    for axis in axes.flat:
        add_warmup_marker(axis, warmup)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7, frameon=False, loc="best")
    for axis in extra_axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Training and representation diagnostics", fontsize=15, fontweight="bold")
    save_figure(fig, output)


def sampler_dashboard(
    rows: list[dict[str, Any]],
    *,
    x_axis: str,
    output: Path,
) -> None:
    displays = display_lookup(rows)
    colors = strategy_colors(displays)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, SAMPLER_METRICS, strict=True):
        curves = aggregate_curves(rows, metric=metric, split="sampler", x_axis=x_axis)
        plot_curves(
            axis,
            curves,
            colors=colors,
            displays=displays,
            selected_x={},
            show_final=False,
            show_selected=False,
        )
        _style_axis(axis, title, x_axis)
    handles, labels = deduplicated_legend(axes.flat)
    fig.suptitle("Sampler diagnostics", fontsize=15, fontweight="bold")
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save_figure(fig, output)


def final_test_by_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    displays: dict[str, str] = {}
    for row in rows:
        if row.get("split") != "test":
            continue
        series = series_id(row)
        seed = int(float(row.get("seed", 0)))
        grouped[(series, seed)][str(row["metric"])] = float(row["value"])
        displays[series] = series_display(row)
    output: list[dict[str, Any]] = []
    for (series, seed), metrics in sorted(grouped.items()):
        output.append(
            {
                "run_label": series,
                "display_name": displays.get(series, series),
                "seed": seed,
                **metrics,
            }
        )
    return output


def aggregate_final(rows: list[dict[str, Any]], metrics: list[tuple[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["run_label"])].append(row)
    summary: list[dict[str, Any]] = []
    for series, seed_rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "run_label": series,
            "display_name": seed_rows[0].get("display_name", series),
            "seeds": len(seed_rows),
        }
        for metric, _ in metrics:
            values = [float(row[metric]) for row in seed_rows if metric in row]
            if values:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values))
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def final_test_dashboard(
    summary: list[dict[str, Any]],
    metrics: list[tuple[str, str]],
    output: Path,
) -> None:
    if not summary:
        return
    shape = (2, 2) if len(metrics) == 4 else (2, 3)
    fig, axes = plt.subplots(*shape, figsize=(12.5, 7.3) if len(metrics) == 4 else (14.5, 7.8), constrained_layout=True)
    series = [str(row["run_label"]) for row in summary]
    lookup = {str(row["run_label"]): row for row in summary}
    x = np.arange(len(series))
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        means = [float(lookup[item].get(f"{metric}_mean", np.nan)) for item in series]
        stds = [float(lookup[item].get(f"{metric}_std", 0.0)) for item in series]
        axis.bar(x, means, yerr=stds, capsize=3, alpha=0.82)
        axis.set_title(label, fontsize=11)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [str(lookup[item].get("display_name", item)) for item in series],
            rotation=25,
            ha="right",
            fontsize=8,
        )
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Held-out test metrics at selected checkpoint", fontsize=15, fontweight="bold")
    save_figure(fig, output)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output_dir) if args.output_dir else root / "report"
    output.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(root, args.include_incomplete)
    pretrain_rows = collect_long_rows(runs, "pretrain_metrics.csv")
    sampler_rows = collect_long_rows(runs, "sampler_metrics.csv")
    probe_rows = collect_long_rows(runs, "probe_metrics.csv")
    if not probe_rows:
        raise SystemExit("no probe_metrics.csv files found; run eval_moving_mnist_probes.py first")

    config = load_yaml(runs[0] / "config.yaml")
    metrics = COMPACT_QUALITY if args.view == "compact" else FULL_QUALITY
    quality_dashboard(
        probe_rows,
        run_dirs=runs,
        x_axis=args.x_axis,
        num_directions=int(config["data"].get("num_directions", 8)),
        output=output / "quality_dashboard",
        view=args.view,
        show_mean_probe=args.show_mean_probe,
    )
    training_dashboard(
        pretrain_rows,
        x_axis=args.x_axis,
        warmup=warmup_x(runs, args.x_axis),
        output=output / "diagnostics_dashboard",
        view=args.view,
    )
    adaptive_present = any(
        str(row.get("strategy")) not in {"uniform_shuffle", "random_init"}
        for row in sampler_rows
    )
    if adaptive_present:
        sampler_dashboard(sampler_rows, x_axis=args.x_axis, output=output / "sampler_dashboard")

    by_seed = final_test_by_seed(probe_rows)
    summary = aggregate_final(by_seed, metrics)
    write_csv(output / "final_test_by_seed.csv", by_seed)
    write_csv(output / "final_test_summary.csv", summary)
    final_test_dashboard(summary, metrics, output / "final_test_dashboard")
    print(output)


if __name__ == "__main__":
    main()
