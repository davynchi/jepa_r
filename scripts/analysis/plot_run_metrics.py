#!/usr/bin/env python3
"""Plot all available training and diagnostic metrics for one run directory."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Series = dict[str, dict[float, float]]

PANELS: tuple[tuple[str, tuple[tuple[str, str], ...], bool], ...] = (
    (
        "Prediction loss",
        (
            ("train/epoch_loss", "train"),
            ("eval/test_loss", "test"),
            ("diag/test_loss", "diagnostic test"),
        ),
        True,
    ),
    (
        "Effective rank",
        (
            ("repr/effective_rank", "online"),
            ("diag/effective_rank", "diagnostic"),
        ),
        False,
    ),
    (
        "Representation scale",
        (
            ("repr/trace_covariance", "trace covariance"),
            ("diag/trace_covariance", "diagnostic trace"),
            ("repr/mean_latent_norm", "mean latent norm"),
        ),
        True,
    ),
    (
        "Held-out probes",
        (
            ("diag/entity_accuracy", "entity accuracy"),
            ("diag/context_accuracy_mean", "context accuracy"),
            ("diag/class_accuracy", "class top-1"),
            ("diag/class_top5_accuracy", "class top-5"),
        ),
        False,
    ),
    (
        "Weighting support",
        (
            ("weighting/effective_sample_size", "ESS"),
            ("weighting/prob_entropy_normalized", "normalized entropy"),
        ),
        True,
    ),
    (
        "Sampling concentration",
        (
            ("weighting/top_10pct_mass", "top 10% mass"),
            ("weighting/prob_max", "max probability"),
            ("corruption/probability_mass", "corrupted mass"),
        ),
        True,
    ),
    (
        "RAS diagnostics",
        (
            ("ras/richness_value", "richness"),
            ("ras/richness_pr", "participation ratio"),
            ("ras/grad_richness_norm", "grad norm"),
            ("ras/positive_fraction", "positive fraction"),
        ),
        False,
    ),
    (
        "Entity geometry",
        (
            ("diag/raw_q_entity", "Q_E raw"),
            ("diag/white_q_entity", "Q_E whitened"),
            ("diag/mi_ratio", "MI ratio"),
        ),
        False,
    ),
    (
        "Corruption selection",
        (
            ("corruption/sampled_fraction", "sampled fraction"),
            ("corruption/probability_mass", "probability mass"),
            ("corruption/mean_probability_ratio", "probability ratio"),
            ("corruption/top_10pct_fraction", "fraction in top 10%"),
        ),
        False,
    ),
)

DIAGNOSTIC_KEYS = {
    "test_loss": "diag/test_loss",
    "effective_rank": "diag/effective_rank",
    "trace_covariance": "diag/trace_covariance",
    "entity_accuracy": "diag/entity_accuracy",
    "context_accuracy_mean": "diag/context_accuracy_mean",
    "class_accuracy": "diag/class_accuracy",
    "class_top5_accuracy": "diag/class_top5_accuracy",
    "raw_q_entity": "diag/raw_q_entity",
    "white_q_entity": "diag/white_q_entity",
    "mi_ratio": "diag/mi_ratio",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/metrics_overview.png",
    )
    return parser.parse_args()


def _add(series: Series, key: str, x: Any, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return
    x_value = float(x)
    y_value = float(value)
    if math.isfinite(x_value) and math.isfinite(y_value):
        series[key][x_value] = y_value


def _load_jsonl(path: Path, series: Series) -> None:
    if not path.exists():
        return
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
        for key, value in scalars.items():
            if key == "train/loss":
                x = scalars.get("train/epoch_fraction", epoch)
            else:
                x = epoch
            _add(series, key, x, value)


def _load_diagnostics(path: Path, series: Series) -> None:
    if not path.exists():
        return
    try:
        records = json.loads(path.read_text())
    except json.JSONDecodeError:
        return
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("epoch"), int | float):
            continue
        for source_key, target_key in DIAGNOSTIC_KEYS.items():
            _add(series, target_key, record["epoch"], record.get(source_key))


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _config_summary(config: dict[str, Any]) -> str:
    spatial = config.get("spatial", {})
    if not isinstance(spatial, dict):
        return ""
    weighting = spatial.get("weighting", {})
    if not isinstance(weighting, dict):
        weighting = {}
    parts = [
        str(spatial.get("dataset", "unknown dataset")),
        str(spatial.get("architecture", spatial.get("model_name", "unknown model"))),
        f"weighting={weighting.get('method', 'unknown')}",
        f"epochs={spatial.get('epochs', '?')}",
        f"batch={spatial.get('batch_size', '?')}",
    ]
    richness = weighting.get("richness_functional")
    if weighting.get("method") == "ras" and richness:
        parts.append(f"R={richness}")
    return " | ".join(parts)


def _plot_panel(
    axis: plt.Axes,
    series: Series,
    title: str,
    entries: tuple[tuple[str, str], ...],
    log_scale: bool,
) -> bool:
    plotted = False
    positive_only = True
    for key, label in entries:
        points = sorted(series.get(key, {}).items())
        if not points:
            continue
        x, y = zip(*points, strict=True)
        axis.plot(x, y, marker=".", markersize=3, linewidth=1.2, label=label)
        plotted = True
        positive_only = positive_only and min(y) > 0
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.25)
    if plotted:
        if log_scale and positive_only:
            axis.set_yscale("log")
        axis.legend(fontsize=7)
    else:
        axis.text(
            0.5,
            0.5,
            "not logged",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        axis.set_xticks([])
        axis.set_yticks([])
    return plotted


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output or run_dir / "metrics_overview.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    series: Series = defaultdict(dict)
    _load_jsonl(run_dir / "metrics.jsonl", series)
    _load_diagnostics(run_dir / "metrics" / "diagnostics.json", series)
    config = _load_config(run_dir)

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    available_panels = 0
    for axis, (title, entries, log_scale) in zip(axes.flat, PANELS, strict=True):
        available_panels += _plot_panel(axis, series, title, entries, log_scale)

    title = run_dir.name
    summary = _config_summary(config)
    fig.suptitle(f"{title}\n{summary}" if summary else title, fontsize=13)
    if available_panels == 0:
        fig.text(
            0.5,
            0.01,
            "Checkpoints exist, but no supported metrics or diagnostics were found.",
            ha="center",
            color="firebrick",
            fontsize=11,
        )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output} ({available_panels}/{len(PANELS)} panels)")


if __name__ == "__main__":
    main()
