"""Dependency-free SVG learning curves for run and sweep artifacts."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
    "#475569",
    "#ca8a04",
)


def load_history(path: Path) -> list[dict[str, Any]]:
    """Load the public history CSV without depending on trainer internals."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = dict(raw)
            row["epoch"] = int(raw["epoch"])
            for key, value in raw.items():
                if key not in {"run_id", "split", "epoch"} and not key.endswith("_reason"):
                    row[key] = None if value == "" else float(value)
            rows.append(row)
    return rows


def _points(rows: Iterable[Mapping[str, Any]], field: str, split: str) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    for row in rows:
        value = row.get(field)
        if row.get("split") != split or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            result.append((int(row["epoch"]), numeric))
    return sorted(result)


def _path(
    points: list[tuple[int, float]],
    *,
    x_min: int,
    x_max: int,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
    logarithmic: bool,
) -> str:
    transformed = [
        (epoch, math.log10(max(value, 1e-12)) if logarithmic else value) for epoch, value in points
    ]
    x_span = max(x_max - x_min, 1)
    y_span = max(y_max - y_min, 1e-12)
    coordinates = [
        (
            left + (epoch - x_min) / x_span * width,
            top + height - (value - y_min) / y_span * height,
        )
        for epoch, value in transformed
    ]
    return " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}" for index, (x, y) in enumerate(coordinates)
    )


def _render(
    title: str,
    panels: list[tuple[str, bool, list[tuple[str, list[tuple[int, float]]]]]],
) -> str:
    width = 1280
    panel_height = 260
    height = 70 + panel_height * len(panels)
    left = 80.0
    plot_width = width - left - 260
    plot_height = 165.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#172033}"
        ".title{font-size:24px;font-weight:700}.panel{font-size:16px;font-weight:650}"
        ".axis{stroke:#94a3b8;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}"
        ".curve{fill:none;stroke-width:2}.dot{stroke:none}.legend{font-size:11px}</style>",
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        f'<text x="30" y="38" class="title">{escape(title)}</text>',
    ]
    for panel_index, (panel_title, logarithmic, curves) in enumerate(panels):
        top = 70.0 + panel_index * panel_height
        plot_top = top + 35
        all_points = [point for _, curve in curves for point in curve]
        parts.append(f'<text x="30" y="{top + 18:.2f}" class="panel">{escape(panel_title)}</text>')
        if not all_points:
            parts.append(f'<text x="{left}" y="{plot_top + 25:.2f}" fill="#64748b">No data</text>')
            continue
        x_min = min(epoch for epoch, _ in all_points)
        x_max = max(epoch for epoch, _ in all_points)
        transformed_values = [
            math.log10(max(value, 1e-12)) if logarithmic else value for _, value in all_points
        ]
        y_min = min(transformed_values)
        y_max = max(transformed_values)
        if math.isclose(y_min, y_max):
            padding = max(abs(y_min) * 0.1, 0.1)
            y_min -= padding
            y_max += padding
        else:
            padding = (y_max - y_min) * 0.08
            y_min -= padding
            y_max += padding
        for tick in range(5):
            fraction = tick / 4
            y = plot_top + plot_height - fraction * plot_height
            raw_value = y_min + fraction * (y_max - y_min)
            y_label = 10**raw_value if logarithmic else raw_value
            parts.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
                f'y2="{y:.2f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-size="10">{y_label:.3g}</text>'
            )
        parts.append(
            f'<line x1="{left}" y1="{plot_top + plot_height}" '
            f'x2="{left + plot_width}" y2="{plot_top + plot_height}" class="axis"/>'
        )
        for tick in range(5):
            fraction = tick / 4
            x = left + fraction * plot_width
            epoch = round(x_min + fraction * (x_max - x_min))
            parts.append(
                f'<text x="{x:.2f}" y="{plot_top + plot_height + 17:.2f}" '
                f'text-anchor="middle" font-size="10">{epoch}</text>'
            )
        for curve_index, (label, curve) in enumerate(curves):
            if not curve:
                continue
            color = _COLORS[curve_index % len(_COLORS)]
            path = _path(
                curve,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                left=left,
                top=plot_top,
                width=plot_width,
                height=plot_height,
                logarithmic=logarithmic,
            )
            parts.append(f'<path d="{path}" class="curve" stroke="{color}"/>')
            for epoch, value in curve:
                transformed = math.log10(max(value, 1e-12)) if logarithmic else value
                x = left + (epoch - x_min) / max(x_max - x_min, 1) * plot_width
                y = plot_top + plot_height - (transformed - y_min) / (y_max - y_min) * plot_height
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" class="dot" '
                    f'fill="{color}"><title>epoch {epoch}: {value:.6g}</title></circle>'
                )
            legend_y = plot_top + 12 + curve_index * 17
            parts.append(
                f'<line x1="{left + plot_width + 18}" y1="{legend_y - 4}" '
                f'x2="{left + plot_width + 40}" y2="{legend_y - 4}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{left + plot_width + 47}" y="{legend_y}" '
                f'class="legend">{escape(label)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def run_history_svg(rows: Sequence[Mapping[str, Any]], run_id: str) -> str:
    """Render the main training and representation metrics for one run."""
    panels = [
        (
            "Objective MSE (log scale)",
            True,
            [(split, _points(rows, "mse", split)) for split in ("train", "validation", "test")],
        ),
        (
            "Context normalized effective rank",
            False,
            [
                (split, _points(rows, "context_effective_rank_normalized", split))
                for split in ("train", "validation", "test")
            ],
        ),
        (
            "Target normalized effective rank",
            False,
            [
                (split, _points(rows, "target_effective_rank_normalized", split))
                for split in ("train", "validation", "test")
            ],
        ),
        (
            "Latent standard deviation",
            False,
            [
                (f"{split} context", _points(rows, "context_latent_std_mean", split))
                for split in ("train", "validation")
            ]
            + [
                (f"{split} target", _points(rows, "target_latent_std_mean", split))
                for split in ("train", "validation")
            ],
        ),
        (
            "Training gradient norms (log scale)",
            True,
            [
                ("context encoder", _points(rows, "context_encoder_grad_norm", "train")),
                ("predictor", _points(rows, "predictor_grad_norm", "train")),
                ("target encoder", _points(rows, "target_encoder_grad_norm", "train")),
            ],
        ),
    ]
    return _render(f"JEPA learning curves — {run_id}", panels)


def sweep_learning_curves_svg(runs: Sequence[Mapping[str, Any]]) -> str:
    """Render validation-loss curves grouped into one panel per objective."""
    objectives = sorted({str(run["objective"]) for run in runs})
    panels: list[tuple[str, bool, list[tuple[str, list[tuple[int, float]]]]]] = []
    for objective in objectives:
        curves = [
            (
                str(run["variant"]),
                _points(run["history"], "mse", "validation"),
            )
            for run in runs
            if run["objective"] == objective
        ]
        panels.append((f"Validation MSE — {objective} (log scale)", True, curves))
    return _render("JEPA sweep learning curves", panels)
