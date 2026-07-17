from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


GROUP_NAMES = {
    0: "easy",
    1: "hard",
    2: "noise",
}


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-sample latent prediction discrepancy.

    Supports tensors [B, D] or [B, T, D].
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.ndim < 2:
        raise ValueError("prediction must have at least batch and feature dimensions")

    dims = tuple(range(1, prediction.ndim))
    return (prediction - target).square().mean(dim=dims)


def per_sample_cosine_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    prediction_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    return 1.0 - F.cosine_similarity(prediction_flat, target_flat, dim=1, eps=eps)


def per_sample_relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    dims = tuple(range(1, prediction.ndim))
    numerator = (prediction - target).square().mean(dim=dims)
    denominator = target.square().mean(dim=dims).clamp_min(eps)
    return numerator / denominator


def per_sample_norm_gap(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    prediction_norm = prediction.flatten(1).norm(dim=1)
    target_norm = target.flatten(1).norm(dim=1).clamp_min(eps)
    return (prediction_norm - target_norm).abs() / target_norm


def bandpass_useful_surprise_score(
    current_surprise: torch.Tensor,
    previous_surprise: torch.Tensor,
    lower: float = 0.0,
    upper: float = 2.0,
    temperature: float = 0.5,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    lp_pos = positive_learning_progress(previous_surprise, current_surprise)
    normalized = zscore(current_surprise, eps=eps)
    lower_gate = torch.sigmoid((normalized - lower) / temperature)
    upper_gate = torch.sigmoid((upper - normalized) / temperature)
    return lp_pos * lower_gate * upper_gate


def learnable_sampling_score(
    current_surprise: torch.Tensor,
    previous_surprise: torch.Tensor,
    lower: float = 0.0,
    upper: float = 1.5,
    temperature: float = 0.5,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    lp_pos = positive_learning_progress(previous_surprise, current_surprise)
    normalized = zscore(current_surprise, eps=eps)
    not_easy_gate = torch.sigmoid((normalized - lower) / temperature)
    not_extreme_gate = torch.sigmoid((upper - normalized) / temperature)
    return lp_pos * not_easy_gate * not_extreme_gate



def zscore(values: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize surprise values inside one split/epoch."""
    if values.ndim != 1:
        raise ValueError("values must have shape [num_samples]")
    return (values - values.mean()) / (values.std(unbiased=False) + eps)


def learning_progress(
    previous_surprise: torch.Tensor,
    current_surprise: torch.Tensor,
) -> torch.Tensor:
    """Positive values mean the model became better on this sample."""
    if previous_surprise.shape != current_surprise.shape:
        raise ValueError("previous and current surprise arrays must have same shape")
    return previous_surprise - current_surprise


def positive_learning_progress(
    previous_surprise: torch.Tensor,
    current_surprise: torch.Tensor,
) -> torch.Tensor:
    """Learning progress clipped to useful positive part."""
    return learning_progress(previous_surprise, current_surprise).clamp_min(0.0)


def useful_surprise_score(
    current_surprise: torch.Tensor,
    previous_surprise: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Useful surprise = currently non-trivial + actually learnable.

    High raw surprise alone selects both hard and noise samples.
    This score prefers samples with above-average current surprise and
    positive learning progress.
    """
    normalized = zscore(current_surprise, eps=eps).clamp_min(0.0)
    lp_pos = positive_learning_progress(previous_surprise, current_surprise)
    return lp_pos / (1.0 + current_surprise.clamp_min(0.0) + eps)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _points(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    group: str,
    field: str,
) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    for row in rows:
        if row["split"] != split or row["group"] != group:
            continue
        value = row.get(field)
        if value is None or value == "":
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
) -> str:
    x_span = max(x_max - x_min, 1)
    y_span = max(y_max - y_min, 1e-12)

    coordinates = [
        (
            left + (epoch - x_min) / x_span * width,
            top + height - (value - y_min) / y_span * height,
        )
        for epoch, value in points
    ]

    return " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(coordinates)
    )


def render_surprise_summary_svg(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    title: str,
) -> str:
    """Render group-level surprise curves as dependency-free SVG."""
    colors = {
        "easy": "#2563eb",
        "hard": "#dc2626",
        "noise": "#059669",
    }

    panels = [
        ("Raw MSE surprise", "raw_mse_surprise_mean"),
        ("Positive learning progress", "positive_learning_progress_mean"),
        ("Relative sampling score", "sampling_score_relative_mean"),
    ]

    width = 1280
    panel_height = 260
    height = 90 + panel_height * len(panels)
    left = 90.0
    plot_width = width - left - 250
    plot_height = 165.0

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="32" y="42" font-family="Arial" font-size="24" '
        f'font-weight="700" fill="#0f172a">{escape(title)}</text>',
    ]

    for panel_index, (panel_title, field) in enumerate(panels):
        top = 80.0 + panel_index * panel_height
        plot_top = top + 35.0

        curves = [
            (group, _points(rows, split=split, group=group, field=field))
            for group in ("easy", "hard", "noise")
        ]
        all_points = [point for _, curve in curves for point in curve]

        parts.append(
            f'<text x="32" y="{top + 10:.1f}" font-family="Arial" '
            f'font-size="17" font-weight="700" fill="#0f172a">'
            f'{escape(panel_title)} — {escape(split)}</text>'
        )

        if not all_points:
            parts.append(
                f'<text x="{left}" y="{plot_top + 80}" font-family="Arial" '
                f'font-size="14" fill="#64748b">No data</text>'
            )
            continue

        x_min = min(epoch for epoch, _ in all_points)
        x_max = max(epoch for epoch, _ in all_points)
        y_min = min(value for _, value in all_points)
        y_max = max(value for _, value in all_points)

        if math.isclose(y_min, y_max):
            padding = max(abs(y_min) * 0.1, 0.1)
        else:
            padding = (y_max - y_min) * 0.1
        y_min -= padding
        y_max += padding

        for tick in range(5):
            fraction = tick / 4
            y = plot_top + plot_height - fraction * plot_height
            value = y_min + fraction * (y_max - y_min)
            parts.append(
                f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.2f}" '
                f'y2="{y:.2f}" stroke="#e2e8f0"/>'
            )
            parts.append(
                f'<text x="28" y="{y + 4:.2f}" font-family="Arial" '
                f'font-size="11" fill="#334155">{value:.3g}</text>'
            )

        parts.append(
            f'<line x1="{left}" x2="{left + plot_width}" '
            f'y1="{plot_top + plot_height:.2f}" y2="{plot_top + plot_height:.2f}" '
            f'stroke="#94a3b8"/>'
        )

        for tick in range(5):
            fraction = tick / 4
            x = left + fraction * plot_width
            epoch = round(x_min + fraction * (x_max - x_min))
            parts.append(
                f'<text x="{x:.2f}" y="{plot_top + plot_height + 24:.2f}" '
                f'font-family="Arial" font-size="11" fill="#334155" '
                f'text-anchor="middle">{epoch}</text>'
            )

        for curve_index, (group, curve) in enumerate(curves):
            if not curve:
                continue

            color = colors[group]
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
            )
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" '
                f'stroke-width="2.2"/>'
            )

            for epoch, value in curve:
                x = left + (epoch - x_min) / max(x_max - x_min, 1) * plot_width
                y = plot_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}">'
                    f'<title>epoch {epoch}: {group} = {value:.6g}</title></circle>'
                )

            legend_y = plot_top + 12 + curve_index * 20
            parts.append(
                f'<line x1="{left + plot_width + 30}" x2="{left + plot_width + 55}" '
                f'y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"/>'
            )
            parts.append(
                f'<text x="{left + plot_width + 65}" y="{legend_y + 4}" '
                f'font-family="Arial" font-size="13" fill="#0f172a">'
                f'{escape(group)}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)
