"""Failure-isolated experiment sweeps and dependency-free SVG reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from jepa.configs.base import (
    ExperimentConfig,
    SystemKind,
    apply_paired_replicate,
    config_identity_hash,
)
from jepa.training.core import SCHEMA_VERSION, build_run_id, train_experiment

SUMMARY_COLUMNS = (
    "replicate_seed",
    "dynamics",
    "architecture",
    "stop_gradient",
    "ema_enabled",
    "variant",
    "state",
    "run_id",
    "run_dir",
    "validation_mse",
    "test_mse",
    "context_effective_rank_normalized",
    "target_effective_rank_normalized",
    "error_type",
    "error_message",
)
AGGREGATE_COLUMNS = (
    "dynamics",
    "architecture",
    "stop_gradient",
    "ema_enabled",
    "variant",
    "completed_count",
    "failed_count",
    "validation_mse_mean",
    "test_mse_mean",
    "context_effective_rank_normalized_mean",
    "target_effective_rank_normalized_mean",
)

SystemSelection = Literal["linear", "nonlinear", "all"]


@dataclass(frozen=True, slots=True)
class SweepResult:
    sweep_dir: Path
    rows: tuple[dict[str, Any], ...]
    aggregates: tuple[dict[str, Any], ...]
    completed_count: int
    failed_count: int


def variant_label(dynamics: str, architecture: str, stop_gradient: bool, ema_enabled: bool) -> str:
    return (
        f"{dynamics} | {architecture} | "
        f"SG {'on' if stop_gradient else 'off'} | EMA {'on' if ema_enabled else 'off'}"
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path, content.encode())


def _atomic_csv(path: Path, columns: tuple[str, ...], rows: Iterable[Mapping[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column) for column in columns})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sweep_id(config: ExperimentConfig, seeds: tuple[int, ...], selection: str) -> str:
    payload = json.dumps(
        [config_identity_hash(config), list(seeds), selection], separators=(",", ":")
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    seed_label = "-".join(str(seed) for seed in seeds)
    return f"sweep-{selection}-seeds-{seed_label}-cfg-{digest}"


def _cell_configs(
    config: ExperimentConfig,
    *,
    seeds: tuple[int, ...],
    system_kind: SystemSelection,
    runs_root: Path,
) -> Iterable[tuple[int, ExperimentConfig]]:
    dynamics_values: tuple[SystemKind, ...]
    if system_kind == "all":
        dynamics_values = ("linear", "nonlinear")
    else:
        dynamics_values = (system_kind,)
    for replicate_seed in seeds:
        paired, _ = apply_paired_replicate(config, replicate_seed)
        for dynamics in dynamics_values:
            for architecture in ("linear", "nonlinear"):
                for stop_gradient in (False, True):
                    for ema_enabled in (False, True):
                        yield (
                            replicate_seed,
                            replace(
                                paired,
                                data=replace(paired.data, system_kind=dynamics),
                                model=replace(paired.model, architecture=architecture),
                                training=replace(
                                    paired.training,
                                    stop_gradient=stop_gradient,
                                    ema=replace(paired.training.ema, enabled=ema_enabled),
                                ),
                                output=replace(
                                    paired.output,
                                    root=str(runs_root),
                                    overwrite=config.output.overwrite,
                                ),
                            ),
                        )


def _metric_value(metrics: Mapping[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = metrics
    for key in path:
        value = value[key]
    if isinstance(value, Mapping):
        value = value.get("value")
    return None if value is None else float(value)


def _completed_row(
    replicate_seed: int,
    config: ExperimentConfig,
    run_id: str,
    run_dir: Path,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "replicate_seed": replicate_seed,
        "dynamics": config.data.system_kind,
        "architecture": config.model.architecture,
        "stop_gradient": config.training.stop_gradient,
        "ema_enabled": config.training.ema.enabled,
        "variant": variant_label(
            config.data.system_kind,
            config.model.architecture,
            config.training.stop_gradient,
            config.training.ema.enabled,
        ),
        "state": "complete",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "validation_mse": _metric_value(metrics, ("objective", "validation")),
        "test_mse": _metric_value(metrics, ("objective", "test")),
        "context_effective_rank_normalized": _metric_value(
            metrics, ("representations", "validation", "context", "effective_rank_normalized")
        ),
        "target_effective_rank_normalized": _metric_value(
            metrics, ("representations", "validation", "target", "effective_rank_normalized")
        ),
        "error_type": None,
        "error_message": None,
    }


def _failed_row(
    replicate_seed: int,
    config: ExperimentConfig,
    run_id: str,
    run_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    return {
        "replicate_seed": replicate_seed,
        "dynamics": config.data.system_kind,
        "architecture": config.model.architecture,
        "stop_gradient": config.training.stop_gradient,
        "ema_enabled": config.training.ema.enabled,
        "variant": variant_label(
            config.data.system_kind,
            config.model.architecture,
            config.training.stop_gradient,
            config.training.ema.enabled,
        ),
        "state": "failed",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "validation_mse": None,
        "test_mse": None,
        "context_effective_rank_normalized": None,
        "target_effective_rank_normalized": None,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def aggregate_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row["dynamics"],
            row["architecture"],
            row["stop_gradient"],
            row["ema_enabled"],
            row["variant"],
        )
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    metric_names = (
        "validation_mse",
        "test_mse",
        "context_effective_rank_normalized",
        "target_effective_rank_normalized",
    )
    for key, group in grouped.items():
        completed = [row for row in group if row["state"] == "complete"]
        aggregate = {
            "dynamics": key[0],
            "architecture": key[1],
            "stop_gradient": key[2],
            "ema_enabled": key[3],
            "variant": key[4],
            "completed_count": len(completed),
            "failed_count": len(group) - len(completed),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in completed if row[metric_name] is not None]
            aggregate[f"{metric_name}_mean"] = fmean(values) if values else None
        aggregates.append(aggregate)
    return aggregates


def _svg_report(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> str:
    width = 1440
    panel_height = 310
    margin_left = 80
    plot_width = width - margin_left - 30
    panels = (
        ("validation_mse_mean", "Validation MSE"),
        ("context_effective_rank_normalized_mean", "Context normalized effective rank"),
        ("target_effective_rank_normalized_mean", "Target normalized effective rank"),
    )
    height = 70 + panel_height * len(panels) + 130
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#172033}"
        ".title{font-size:24px;font-weight:700}.panel{font-size:16px;font-weight:650}"
        ".axis{stroke:#9aa4b2;stroke-width:1}.grid{stroke:#e5e9f0;stroke-width:1}"
        ".point{fill:#3b6eea}.failed{fill:#cf334f;font-weight:700}.label{font-size:9px}</style>",
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        '<text x="30" y="38" class="title">JEPA sweep summary</text>',
    ]
    count = max(len(aggregates), 1)
    x_step = plot_width / count
    failures = {row["variant"] for row in rows if row["state"] == "failed"}
    for panel_index, (metric, title) in enumerate(panels):
        top = 70 + panel_index * panel_height
        bottom = top + 220
        values = [float(row[metric]) for row in aggregates if row[metric] is not None]
        maximum = max(values) if values else 1.0
        minimum = min(values) if values else 0.0
        if math.isclose(maximum, minimum):
            padding = max(abs(maximum) * 0.1, 0.1)
            minimum -= padding
            maximum += padding
        parts.append(f'<text x="30" y="{top + 18}" class="panel">{escape(title)}</text>')
        for tick in range(5):
            fraction = tick / 4
            y = bottom - fraction * 180
            value = minimum + fraction * (maximum - minimum)
            parts.append(
                f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - 30}" '
                f'y2="{y:.2f}" class="grid"/>'
            )
            parts.append(
                f'<text x="72" y="{y + 4:.2f}" text-anchor="end" font-size="10">{value:.4g}</text>'
            )
        parts.append(
            f'<line x1="{margin_left}" y1="{bottom}" x2="{width - 30}" y2="{bottom}" class="axis"/>'
        )
        for index, aggregate in enumerate(aggregates):
            x = margin_left + (index + 0.5) * x_step
            value = aggregate[metric]
            if value is not None:
                y = bottom - (float(value) - minimum) / (maximum - minimum) * 180
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" class="point">'
                    f"<title>{escape(aggregate['variant'])}: "
                    f"{float(value):.6g}</title></circle>"
                )
            if aggregate["variant"] in failures:
                parts.append(
                    f'<text x="{x:.2f}" y="{bottom - 188}" text-anchor="middle" '
                    'class="failed">× FAILED</text>'
                )
            if panel_index == len(panels) - 1:
                label = escape(aggregate["variant"].replace(" | ", " / "))
                parts.append(
                    f'<text x="{x:.2f}" y="{bottom + 16}" text-anchor="end" '
                    f'transform="rotate(-35 {x:.2f} {bottom + 16})" '
                    f'class="label">{label}</text>'
                )
    completed = sum(row["state"] == "complete" for row in rows)
    failed = len(rows) - completed
    parts.append(
        f'<text x="30" y="{height - 24}" font-size="13">'
        f"Completed: {completed} · Failed: {failed}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def run_sweep(
    config: ExperimentConfig,
    *,
    seeds: Iterable[int],
    system_kind: SystemSelection | None = None,
) -> SweepResult:
    """Run every requested cell, isolating failures and always producing summaries."""
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("sweep requires at least one replicate seed")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seed_values):
        raise ValueError("sweep seeds must be integers")
    selection: SystemSelection = system_kind or config.data.system_kind
    if selection not in {"linear", "nonlinear", "all"}:
        raise ValueError(f"invalid system selection: {selection!r}")
    sweep_dir = Path(config.output.root).expanduser().resolve() / _sweep_id(
        config, seed_values, selection
    )
    if sweep_dir.exists() and any(sweep_dir.iterdir()):
        if not config.output.overwrite:
            raise FileExistsError(f"sweep directory already exists: {sweep_dir}")
        shutil.rmtree(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    runs_root = sweep_dir / "runs"
    runs_root.mkdir()
    rows: list[dict[str, Any]] = []
    for replicate_seed, cell_config in _cell_configs(
        config, seeds=seed_values, system_kind=selection, runs_root=runs_root
    ):
        run_id = build_run_id(cell_config, replicate_seed)
        run_dir = runs_root / run_id
        try:
            result = train_experiment(cell_config, seed_label=replicate_seed)
            rows.append(
                _completed_row(
                    replicate_seed, cell_config, result.run_id, result.run_dir, result.metrics
                )
            )
        except Exception as error:
            run_dir.mkdir(parents=True, exist_ok=True)
            status_path = run_dir / "status.json"
            if not status_path.exists():
                _atomic_json(
                    status_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": run_id,
                        "state": "failed",
                        "completed_epoch": 0,
                        "config_hash": config_identity_hash(cell_config),
                        "error": {"type": type(error).__name__, "message": str(error)},
                    },
                )
            rows.append(_failed_row(replicate_seed, cell_config, run_id, run_dir, error))
    aggregates = aggregate_rows(rows)
    _atomic_csv(sweep_dir / "summary.csv", SUMMARY_COLUMNS, rows)
    _atomic_csv(sweep_dir / "summary_by_variant.csv", AGGREGATE_COLUMNS, aggregates)
    _atomic_bytes(sweep_dir / "summary.svg", _svg_report(rows, aggregates).encode())
    completed_count = sum(row["state"] == "complete" for row in rows)
    failed_count = len(rows) - completed_count
    _atomic_json(
        sweep_dir / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "complete" if failed_count == 0 else "partial_failure",
            "completed_count": completed_count,
            "failed_count": failed_count,
            "cell_count": len(rows),
        },
    )
    return SweepResult(
        sweep_dir,
        tuple(rows),
        tuple(aggregates),
        completed_count,
        failed_count,
    )
