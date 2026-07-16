"""Failure-isolated experiment sweeps and dependency-free SVG reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from jepa.artifacts import create_timestamped_directory
from jepa.config import (
    BinanceDataConfig,
    DatasetConfig,
    ExperimentConfig,
    ObjectiveKind,
    SystemKind,
    apply_paired_replicate,
    config_identity_hash,
)
from jepa.data import DatasetBundle, build_dataset_bundle, resolved_data_config
from jepa.training import SCHEMA_VERSION, ProgressCallback, build_run_id, train_experiment
from jepa.visualization import load_history, run_history_svg, sweep_learning_curves_svg

SUMMARY_COLUMNS = (
    "replicate_seed",
    "objective",
    "dataset_fingerprint",
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
    "forecast_r2",
    "wall_clock_seconds",
    "error_type",
    "error_message",
)
AGGREGATE_COLUMNS = (
    "objective",
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
    "forecast_r2_mean",
    "wall_clock_seconds_mean",
)

SystemSelection = Literal["linear", "nonlinear", "all"]
ObjectiveSelection = Literal["configured", "future_window", "masked_patches", "all"]


@dataclass(frozen=True, slots=True)
class SweepResult:
    sweep_dir: Path
    rows: tuple[dict[str, Any], ...]
    aggregates: tuple[dict[str, Any], ...]
    completed_count: int
    failed_count: int


def variant_label(
    dynamics: str,
    architecture: str,
    stop_gradient: bool,
    ema_enabled: bool,
    objective: str | None = None,
) -> str:
    prefix = "" if objective is None else f"{objective} | "
    return (
        f"{prefix}{dynamics} | {architecture} | "
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


def _sweep_id(
    config: ExperimentConfig,
    seeds: tuple[int, ...],
    selection: str,
    objectives: ObjectiveSelection,
) -> str:
    identity = [config_identity_hash(config), list(seeds), selection]
    effective_objectives = config.objective.kind if objectives == "configured" else objectives
    legacy = (
        objectives == "configured"
        and config.objective.kind == "future_window"
        and not isinstance(config.data, BinanceDataConfig)
    )
    if not legacy:
        identity.append(effective_objectives)
    payload = json.dumps(identity, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    seed_label = "-".join(str(seed) for seed in seeds)
    objective_label = "" if legacy else f"-{effective_objectives}"
    return f"sweep-{selection}{objective_label}-seeds-{seed_label}-cfg-{digest}"


def _reusable_sweep_directory(root: Path, sweep_key: str) -> Path | None:
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        status_path = candidate / "status.json"
        if not candidate.is_dir() or not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("sweep_key") == sweep_key:
            return candidate
    return None


def _cell_configs(
    config: ExperimentConfig,
    *,
    seeds: tuple[int, ...],
    system_kind: SystemSelection,
    objectives: ObjectiveSelection,
    runs_root: Path,
) -> Iterable[tuple[int, ExperimentConfig]]:
    dynamics_values: tuple[SystemKind | None, ...]
    if isinstance(config.data, BinanceDataConfig):
        dynamics_values = (None,)
    elif system_kind == "all":
        dynamics_values = ("linear", "nonlinear")
    else:
        dynamics_values = (system_kind,)
    objective_values: tuple[ObjectiveKind, ...]
    if objectives == "all":
        objective_values = ("future_window", "masked_patches")
    elif objectives == "configured":
        objective_values = (config.objective.kind,)
    else:
        objective_values = (objectives,)
    for replicate_seed in seeds:
        paired, _ = apply_paired_replicate(config, replicate_seed)
        for dynamics in dynamics_values:
            data: DatasetConfig
            if isinstance(paired.data, BinanceDataConfig):
                data = paired.data
            else:
                if dynamics is None:
                    raise AssertionError("synthetic sweep requires a dynamics value")
                data = replace(paired.data, system_kind=dynamics)
            for objective in objective_values:
                for architecture in ("linear", "nonlinear"):
                    for stop_gradient in (False, True):
                        for ema_enabled in (False, True):
                            yield (
                                replicate_seed,
                                replace(
                                    paired,
                                    data=data,
                                    objective=replace(paired.objective, kind=objective),
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
    dynamics = (
        config.data.system_kind
        if not isinstance(config.data, BinanceDataConfig)
        else "binance_spot"
    )
    return {
        "replicate_seed": replicate_seed,
        "objective": config.objective.kind,
        "dataset_fingerprint": metrics.get("dataset_fingerprint"),
        "dynamics": dynamics,
        "architecture": config.model.architecture,
        "stop_gradient": config.training.stop_gradient,
        "ema_enabled": config.training.ema.enabled,
        "variant": variant_label(
            dynamics,
            config.model.architecture,
            config.training.stop_gradient,
            config.training.ema.enabled,
            config.objective.kind,
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
        "forecast_r2": _metric_value(metrics, ("probes", "context_to_target_window", "test")),
        "wall_clock_seconds": metrics.get("wall_clock_seconds"),
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
    dynamics = (
        config.data.system_kind
        if not isinstance(config.data, BinanceDataConfig)
        else "binance_spot"
    )
    return {
        "replicate_seed": replicate_seed,
        "objective": config.objective.kind,
        "dataset_fingerprint": (
            config.data.fingerprint if isinstance(config.data, BinanceDataConfig) else None
        ),
        "dynamics": dynamics,
        "architecture": config.model.architecture,
        "stop_gradient": config.training.stop_gradient,
        "ema_enabled": config.training.ema.enabled,
        "variant": variant_label(
            dynamics,
            config.model.architecture,
            config.training.stop_gradient,
            config.training.ema.enabled,
            config.objective.kind,
        ),
        "state": "failed",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "validation_mse": None,
        "test_mse": None,
        "context_effective_rank_normalized": None,
        "target_effective_rank_normalized": None,
        "forecast_r2": None,
        "wall_clock_seconds": None,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _ensure_history_plot(run_dir: Path, run_id: str) -> list[dict[str, Any]]:
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        return []
    history = load_history(history_path)
    _atomic_bytes(
        run_dir / "history.svg",
        run_history_svg(history, run_id).encode("utf-8"),
    )
    return history


def aggregate_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row["objective"],
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
        "forecast_r2",
        "wall_clock_seconds",
    )
    for key, group in grouped.items():
        completed = [row for row in group if row["state"] == "complete"]
        aggregate = {
            "objective": key[0],
            "dynamics": key[1],
            "architecture": key[2],
            "stop_gradient": key[3],
            "ema_enabled": key[4],
            "variant": key[5],
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
    objectives = sorted({str(row["objective"]) for row in aggregates}) or ["unknown"]
    metric_panels = (
        ("validation_mse_mean", "Validation MSE"),
        ("context_effective_rank_normalized_mean", "Context normalized effective rank"),
        ("target_effective_rank_normalized_mean", "Target normalized effective rank"),
    )
    panels = tuple(
        (objective, metric, f"{title} — {objective}")
        for objective in objectives
        for metric, title in metric_panels
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
    failures = {row["variant"] for row in rows if row["state"] == "failed"}
    for panel_index, (objective, metric, title) in enumerate(panels):
        top = 70 + panel_index * panel_height
        bottom = top + 220
        panel_aggregates = [row for row in aggregates if row["objective"] == objective]
        count = max(len(panel_aggregates), 1)
        x_step = plot_width / count
        values = [float(row[metric]) for row in panel_aggregates if row[metric] is not None]
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
        for index, aggregate in enumerate(panel_aggregates):
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
    objectives: ObjectiveSelection = "configured",
    progress: ProgressCallback | None = None,
) -> SweepResult:
    """Run every requested cell, isolating failures and always producing summaries."""
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("sweep requires at least one replicate seed")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seed_values):
        raise ValueError("sweep seeds must be integers")
    if objectives not in {"configured", "future_window", "masked_patches", "all"}:
        raise ValueError(f"invalid objective selection: {objectives!r}")
    shared_bundle: DatasetBundle | None = None
    if isinstance(config.data, BinanceDataConfig):
        if system_kind not in {None}:
            raise ValueError("--system-kind is only available for synthetic data")
        shared_bundle = build_dataset_bundle(config.data)
        config = replace(config, data=resolved_data_config(config.data, shared_bundle))
        selection: str = "binance_spot"
        cell_system_kind: SystemSelection = "linear"
    else:
        selection = system_kind or config.data.system_kind
        cell_system_kind = selection
    if selection not in {"linear", "nonlinear", "all", "binance_spot"}:
        raise ValueError(f"invalid system selection: {selection!r}")
    output_root = Path(config.output.root).expanduser().resolve()
    sweep_key = _sweep_id(config, seed_values, selection, objectives)
    sweep_dir = (
        None if config.output.overwrite else _reusable_sweep_directory(output_root, sweep_key)
    )
    if sweep_dir is None:
        sweep_dir = create_timestamped_directory(output_root)
    runs_root = sweep_dir / "runs"
    runs_root.mkdir(exist_ok=True)
    existing_status_path = sweep_dir / "status.json"
    existing_status = (
        json.loads(existing_status_path.read_text()) if existing_status_path.is_file() else {}
    )
    _atomic_json(
        existing_status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "state": "running",
            "sweep_key": sweep_key,
            "created_at_moscow": existing_status.get("created_at_moscow", sweep_dir.name),
            "completed_count": 0,
            "failed_count": 0,
            "cell_count": 0,
        },
    )
    rows: list[dict[str, Any]] = []
    bundle_cache: dict[Any, DatasetBundle] = {}
    cells = tuple(
        _cell_configs(
            config,
            seeds=seed_values,
            system_kind=cell_system_kind,
            objectives=objectives,
            runs_root=runs_root,
        )
    )

    def scoped_progress(metadata: Mapping[str, Any]) -> ProgressCallback:
        def emit(event: dict[str, Any]) -> None:
            if progress is not None:
                progress({**event, **metadata})

        return emit

    for cell_index, (replicate_seed, cell_config) in enumerate(cells, start=1):
        cell_metadata = {
            "cell_index": cell_index,
            "cell_count": len(cells),
            "objective": cell_config.objective.kind,
            "architecture": cell_config.model.architecture,
            "stop_gradient": cell_config.training.stop_gradient,
            "ema_enabled": cell_config.training.ema.enabled,
        }
        if progress is not None:
            progress({"event": "cell_start", **cell_metadata})
        cell_progress = scoped_progress(cell_metadata)

        bundle = shared_bundle
        if bundle is None:
            bundle = bundle_cache.get(cell_config.data)
            if bundle is None:
                bundle = build_dataset_bundle(cell_config.data)
                bundle_cache[cell_config.data] = bundle
        run_id = build_run_id(cell_config, replicate_seed)
        run_dir = runs_root / run_id
        try:
            completed_status = run_dir / "status.json"
            completed_metrics_path = run_dir / "metrics.json"
            if (
                completed_status.is_file()
                and completed_metrics_path.is_file()
                and json.loads(completed_status.read_text()).get("state") == "complete"
            ):
                metrics = json.loads(completed_metrics_path.read_text())
                rows.append(_completed_row(replicate_seed, cell_config, run_id, run_dir, metrics))
                if progress is not None:
                    progress(
                        {
                            "event": "cell_reused",
                            **cell_metadata,
                            "run_id": run_id,
                            "wall_clock_seconds": metrics.get("wall_clock_seconds"),
                        }
                    )
                continue
            cell_started = time.monotonic()
            retry_config = cell_config
            if run_dir.exists() and any(run_dir.iterdir()):
                retry_config = replace(
                    cell_config,
                    output=replace(cell_config.output, overwrite=True),
                )
            result = train_experiment(
                retry_config,
                seed_label=replicate_seed,
                datasets=bundle,
                progress=cell_progress,
                run_dir_override=run_dir,
            )
            completed_metrics = dict(result.metrics)
            completed_metrics["wall_clock_seconds"] = time.monotonic() - cell_started
            rows.append(
                _completed_row(
                    replicate_seed,
                    cell_config,
                    result.run_id,
                    result.run_dir,
                    completed_metrics,
                )
            )
            if progress is not None:
                progress(
                    {
                        "event": "cell_complete",
                        **cell_metadata,
                        "run_id": result.run_id,
                        "wall_clock_seconds": completed_metrics["wall_clock_seconds"],
                    }
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
                        "objective": cell_config.objective.kind,
                        "dataset_fingerprint": bundle.fingerprint,
                        "replicate_seed": replicate_seed,
                        "error": {"type": type(error).__name__, "message": str(error)},
                    },
                )
            rows.append(_failed_row(replicate_seed, cell_config, run_id, run_dir, error))
            if progress is not None:
                progress(
                    {
                        "event": "cell_failed",
                        **cell_metadata,
                        "run_id": run_id,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
    aggregates = aggregate_rows(rows)
    curve_runs: list[dict[str, Any]] = []
    for row in rows:
        if row["state"] != "complete":
            continue
        history = _ensure_history_plot(Path(row["run_dir"]), str(row["run_id"]))
        if history:
            curve_runs.append(
                {
                    "objective": row["objective"],
                    "variant": row["variant"],
                    "history": history,
                }
            )
    _atomic_csv(sweep_dir / "summary.csv", SUMMARY_COLUMNS, rows)
    _atomic_csv(sweep_dir / "summary_by_variant.csv", AGGREGATE_COLUMNS, aggregates)
    _atomic_bytes(sweep_dir / "summary.svg", _svg_report(rows, aggregates).encode())
    _atomic_bytes(
        sweep_dir / "learning_curves.svg",
        sweep_learning_curves_svg(curve_runs).encode("utf-8"),
    )
    completed_count = sum(row["state"] == "complete" for row in rows)
    failed_count = len(rows) - completed_count
    _atomic_json(
        sweep_dir / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "complete" if failed_count == 0 else "partial_failure",
            "sweep_key": sweep_key,
            "created_at_moscow": existing_status.get("created_at_moscow", sweep_dir.name),
            "completed_count": completed_count,
            "failed_count": failed_count,
            "cell_count": len(rows),
            "objectives": sorted({row["objective"] for row in rows}),
            "dataset_fingerprints": sorted(
                {row["dataset_fingerprint"] for row in rows if row["dataset_fingerprint"]}
            ),
        },
    )
    return SweepResult(
        sweep_dir,
        tuple(rows),
        tuple(aggregates),
        completed_count,
        failed_count,
    )
