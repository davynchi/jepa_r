#!/usr/bin/env python3
"""Plan, run, resume, and reconcile the Shapes3D quality experiment grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import yaml  # noqa: E402

from jepa.configs.base import derive_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/images/shapes3d/quality.yaml"
TRAIN_SCRIPT = "scripts/images/train_ijepa_spatial.py"
CURRICULA: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("uniform", ("--weighting-method", "uniform"), False),
    ("loss", ("--weighting-method", "loss"), False),
    (
        "ras-logdet",
        ("--weighting-method", "ras", "--weighting-richness", "logdet"),
        False,
    ),
    (
        "coord-covariance",
        ("--weighting-method", "coord", "--coordinate-importance", "covariance"),
        False,
    ),
    (
        "coord-transformation",
        ("--weighting-method", "coord", "--coordinate-importance", "transformation"),
        True,
    ),
    (
        "coord-dynamics",
        ("--weighting-method", "coord", "--coordinate-importance", "dynamics"),
        False,
    ),
)
TERMINAL_STATES = {"complete", "failed"}
PHASE_ORDER = ("pilot", "discovery", "replication")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("quality grid config must be a mapping")
    return payload


def _phase_cells(
    phase: str, phase_config: Mapping[str, Any], common: Mapping[str, Any], grid_root: Path
) -> list[dict[str, Any]]:
    replicates = int(phase_config["replicates"])
    epochs = int(phase_config["epochs"])
    namespace = str(phase_config["seed_namespace"])
    cells: list[dict[str, Any]] = []
    for replicate in range(replicates):
        seed = derive_seed(0, namespace, replicate)
        for curriculum, flags, uses_metadata in CURRICULA:
            cell_id = f"{phase}-block-{replicate:02d}-{curriculum}"
            run_name = f"{phase}_seed{seed}_{curriculum}"
            command = [
                "uv",
                "run",
                "python",
                TRAIN_SCRIPT,
                "--dataset",
                "shapes3d",
                "--run-name",
                run_name,
                "--output-root",
                str(grid_root / "runs"),
                "--seed",
                str(seed),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(int(common["batch_size"])),
                "--device",
                str(common["device"]),
                "--num-train-samples",
                str(int(common["num_train_samples"])),
                "--num-val-samples",
                str(int(common["num_val_samples"])),
                "--num-test-samples",
                str(int(common["num_test_samples"])),
                "--checkpoint-every-epochs",
                str(int(common["checkpoint_every_epochs"])),
                *flags,
            ]
            cells.append(
                {
                    "cell_id": cell_id,
                    "phase": phase,
                    "seed_block": seed,
                    "replicate_index": replicate,
                    "curriculum": curriculum,
                    "training_uses_shape_metadata": uses_metadata,
                    "command": command,
                    "command_hash": _canonical_hash(command),
                    "run_dir": str(grid_root / "runs" / run_name),
                    "state": "pending",
                    "retry_eligible": False,
                    "attempts": 0,
                    "resource_usage": {},
                }
            )
    return cells


def _validate_crossing(cells: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    phases = config["phases"]
    if set(phases) != set(PHASE_ORDER):
        raise ValueError(f"phases must be exactly {PHASE_ORDER}")
    seed_sets: dict[str, set[int]] = {}
    for phase, phase_config in phases.items():
        phase_cells = [cell for cell in cells if cell["phase"] == phase]
        expected = int(phase_config["replicates"]) * len(CURRICULA)
        if len(phase_cells) != expected:
            raise ValueError(f"{phase} grid is incomplete")
        grouped: dict[int, set[str]] = {}
        for cell in phase_cells:
            grouped.setdefault(int(cell["seed_block"]), set()).add(str(cell["curriculum"]))
        if len(grouped) != int(phase_config["replicates"]) or any(
            values != {item[0] for item in CURRICULA} for values in grouped.values()
        ):
            raise ValueError(f"{phase} does not have a complete seed-block crossing")
        seed_sets[str(phase)] = set(grouped)
    if seed_sets["discovery"].intersection(seed_sets["replication"]):
        raise ValueError("discovery and replication seeds must be disjoint")


def plan(config_path: Path, grid_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported quality grid config schema")
    common = config["training"]
    cells = [
        cell
        for phase, phase_config in config["phases"].items()
        for cell in _phase_cells(phase, phase_config, common, grid_root)
    ]
    _validate_crossing(cells, config)
    frozen_config = {
        **config,
        "source_config": str(config_path.resolve()),
        "data_config": str((REPO_ROOT / "configs/images/shapes3d/quick.yaml").resolve()),
    }
    config_hash = _canonical_hash(frozen_config)
    manifest = {
        "schema_version": 1,
        "phase_config_hash": config_hash,
        "config": frozen_config,
        "cells": cells,
    }
    grid_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(grid_root / f"phase_config_{config_hash}.json", frozen_config)
    _atomic_json(grid_root / "grid_manifest.json", manifest)
    return manifest


def _load_manifest(grid_root: Path) -> dict[str, Any]:
    payload = json.loads((grid_root / "grid_manifest.json").read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("cells"), list):
        raise ValueError("invalid grid manifest")
    frozen = payload["config"]
    if _canonical_hash(frozen) != payload["phase_config_hash"]:
        raise ValueError("phase config hash mismatch")
    _validate_crossing(payload["cells"], frozen)
    return payload


def _failure_rate_stop(cells: Sequence[Mapping[str, Any]]) -> bool:
    terminal = [cell for cell in cells if cell["state"] in TERMINAL_STATES]
    failed = sum(cell["state"] == "failed" for cell in terminal)
    return len(terminal) >= 10 and failed >= 3 and failed / len(terminal) > 0.2


def _run_cell(cell: Mapping[str, Any]) -> tuple[int, str]:
    result = subprocess.run(
        list(cell["command"]),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout[-20_000:]


def _archive_failed_attempt(cell: dict[str, Any]) -> None:
    run_dir = Path(cell["run_dir"])
    if not run_dir.exists():
        return
    attempt = int(cell["attempts"])
    archive = run_dir.with_name(f"{run_dir.name}.failed-attempt-{attempt:02d}")
    if archive.exists():
        raise FileExistsError(f"failed-attempt archive already exists: {archive}")
    run_dir.replace(archive)
    cell.setdefault("archived_attempts", []).append(str(archive))


def run_local(grid_root: Path, *, max_concurrency: int, retry_failed: bool) -> dict[str, Any]:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    manifest = _load_manifest(grid_root)
    cells: list[dict[str, Any]] = manifest["cells"]
    for cell in cells:
        if cell["state"] == "running":
            cell["state"] = "failed"
            cell["retry_eligible"] = True
            cell["error_type"] = "interrupted"
    eligible = [
        cell
        for cell in cells
        if cell["state"] == "pending"
        or (retry_failed and cell["state"] == "failed" and cell["retry_eligible"])
    ]
    active_phase = next(
        (
            phase
            for phase in PHASE_ORDER
            if any(cell["phase"] == phase and cell["state"] != "complete" for cell in cells)
        ),
        None,
    )
    if active_phase is not None:
        eligible = [cell for cell in eligible if cell["phase"] == active_phase]
    if _failure_rate_stop(cells):
        raise RuntimeError("grid failure-rate stop is active")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: dict[concurrent.futures.Future[tuple[int, str]], dict[str, Any]] = {}
        remaining = iter(eligible)

        def submit_next() -> bool:
            cell = next(remaining, None)
            if cell is None:
                return False
            if cell["state"] == "failed":
                _archive_failed_attempt(cell)
            cell["state"] = "running"
            cell["attempts"] = int(cell["attempts"]) + 1
            cell["retry_eligible"] = False
            _atomic_json(grid_root / "grid_manifest.json", manifest)
            futures[executor.submit(_run_cell, cell)] = cell
            return True

        for _ in range(max_concurrency):
            if not submit_next():
                break
        while futures:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                cell = futures.pop(future)
                returncode, output = future.result()
                cell["state"] = "complete" if returncode == 0 else "failed"
                cell["retry_eligible"] = returncode != 0
                cell["returncode"] = returncode
                cell["output_tail"] = output
                _atomic_json(grid_root / "grid_manifest.json", manifest)
            if _failure_rate_stop(cells):
                break
            while len(futures) < max_concurrency and submit_next():
                pass
    return manifest


def reconcile(grid_root: Path, status_files: Sequence[Path]) -> dict[str, Any]:
    manifest = _load_manifest(grid_root)
    by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    for path in status_files:
        status = json.loads(path.read_text())
        cell = by_id.get(status.get("cell_id"))
        if cell is None:
            raise ValueError(f"unknown reconciled cell: {status.get('cell_id')}")
        if status.get("command_hash") != cell["command_hash"]:
            raise ValueError("reconciled command hash mismatch")
        state = status.get("state")
        if state not in {"running", "complete", "failed"}:
            raise ValueError("invalid reconciled state")
        if cell["state"] == "complete" and state != "complete":
            raise ValueError("completed cells are immutable")
        cell["state"] = state
        cell["retry_eligible"] = bool(status.get("retry_eligible", False))
        cell["resource_usage"] = status.get("resource_usage", {})
        if status.get("run_dir"):
            cell["run_dir"] = status["run_dir"]
    _atomic_json(grid_root / "grid_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan_parser.add_argument("--grid-root", type=Path, required=True)
    for name in ("run-local", "resume"):
        run_parser = subparsers.add_parser(name)
        run_parser.add_argument("--grid-root", type=Path, required=True)
        run_parser.add_argument("--max-concurrency", type=int, default=1)
    reconcile_parser = subparsers.add_parser("reconcile")
    reconcile_parser.add_argument("--grid-root", type=Path, required=True)
    reconcile_parser.add_argument("status_files", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.command == "plan":
        result = plan(arguments.config.resolve(), arguments.grid_root.resolve())
    elif arguments.command in {"run-local", "resume"}:
        result = run_local(
            arguments.grid_root.resolve(),
            max_concurrency=arguments.max_concurrency,
            retry_failed=arguments.command == "resume",
        )
    else:
        result = reconcile(arguments.grid_root.resolve(), arguments.status_files)
    counts = {
        state: sum(cell["state"] == state for cell in result["cells"])
        for state in ("pending", "running", "complete", "failed")
    }
    print(json.dumps(counts, sort_keys=True))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
