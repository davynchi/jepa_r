from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts/images/run_spatial_quality_grid.py"
    spec = importlib.util.spec_from_file_location("spatial_quality_grid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_is_deterministic_and_has_exact_crossing(tmp_path) -> None:
    grid = _module()
    config = Path(__file__).parents[1] / "configs/images/shapes3d/quality.yaml"
    first = grid.plan(config, tmp_path / "first")
    second = grid.plan(config, tmp_path / "second")
    assert len(first["cells"]) == 66
    assert [cell["seed_block"] for cell in first["cells"]] == [
        cell["seed_block"] for cell in second["cells"]
    ]
    for phase, count in (("pilot", 6), ("discovery", 30), ("replication", 30)):
        assert sum(cell["phase"] == phase for cell in first["cells"]) == count
    discovery = {cell["seed_block"] for cell in first["cells"] if cell["phase"] == "discovery"}
    replication = {cell["seed_block"] for cell in first["cells"] if cell["phase"] == "replication"}
    assert discovery.isdisjoint(replication)
    assert all(
        cell["command"][cell["command"].index("--device") + 1] == "mps" for cell in first["cells"]
    )
    assert not (tmp_path / "first/runs").exists()


def test_run_local_and_resume_preserve_completed_cells(tmp_path, monkeypatch) -> None:
    grid = _module()
    config = Path(__file__).parents[1] / "configs/images/shapes3d/quality.yaml"
    root = tmp_path / "grid"
    manifest = grid.plan(config, root)
    for cell in manifest["cells"][2:]:
        cell["state"] = "complete"
    grid._atomic_json(root / "grid_manifest.json", manifest)
    outcomes = iter(((0, "ok"), (1, "failed"), (0, "retried")))
    monkeypatch.setattr(grid, "_run_cell", lambda cell: next(outcomes))
    first = grid.run_local(root, max_concurrency=1, retry_failed=False)
    assert [cell["state"] for cell in first["cells"][:2]] == ["complete", "failed"]
    assert first["cells"][1]["retry_eligible"] is True
    failed_run = Path(first["cells"][1]["run_dir"])
    failed_run.mkdir(parents=True)
    (failed_run / "partial.log").write_text("first attempt")
    second = grid.run_local(root, max_concurrency=1, retry_failed=True)
    assert second["cells"][0]["attempts"] == 1
    assert second["cells"][1]["attempts"] == 2
    archive = Path(second["cells"][1]["archived_attempts"][0])
    assert archive.name.endswith(".failed-attempt-01")
    assert (archive / "partial.log").read_text() == "first attempt"
    assert all(cell["state"] == "complete" for cell in second["cells"])


def test_run_local_starts_pilot_first_and_only_marks_dispatched_cells(
    tmp_path, monkeypatch
) -> None:
    grid = _module()
    config = Path(__file__).parents[1] / "configs/images/shapes3d/quality.yaml"
    root = tmp_path / "grid"
    grid.plan(config, root)
    observed = []

    def fail_first(cell):
        manifest = json.loads((root / "grid_manifest.json").read_text())
        observed.append(
            (
                cell["phase"],
                sum(item["state"] == "running" for item in manifest["cells"]),
            )
        )
        raise RuntimeError("stop after observing dispatch state")

    monkeypatch.setattr(grid, "_run_cell", fail_first)
    with pytest.raises(RuntimeError, match="observing dispatch"):
        grid.run_local(root, max_concurrency=1, retry_failed=False)
    assert observed == [("pilot", 1)]


def test_reconcile_validates_identity_and_completed_immutability(tmp_path) -> None:
    grid = _module()
    config = Path(__file__).parents[1] / "configs/images/shapes3d/quality.yaml"
    root = tmp_path / "grid"
    manifest = grid.plan(config, root)
    cell = manifest["cells"][0]
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            {
                "cell_id": cell["cell_id"],
                "command_hash": cell["command_hash"],
                "state": "complete",
                "resource_usage": {"gpu_hours": 1.0},
            }
        )
    )
    reconciled = grid.reconcile(root, [status])
    assert reconciled["cells"][0]["state"] == "complete"
    status.write_text(
        json.dumps(
            {
                "cell_id": cell["cell_id"],
                "command_hash": cell["command_hash"],
                "state": "failed",
            }
        )
    )
    with pytest.raises(ValueError, match="immutable"):
        grid.reconcile(root, [status])
