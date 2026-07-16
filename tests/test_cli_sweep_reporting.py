from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

import jepa.reporting as reporting_module
from jepa.cli import main
from jepa.config import (
    DataConfig,
    EMAConfig,
    EvaluationConfig,
    ExperimentConfig,
    MaskedPatchesConfig,
    ModelConfig,
    ObjectiveConfig,
    OutputConfig,
    TrainingConfig,
)
from jepa.reporting import AGGREGATE_COLUMNS, SUMMARY_COLUMNS, SweepResult, run_sweep


def _config(root: Path, *, overwrite: bool = False) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            num_samples=8,
            validation_samples=4,
            test_samples=4,
            sequence_length=6,
            window_size=2,
            burn_in_steps=2,
            latent_state_dim=2,
            observation_dim=2,
        ),
        model=ModelConfig(latent_dim=3, hidden_dim=4, hidden_layers=1),
        training=TrainingConfig(
            batch_size=4,
            epochs=1,
            device="cpu",
            evaluation_every_epochs=1,
            checkpoint_every_epochs=1,
            ema=EMAConfig(decay=0.9),
        ),
        evaluation=EvaluationConfig(probe_ridge=1.0e-4),
        output=OutputConfig(root=str(root), overwrite=overwrite),
    )


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def test_one_dynamics_sweep_produces_eight_cells_and_reports(tmp_path: Path) -> None:
    progress: list[dict[str, Any]] = []
    result = run_sweep(
        _config(tmp_path),
        seeds=[7],
        system_kind="linear",
        progress=progress.append,
    )

    assert result.completed_count == 8
    assert result.failed_count == 0
    assert len(result.rows) == 8
    assert len({row["run_id"] for row in result.rows}) == 8
    assert len(result.aggregates) == 8
    assert sum(event["event"] == "cell_start" for event in progress) == 8
    assert sum(event["event"] == "epoch_complete" for event in progress) == 8
    assert sum(event["event"] == "cell_complete" for event in progress) == 8
    assert {path.name for path in result.sweep_dir.iterdir()} == {
        "runs",
        "status.json",
        "summary.csv",
        "summary_by_variant.csv",
        "summary.svg",
        "learning_curves.svg",
    }
    fields, rows = _read_csv(result.sweep_dir / "summary.csv")
    assert fields == SUMMARY_COLUMNS
    assert len(rows) == 8
    aggregate_fields, aggregates = _read_csv(result.sweep_dir / "summary_by_variant.csv")
    assert aggregate_fields == AGGREGATE_COLUMNS
    assert len(aggregates) == 8
    assert all(row["completed_count"] == "1" for row in aggregates)
    ET.parse(result.sweep_dir / "summary.svg")
    ET.parse(result.sweep_dir / "learning_curves.svg")
    learning_curves = (result.sweep_dir / "learning_curves.svg").read_text()
    assert "Validation MSE — future_window (log scale)" in learning_curves
    assert "SG on" in learning_curves
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_MSK(?:_\d{2})?",
        result.sweep_dir.name,
    )
    svg = (result.sweep_dir / "summary.svg").read_text()
    assert "Validation MSE" in svg
    assert "Context normalized effective rank" in svg
    assert "Target normalized effective rank" in svg
    assert all(row["variant"].replace(" | ", " / ") in svg for row in result.rows)
    paired_seed_sets: set[tuple[int, ...]] = set()
    model_seeds: dict[str, set[int]] = {"linear": set(), "nonlinear": set()}
    for row in result.rows:
        run_dir = Path(row["run_dir"])
        assert json.loads((run_dir / "status.json").read_text())["state"] == "complete"
        ET.parse(run_dir / "history.svg")
        artifact = yaml.safe_load((run_dir / "config.yaml").read_text())
        seeds = artifact["derived_seeds"]
        paired_seed_sets.add(
            (
                seeds["system"],
                seeds["train_samples"],
                seeds["validation_samples"],
                seeds["test_samples"],
                seeds["training"],
            )
        )
        model_seeds[row["architecture"]].add(artifact["identity"]["model_seed"])
    assert len(paired_seed_sets) == 1
    assert all(len(seeds) == 1 for seeds in model_seeds.values())
    assert model_seeds["linear"] != model_seeds["nonlinear"]


def test_all_dynamics_sweep_produces_sixteen_cells(tmp_path: Path) -> None:
    result = run_sweep(_config(tmp_path), seeds=[0], system_kind="all")

    assert result.completed_count == 16
    assert len(result.rows) == 16
    assert {row["dynamics"] for row in result.rows} == {"linear", "nonlinear"}
    assert len({row["run_id"] for row in result.rows}) == 16


def test_failed_cell_is_isolated_and_marked_in_every_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = reporting_module.train_experiment

    def fail_one(config: ExperimentConfig, **kwargs: Any) -> Any:
        if (
            config.model.architecture == "linear"
            and not config.training.stop_gradient
            and not config.training.ema.enabled
        ):
            raise RuntimeError("injected cell failure")
        return original(config, **kwargs)

    monkeypatch.setattr(reporting_module, "train_experiment", fail_one)
    result = run_sweep(_config(tmp_path), seeds=[5], system_kind="linear")

    assert result.completed_count == 7
    assert result.failed_count == 1
    failed = next(row for row in result.rows if row["state"] == "failed")
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "injected cell failure"
    status = json.loads((Path(failed["run_dir"]) / "status.json").read_text())
    assert status["state"] == "failed"
    sweep_status = json.loads((result.sweep_dir / "status.json").read_text())
    assert sweep_status["schema_version"] == 2
    assert sweep_status["state"] == "partial_failure"
    assert sweep_status["completed_count"] == 7
    assert sweep_status["failed_count"] == 1
    assert sweep_status["cell_count"] == 8
    assert sweep_status["objectives"] == ["future_window"]
    failed_aggregate = next(row for row in result.aggregates if row["variant"] == failed["variant"])
    assert failed_aggregate["completed_count"] == 0
    assert failed_aggregate["failed_count"] == 1
    assert failed_aggregate["validation_mse_mean"] is None
    assert "FAILED" in (result.sweep_dir / "summary.svg").read_text()

    monkeypatch.setattr(reporting_module, "train_experiment", original)
    retry_progress: list[dict[str, Any]] = []
    retried = run_sweep(
        _config(tmp_path),
        seeds=[5],
        system_kind="linear",
        progress=retry_progress.append,
    )
    assert retried.completed_count == 8
    assert retried.failed_count == 0
    assert sum(event["event"] == "cell_reused" for event in retry_progress) == 7
    assert sum(event["event"] == "cell_complete" for event in retry_progress) == 1


def test_multiple_replicates_are_aggregated_by_variant(tmp_path: Path) -> None:
    result = run_sweep(_config(tmp_path), seeds=[2, 3], system_kind="nonlinear")

    assert result.completed_count == 16
    assert len(result.aggregates) == 8
    assert all(row["completed_count"] == 2 for row in result.aggregates)
    assert all(row["failed_count"] == 0 for row in result.aggregates)


def test_sweep_reuses_completed_runs_and_overwrite_replaces_them(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_sweep(config, seeds=[1])
    reused = run_sweep(config, seeds=[1])
    assert reused.sweep_dir == first.sweep_dir
    assert reused.completed_count == 8

    overwritten = run_sweep(_config(tmp_path, overwrite=True), seeds=[1])
    assert overwritten.sweep_dir != first.sweep_dir


def test_all_objectives_create_sixteen_objective_aware_cells(tmp_path: Path) -> None:
    base = _config(tmp_path)
    config = replace(
        base,
        objective=ObjectiveConfig(
            kind="future_window",
            masked_patches=MaskedPatchesConfig(
                patch_size=1,
                mask_ratio=0.5,
                position_dim=4,
            ),
        ),
    )
    result = run_sweep(config, seeds=[0], system_kind="linear", objectives="all")

    assert result.completed_count == 16
    assert len(result.rows) == 16
    assert len(result.aggregates) == 16
    assert {row["objective"] for row in result.rows} == {
        "future_window",
        "masked_patches",
    }
    assert len({row["run_id"] for row in result.rows}) == 16
    assert len({row["dataset_fingerprint"] for row in result.rows}) == 1
    assert all(row["wall_clock_seconds"] < 1800 for row in result.rows)
    replicate_seeds = {
        yaml.safe_load((Path(row["run_dir"]) / "config.yaml").read_text())["identity"][
            "replicate_seed"
        ]
        for row in result.rows
    }
    assert replicate_seeds == {0}


def test_cli_train_applies_flags_and_prints_machine_readable_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(_config(tmp_path / "ignored"))))
    output_root = tmp_path / "cli-output"

    exit_code = main(
        [
            "train",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--architecture",
            "nonlinear",
            "--sg",
            "off",
            "--ema",
            "on",
            "--system-kind",
            "nonlinear",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["state"] == "complete"
    assert "Run | future_window | nonlinear | SG off | EMA on" in captured.err
    assert "Epoch 1/1" in captured.err
    assert "train MSE" in captured.err
    assert "ETA" in captured.err
    assert "Complete | elapsed" in captured.err
    metrics = json.loads((Path(output["run_dir"]) / "metrics.json").read_text())
    assert metrics["policy"]["stop_gradient"] is False
    assert metrics["policy"]["ema_enabled"] is True
    assert "dynamics-nonlinear_model-nonlinear_sg-off_ema-on" in output["run_id"]


def test_cli_returns_nonzero_after_partial_sweep_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep_dir = tmp_path / "sweep"
    fake = SweepResult(sweep_dir, (), (), 7, 1)
    monkeypatch.setattr("jepa.cli.run_sweep", lambda *args, **kwargs: fake)

    exit_code = main(["sweep", "--seeds", "0", "--output-root", str(tmp_path)])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["state"] == "partial_failure"
