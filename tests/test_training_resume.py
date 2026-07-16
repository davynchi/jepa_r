from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

import jepa.training as training_module
from jepa.config import (
    DataConfig,
    EMAConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from jepa.training import HISTORY_COLUMNS, train_experiment


def _config(
    root: Path,
    *,
    epochs: int = 3,
    stop_gradient: bool = True,
    ema: bool = True,
    evaluation_every: int = 2,
    checkpoint_every: int = 2,
    overwrite: bool = False,
) -> ExperimentConfig:
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
            process_noise=0.01,
            observation_noise=0.01,
        ),
        model=ModelConfig(
            architecture="nonlinear",
            latent_dim=3,
            hidden_dim=5,
            hidden_layers=1,
        ),
        training=TrainingConfig(
            seed=17,
            learning_rate=0.005,
            batch_size=4,
            epochs=epochs,
            device="cpu",
            evaluation_every_epochs=evaluation_every,
            checkpoint_every_epochs=checkpoint_every,
            stop_gradient=stop_gradient,
            ema=EMAConfig(enabled=ema, decay=0.9),
        ),
        evaluation=EvaluationConfig(probe_ridge=1.0e-4),
        output=OutputConfig(root=str(root), overwrite=overwrite),
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_training_writes_atomic_complete_artifacts_and_final_rows(tmp_path: Path) -> None:
    result = train_experiment(_config(tmp_path))

    assert {path.name for path in result.run_dir.iterdir()} == {
        "checkpoint.pt",
        "config.yaml",
        "history.csv",
        "history.svg",
        "metrics.json",
        "status.json",
    }
    assert not any(path.name.startswith(".") for path in result.run_dir.iterdir())
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_MSK(?:_\d{2})?", result.run_dir.name)
    ET.parse(result.run_dir / "history.svg")
    history_svg = (result.run_dir / "history.svg").read_text()
    assert "Objective MSE (log scale)" in history_svg
    assert "Context normalized effective rank" in history_svg
    assert "Training gradient norms (log scale)" in history_svg
    assert "epoch 2:" in history_svg
    assert "epoch 3:" in history_svg
    status = json.loads((result.run_dir / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["completed_epoch"] == 3
    checkpoint = _load_checkpoint(result.run_dir / "checkpoint.pt")
    assert checkpoint["epoch"] == 3

    with (result.run_dir / "history.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == HISTORY_COLUMNS
        identities = [(int(row["epoch"]), row["split"]) for row in reader]
    assert identities == [
        (2, "train"),
        (2, "validation"),
        (3, "train"),
        (3, "validation"),
        (3, "test"),
    ]


@pytest.mark.parametrize("stop_gradient", [False, True])
@pytest.mark.parametrize("ema", [False, True])
def test_all_four_sg_ema_policies_train(tmp_path: Path, stop_gradient: bool, ema: bool) -> None:
    result = train_experiment(
        _config(
            tmp_path,
            epochs=1,
            stop_gradient=stop_gradient,
            ema=ema,
            evaluation_every=1,
            checkpoint_every=1,
        )
    )

    assert result.metrics["policy"]["stop_gradient"] is stop_gradient
    assert result.metrics["policy"]["ema_enabled"] is ema
    assert result.metrics["objective"]["test"] >= 0


def test_test_split_is_evaluated_only_during_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    original = training_module._collect_split_outputs

    def recording_collect(*args: Any, **kwargs: Any) -> Any:
        seen.append(args[1].split)
        return original(*args, **kwargs)

    monkeypatch.setattr(training_module, "_collect_split_outputs", recording_collect)
    train_experiment(_config(tmp_path, epochs=3, evaluation_every=1))

    assert seen.count("test") == 1
    assert seen[-1] == "test"


def test_failure_is_recorded_without_marking_run_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_epoch(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(training_module, "_train_epoch", fail_epoch)
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="injected failure"):
        train_experiment(config)

    run_dir = next(tmp_path.iterdir())
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["completed_epoch"] == 0
    assert status["error"] == {"type": "RuntimeError", "message": "injected failure"}
    assert not (run_dir / "metrics.json").exists()


def test_cpu_resume_matches_uninterrupted_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = train_experiment(_config(tmp_path / "baseline", epochs=4))
    interrupted_config = _config(tmp_path / "resumed", epochs=4)
    original = training_module._train_epoch

    def interrupt_at_three(*args: Any, **kwargs: Any) -> Any:
        if args[-1] == 3:
            raise RuntimeError("interrupt")
        return original(*args, **kwargs)

    monkeypatch.setattr(training_module, "_train_epoch", interrupt_at_three)
    with pytest.raises(RuntimeError, match="interrupt"):
        train_experiment(interrupted_config)
    interrupted_dir = next((tmp_path / "resumed").iterdir())
    interrupted_checkpoint = _load_checkpoint(interrupted_dir / "checkpoint.pt")
    assert interrupted_checkpoint["epoch"] == 2

    monkeypatch.setattr(training_module, "_train_epoch", original)
    resumed = train_experiment(interrupted_config, resume_from=interrupted_dir / "checkpoint.pt")

    baseline_checkpoint = _load_checkpoint(baseline.run_dir / "checkpoint.pt")
    resumed_checkpoint = _load_checkpoint(resumed.run_dir / "checkpoint.pt")
    for key in ("context_encoder", "predictor", "target_encoder", "optimizer"):
        _assert_nested_equal(baseline_checkpoint[key], resumed_checkpoint[key])
    assert baseline.metrics["objective"] == resumed.metrics["objective"]
    assert baseline.metrics["probes"] == resumed.metrics["probes"]
    assert baseline.metrics["policy"] == resumed.metrics["policy"]

    def comparable(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in row.items() if key not in {"run_id", "elapsed_seconds"}}
            for row in rows
        ]

    assert comparable(baseline.history) == comparable(resumed.history)


def test_resume_can_extend_epochs_without_changing_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = train_experiment(_config(tmp_path / "baseline", epochs=4))
    short_config = _config(tmp_path / "extended", epochs=2)
    original_final_metrics = training_module._final_metrics

    def fail_finalization(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("finalization failure")

    monkeypatch.setattr(training_module, "_final_metrics", fail_finalization)
    with pytest.raises(RuntimeError, match="finalization failure"):
        train_experiment(short_config)
    run_dir = next((tmp_path / "extended").iterdir())
    original_run_id = _load_checkpoint(run_dir / "checkpoint.pt")["run_id"]
    assert _load_checkpoint(run_dir / "checkpoint.pt")["epoch"] == 2

    monkeypatch.setattr(training_module, "_final_metrics", original_final_metrics)
    extended_config = replace(
        short_config,
        training=replace(short_config.training, epochs=4),
    )
    resumed = train_experiment(extended_config, resume_from=run_dir / "checkpoint.pt")

    assert resumed.run_id == original_run_id
    assert resumed.run_dir == run_dir
    checkpoint = _load_checkpoint(run_dir / "checkpoint.pt")
    assert checkpoint["initial_config"]["training"]["epochs"] == 2
    assert checkpoint["config"]["training"]["epochs"] == 4
    artifact = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert (
        artifact["identity"]["initial_config_hash"] != artifact["identity"]["current_config_hash"]
    )
    assert len(artifact["resume_history"]) == 1
    assert [(row["epoch"], row["split"]) for row in resumed.history].count((4, "test")) == 1
    assert not any(row["split"] == "test" and row["epoch"] == 2 for row in resumed.history)

    baseline_checkpoint = _load_checkpoint(baseline.run_dir / "checkpoint.pt")
    for key in ("context_encoder", "predictor", "target_encoder", "optimizer"):
        _assert_nested_equal(baseline_checkpoint[key], checkpoint[key])


def test_resume_rejects_complete_run_and_config_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1, evaluation_every=1, checkpoint_every=1)
    result = train_experiment(config)
    checkpoint = result.run_dir / "checkpoint.pt"

    with pytest.raises(ValueError, match="completed runs"):
        train_experiment(config, resume_from=checkpoint)

    status = json.loads((result.run_dir / "status.json").read_text())
    status["state"] = "failed"
    (result.run_dir / "status.json").write_text(json.dumps(status))
    changed = replace(config, model=replace(config.model, latent_dim=4))
    with pytest.raises(ValueError, match="differs"):
        train_experiment(changed, resume_from=checkpoint)


def test_resume_rejects_duplicate_history_rows(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=2, evaluation_every=1, checkpoint_every=1)
    result = train_experiment(config)
    status = json.loads((result.run_dir / "status.json").read_text())
    status["state"] = "failed"
    (result.run_dir / "status.json").write_text(json.dumps(status))
    history_path = result.run_dir / "history.csv"
    lines = history_path.read_text().splitlines()
    history_path.write_text("\n".join([*lines, lines[1], ""]))

    with pytest.raises(ValueError, match="duplicate"):
        train_experiment(config, resume_from=result.run_dir / "checkpoint.pt")


def test_fresh_runs_get_unique_timestamp_directories(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1, evaluation_every=1, checkpoint_every=1)
    first = train_experiment(config)
    second = train_experiment(config)

    assert second.run_dir != first.run_dir
    assert second.run_id == first.run_id
    assert json.loads((second.run_dir / "status.json").read_text())["state"] == "complete"


def test_synthetic_schema_v1_checkpoint_remains_resumable(tmp_path: Path) -> None:
    original = _config(tmp_path, epochs=1, evaluation_every=1, checkpoint_every=1)
    result = train_experiment(original)
    checkpoint_path = result.run_dir / "checkpoint.pt"
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint["schema_version"] = 1
    checkpoint.pop("objective")
    checkpoint.pop("dataset_fingerprint")
    checkpoint.pop("replicate_seed")
    checkpoint["config"].pop("objective")
    checkpoint["initial_config"].pop("objective")
    torch.save(checkpoint, checkpoint_path)

    status = json.loads((result.run_dir / "status.json").read_text())
    status.update(schema_version=1, state="failed")
    (result.run_dir / "status.json").write_text(json.dumps(status))
    artifact_path = result.run_dir / "config.yaml"
    artifact = yaml.safe_load(artifact_path.read_text())
    artifact["schema_version"] = 1
    artifact["config"].pop("objective")
    artifact_path.write_text(yaml.safe_dump(artifact))

    extended = replace(original, training=replace(original.training, epochs=2))
    resumed = train_experiment(extended, resume_from=checkpoint_path)

    assert resumed.metrics["schema_version"] == 2
    assert _load_checkpoint(checkpoint_path)["schema_version"] == 2
