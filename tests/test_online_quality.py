from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from jepa.analysis.quality_features import LiveSpatialAdapter, _encode_masked_pooled
from jepa.analysis.quality_timing import (
    QualityTimingRecorder,
    isolated_quality_evaluation,
)
from jepa.training.images.ijepa_spatial import build_spatial_ijepa_core

_ANALYZER_PATH = Path(__file__).parents[1] / "scripts/analysis/analyze_epoch_quality.py"
_ANALYZER_SPEC = importlib.util.spec_from_file_location("analyze_epoch_quality", _ANALYZER_PATH)
assert _ANALYZER_SPEC is not None and _ANALYZER_SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_ANALYZER_SPEC)
_ANALYZER_SPEC.loader.exec_module(_ANALYZER)
_correlations = _ANALYZER._correlations


def _core():
    return build_spatial_ijepa_core(
        "cnn", patch_dim=8 * 8 * 3, patch_latent_dim=16, num_patches=64
    )


def test_live_adapter_uses_logical_observation_without_checkpoint_file(tmp_path) -> None:
    adapter = LiveSpatialAdapter(
        run_dir=tmp_path / "run",
        observation_id="live_epoch_0007",
        epoch=7,
        global_step=875,
        model_seed=11,
        curriculum="uniform",
        training_uses_shape_metadata=False,
        core=_core(),
    )
    assert adapter.checkpoint_path == Path("live_epoch_0007")
    assert not adapter.checkpoint_path.exists()
    assert adapter.checkpoint_hash == adapter.checkpoint_hash


def test_masked_encoding_batches_variable_length_masks_in_one_encoder_call() -> None:
    core = _core()
    patches = torch.randn(4, 64, 8 * 8 * 3)
    masks = [
        torch.arange(3),
        torch.arange(5, 17),
        torch.tensor([0, 2, 7, 11, 31, 63]),
        torch.arange(19, 64),
    ]
    expected = torch.cat(
        [
            core.context_encoder(patches[row : row + 1, mask]).mean(dim=1)
            for row, mask in enumerate(masks)
        ],
        dim=0,
    )

    calls = 0

    def count_call(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    hook = core.context_encoder.register_forward_hook(count_call)
    try:
        actual = _encode_masked_pooled(core.context_encoder, patches, masks)
    finally:
        hook.remove()

    assert calls == 1
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_isolated_quality_evaluation_restores_modes_and_cpu_rng() -> None:
    core = _core()
    core.context_encoder.train()
    core.predictor.eval()
    core.target_encoder.train()
    modes = (
        core.context_encoder.training,
        core.predictor.training,
        core.target_encoder.training,
    )
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    with isolated_quality_evaluation(core, torch.device("cpu")):
        assert not core.context_encoder.training
        assert not core.predictor.training
        assert not core.target_encoder.training
        torch.rand(10)
    assert torch.equal(before, torch.random.get_rng_state())
    assert modes == (
        core.context_encoder.training,
        core.predictor.training,
        core.target_encoder.training,
    )


def test_timing_recorder_upserts_epoch_rows(tmp_path) -> None:
    recorder = QualityTimingRecorder(torch.device("cpu"))
    with recorder.stage("factorization_label_free"):
        torch.linalg.eigh(torch.eye(4))
    with recorder.stage("q1_cross_covariance", metric=True):
        torch.linalg.matrix_norm(torch.eye(4))
    path = recorder.upsert_jsonl(
        tmp_path / "timings.jsonl",
        run_id="uniform",
        epoch=1,
        global_step=100,
    )
    recorder.upsert_jsonl(path, run_id="uniform", epoch=1, global_step=100)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert {(row["kind"], row["name"]) for row in rows} == {
        ("stage", "factorization_label_free"),
        ("metric_formula", "q1_cross_covariance"),
    }
    assert all(row["seconds"] >= 0 for row in rows)


def test_epoch_correlations_include_raw_and_first_difference_coefficients() -> None:
    rows = [
        {
            "metric_name": "q1_cross_covariance",
            "epoch": epoch,
            "q_value": float(epoch),
            "classification_accuracy": float(epoch) / 10,
            "heldout_jepa_loss": float(11 - epoch),
        }
        for epoch in range(1, 11)
    ]
    results = _correlations(rows)
    accuracy = next(
        row for row in results if row["outcome"] == "classification_accuracy"
    )
    loss = next(row for row in results if row["outcome"] == "heldout_jepa_loss")
    assert accuracy["pearson"] == pytest.approx(1.0)
    assert accuracy["spearman"] == pytest.approx(1.0)
    assert accuracy["delta_pearson"] is None  # constant first differences
    assert loss["pearson"] == pytest.approx(-1.0)
