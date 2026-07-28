from __future__ import annotations

import json

import pytest
import torch

from jepa.analysis.quality_features import CheckpointFeatureBundle, SpatialCheckpointAdapter
from jepa.training.images.ijepa_spatial import build_spatial_ijepa_core, save_spatial_checkpoint


def _write_run(tmp_path, *, spatial_overrides=None):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    spatial = {
        "dataset": "shapes3d",
        "architecture": "cnn",
        "patch_size": 8,
        "patch_latent_dim": 16,
        "seed": 3,
        "weighting": {"method": "uniform", "coordinate_importance": "covariance"},
        **(spatial_overrides or {}),
    }
    (run_dir / "config.json").write_text(json.dumps({"spatial": spatial}))
    core = build_spatial_ijepa_core("cnn", patch_dim=8 * 8 * 3, patch_latent_dim=16, num_patches=64)
    checkpoint = checkpoint_dir / "epoch_0001.pt"
    save_spatial_checkpoint(checkpoint, core, epoch=1, global_step=10)
    return run_dir, checkpoint, core


def test_real_checkpoint_adapter_does_not_mutate_state(tmp_path) -> None:
    run_dir, checkpoint, core = _write_run(tmp_path)
    before = {name: value.clone() for name, value in core.context_encoder.state_dict().items()}
    adapter = SpatialCheckpointAdapter.load(run_dir, checkpoint)
    assert adapter.epoch == 1
    assert adapter.global_step == 10
    assert adapter.model_seed == 3
    assert all(
        torch.equal(before[name], value)
        for name, value in adapter.core.context_encoder.state_dict().items()
    )


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"dataset": "tiny-imagenet"}, "dataset"),
        ({"architecture": "mlp"}, "architecture"),
        ({"patch_size": 4}, "patch_size"),
        ({"patch_latent_dim": 8}, "patch_latent_dim"),
    ],
)
def test_adapter_rejects_incompatible_config(tmp_path, override, field) -> None:
    run_dir, checkpoint, _ = _write_run(tmp_path, spatial_overrides=override)
    with pytest.raises(ValueError, match=field):
        SpatialCheckpointAdapter.load(run_dir, checkpoint)


def test_feature_bundle_cache_is_content_addressed_and_tamper_evident(tmp_path) -> None:
    metadata = {
        "checkpoint_hash": "a",
        "manifest_hash": "b",
        "feature_version": "1",
        "bank": "metric",
        "view": "full",
        "ordered_index_hash": "c",
        "mask_seed": -1,
        "architecture": "cnn",
        "latent_dim": 16,
    }
    bundle = CheckpointFeatureBundle(metadata, torch.randn(4, 16))
    path = bundle.save(tmp_path)
    loaded = CheckpointFeatureBundle.load(path, expected_metadata=metadata)
    assert torch.equal(bundle.features, loaded.features)
    with pytest.raises(ValueError, match="cache_identity_mismatch"):
        CheckpointFeatureBundle.load(path, expected_metadata={**metadata, "bank": "other"})
