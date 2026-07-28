"""Strict spatial-checkpoint adapter and immutable feature bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from jepa.analysis.quality_manifests import iter_manifest_images
from jepa.configs.base import derive_seed
from jepa.data.images.shapes3d import Shapes3DSource
from jepa.models.patches import patchify
from jepa.training.images.ijepa_spatial import (
    MaskConfig,
    SpatialIJEPACore,
    build_spatial_ijepa_core,
    load_spatial_checkpoint,
    sample_masks,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SpatialCheckpointAdapter:
    run_dir: Path
    checkpoint_path: Path
    checkpoint_hash: str
    epoch: int
    global_step: int
    model_seed: int
    curriculum: str
    training_uses_shape_metadata: bool
    core: SpatialIJEPACore

    @classmethod
    def load(cls, run_dir: str | Path, checkpoint_path: str | Path) -> SpatialCheckpointAdapter:
        run = Path(run_dir)
        config = json.loads((run / "config.json").read_text())
        spatial = config.get("spatial")
        if not isinstance(spatial, dict):
            raise ValueError("checkpoint_metadata_mismatch: missing spatial config")
        expected = {
            "dataset": "shapes3d",
            "architecture": "cnn",
            "patch_size": 8,
            "patch_latent_dim": 16,
        }
        for key, value in expected.items():
            if spatial.get(key) != value:
                raise ValueError(f"checkpoint_metadata_mismatch: {key}")
        checkpoint_file = Path(checkpoint_path)
        try:
            checkpoint = load_spatial_checkpoint(checkpoint_file)
        except ValueError as error:
            raise ValueError("unsupported_checkpoint_schema") from error
        for key in ("context_encoder", "predictor", "target_encoder", "epoch"):
            if key not in checkpoint:
                raise ValueError(f"checkpoint_metadata_mismatch: missing {key}")
        core = build_spatial_ijepa_core(
            "cnn", patch_dim=8 * 8 * 3, patch_latent_dim=16, num_patches=64
        )
        try:
            core.context_encoder.load_state_dict(checkpoint["context_encoder"], strict=True)
            core.predictor.load_state_dict(checkpoint["predictor"], strict=True)
            core.target_encoder.load_state_dict(checkpoint["target_encoder"], strict=True)
        except RuntimeError as error:
            raise ValueError("checkpoint_metadata_mismatch: state shape") from error
        core.context_encoder.eval()
        core.predictor.eval()
        core.target_encoder.eval()
        weighting = spatial.get("weighting", {})
        curriculum = str(weighting.get("method", "uniform"))
        if curriculum == "coord":
            curriculum = f"coord-{weighting.get('coordinate_importance', 'covariance')}"
        return cls(
            run_dir=run,
            checkpoint_path=checkpoint_file,
            checkpoint_hash=file_sha256(checkpoint_file),
            epoch=int(checkpoint["epoch"]),
            global_step=int(checkpoint.get("global_step") or 0),
            model_seed=int(spatial.get("seed", 0)),
            curriculum=curriculum,
            training_uses_shape_metadata=curriculum == "coord-transformation",
            core=core,
        )


@dataclass(frozen=True, slots=True)
class LiveSpatialAdapter:
    """Adapter for an in-memory model observed immediately after an epoch."""

    run_dir: Path
    observation_id: str
    epoch: int
    global_step: int
    model_seed: int
    curriculum: str
    training_uses_shape_metadata: bool
    core: SpatialIJEPACore

    @property
    def checkpoint_path(self) -> Path:
        # Backward-compatible logical identity for QualityStore. No file is read.
        return Path(self.observation_id)

    @property
    def checkpoint_hash(self) -> str:
        payload = (
            f"live|{self.run_dir.resolve()}|{self.observation_id}|"
            f"{self.epoch}|{self.global_step}|{self.model_seed}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointFeatureBundle:
    metadata: Mapping[str, str | int]
    features: torch.Tensor

    def __post_init__(self) -> None:
        matrix = self.features.detach().to(device="cpu", dtype=torch.float32)
        if matrix.ndim != 2 or not torch.isfinite(matrix).all():
            raise ValueError("feature bundle requires a finite [N,d] matrix")
        object.__setattr__(self, "features", matrix)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def cache_key(self) -> str:
        payload = json.dumps(dict(self.metadata), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, cache_dir: str | Path) -> Path:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.cache_key}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                features=self.features.numpy(),
                metadata=np.array(json.dumps(dict(self.metadata), sort_keys=True)),
            )
        temporary.replace(path)
        return path

    @classmethod
    def load(
        cls, path: str | Path, *, expected_metadata: Mapping[str, str | int]
    ) -> CheckpointFeatureBundle:
        try:
            with np.load(Path(path), allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata"].item()))
                features = torch.from_numpy(payload["features"].copy())
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError("cache_identity_mismatch") from error
        if metadata != dict(expected_metadata):
            raise ValueError("cache_identity_mismatch")
        bundle = cls(metadata, features)
        if Path(path).stem != bundle.cache_key:
            raise ValueError("cache_identity_mismatch")
        return bundle


def feature_metadata(
    adapter: SpatialCheckpointAdapter | LiveSpatialAdapter,
    *,
    manifest_hash: str,
    bank: str,
    view: str,
    indices: Sequence[int],
    mask_seed: int | None,
) -> dict[str, str | int]:
    index_hash = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
    return {
        "checkpoint_hash": adapter.checkpoint_hash,
        "manifest_hash": manifest_hash,
        "feature_version": "1",
        "bank": bank,
        "view": view,
        "ordered_index_hash": index_hash,
        "mask_seed": -1 if mask_seed is None else mask_seed,
        "architecture": "cnn",
        "latent_dim": 16,
    }


def _encode_masked_pooled(
    encoder: torch.nn.Module,
    patches: torch.Tensor,
    masks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Encode variable-length masked patch sets in one batched encoder call."""
    if patches.ndim != 3 or len(masks) != patches.shape[0]:
        raise ValueError("masked encoding requires [B,N,D] patches and one mask per row")
    if not masks or any(mask.ndim != 1 or mask.numel() == 0 for mask in masks):
        raise ValueError("masked encoding requires non-empty 1-D masks")

    device = patches.device
    lengths = torch.tensor([mask.numel() for mask in masks], device=device)
    padded_masks = pad_sequence(
        [mask.to(dtype=torch.long) for mask in masks],
        batch_first=True,
        padding_value=0,
    ).to(device)
    batch_rows = torch.arange(patches.shape[0], device=device).unsqueeze(1)
    masked_patches = patches[batch_rows, padded_masks]
    latents = encoder(masked_patches)

    valid = (
        torch.arange(padded_masks.shape[1], device=device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    summed = (latents * valid.unsqueeze(-1)).sum(dim=1)
    return summed / lengths.to(dtype=latents.dtype).unsqueeze(1)


@torch.no_grad()
def extract_feature_bundle(
    adapter: SpatialCheckpointAdapter | LiveSpatialAdapter,
    source: Shapes3DSource,
    indices: Sequence[int],
    *,
    manifest_hash: str,
    bank: str,
    view: str,
    batch_size: int,
    mask_seed: int | None = None,
) -> CheckpointFeatureBundle:
    if view not in {"full", "mask_a", "mask_b"}:
        raise ValueError("unknown feature view")
    if view != "full" and mask_seed is None:
        raise ValueError("masked feature views require mask_seed")
    resolved_mask_seed = -1 if mask_seed is None else mask_seed
    device = next(adapter.core.context_encoder.parameters()).device
    output = torch.empty((len(indices), 16), dtype=torch.float32)
    cursor = 0
    for chunk_indices, frames in iter_manifest_images(source, indices, batch_size=batch_size):
        patches = patchify(frames, 8).to(device)
        if view == "full":
            encoded = adapter.core.context_encoder(patches).mean(dim=1)
        else:
            masks: list[torch.Tensor] = []
            for flat_index in chunk_indices:
                generator = torch.Generator(device="cpu").manual_seed(
                    derive_seed(resolved_mask_seed, "flat_index", flat_index)
                )
                context_masks, _ = sample_masks(8, 8, MaskConfig(), generator)
                masks.append(context_masks[0])
            encoded = _encode_masked_pooled(adapter.core.context_encoder, patches, masks)
        output[cursor : cursor + len(chunk_indices)] = encoded.detach().cpu()
        cursor += len(chunk_indices)
    return CheckpointFeatureBundle(
        feature_metadata(
            adapter,
            manifest_hash=manifest_hash,
            bank=bank,
            view=view,
            indices=indices,
            mask_seed=mask_seed,
        ),
        output,
    )


__all__ = [
    "CheckpointFeatureBundle",
    "LiveSpatialAdapter",
    "SpatialCheckpointAdapter",
    "extract_feature_bundle",
    "feature_metadata",
    "file_sha256",
]
