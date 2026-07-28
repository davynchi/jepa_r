"""Leakage-free, content-addressed Shapes3D evaluation manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from jepa.configs.base import derive_seed
from jepa.configs.images.shapes3d import Shapes3DDataConfig
from jepa.data.images.shapes3d import (
    Shapes3DSource,
    sample_shapes3d_counterfactual_factor_rows,
    sample_shapes3d_static_factor_row,
)

SHAPES3D_ROWS = 480_000
MANIFEST_VERSION = "1"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class QualitySplitManifest:
    version: str
    manifest_seed: int
    data_config_hash: str
    excluded_indices: tuple[int, ...]
    banks: Mapping[str, tuple[int, ...]]
    named_seeds: Mapping[str, int]
    content_hash: str

    def __post_init__(self) -> None:
        banks = {name: tuple(int(index) for index in values) for name, values in self.banks.items()}
        named_seeds = {name: int(value) for name, value in self.named_seeds.items()}
        object.__setattr__(self, "banks", MappingProxyType(banks))
        object.__setattr__(self, "named_seeds", MappingProxyType(named_seeds))
        seen: set[int] = set(self.excluded_indices)
        for name, indices in banks.items():
            if len(indices) != len(set(indices)):
                raise ValueError(f"bank {name} contains duplicate indices")
            overlap = seen.intersection(indices)
            if overlap and name in _DISJOINT_ROOT_BANKS:
                raise ValueError(f"bank {name} overlaps excluded or earlier root bank")
            if name in _DISJOINT_ROOT_BANKS:
                seen.update(indices)
        if self.content_hash != _manifest_hash(self.payload(include_hash=False)):
            raise ValueError("manifest content hash mismatch")

    def payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "manifest_seed": self.manifest_seed,
            "data_config_hash": self.data_config_hash,
            "excluded_indices": list(self.excluded_indices),
            "banks": {name: list(values) for name, values in self.banks.items()},
            "named_seeds": dict(self.named_seeds),
        }
        if include_hash:
            payload["content_hash"] = self.content_hash
        return payload

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.payload(), indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)

    @classmethod
    def read(cls, path: str | Path) -> QualitySplitManifest:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("manifest root must be an object")
        return cls(
            version=str(payload["version"]),
            manifest_seed=int(payload["manifest_seed"]),
            data_config_hash=str(payload["data_config_hash"]),
            excluded_indices=tuple(payload["excluded_indices"]),
            banks={name: tuple(values) for name, values in payload["banks"].items()},
            named_seeds={name: int(value) for name, value in payload["named_seeds"].items()},
            content_hash=str(payload["content_hash"]),
        )


_DISJOINT_ROOT_BANKS = {
    "unlabeled_fit_bank",
    "unlabeled_metric_bank",
    "classifier_train",
    "classifier_validation",
    "classifier_test",
}


def _manifest_hash(payload: Mapping[str, object]) -> str:
    return _canonical_hash(payload)


def reconstruct_static_indices(config: Shapes3DDataConfig) -> tuple[int, ...]:
    indices: set[int] = set()
    for size, seed in (
        (config.num_train_samples, config.train_sample_seed),
        (config.num_val_samples, config.validation_sample_seed),
        (config.num_test_samples, config.test_sample_seed),
    ):
        for sample_index in range(size):
            _, flat = sample_shapes3d_static_factor_row(seed, sample_index)
            indices.add(flat)
    return tuple(sorted(indices))


def reconstruct_counterfactual_indices(
    config: Shapes3DDataConfig,
    *,
    transformation_model_seeds: Sequence[int],
    ref_size: int,
) -> tuple[int, ...]:
    indices: set[int] = set()
    for model_seed in transformation_model_seeds:
        rows = sample_shapes3d_counterfactual_factor_rows(
            config,
            num_pairs=ref_size,
            seed=derive_seed(model_seed, "coordinate-transform-pairs"),
        )
        indices.update(int(value) for value in rows.same_flat_indices_1)
        indices.update(int(value) for value in rows.same_flat_indices_2)
    return tuple(sorted(indices))


def _sample_uniform(pool: np.ndarray, size: int, generator: np.random.Generator) -> np.ndarray:
    if size > pool.size:
        raise ValueError("requested bank is larger than remaining Shapes3D pool")
    selected_positions = generator.choice(pool.size, size=size, replace=False)
    return np.sort(pool[selected_positions])


def _sample_shape_balanced(
    pool: np.ndarray, size: int, generator: np.random.Generator
) -> np.ndarray:
    if size % 4:
        raise ValueError("shape-balanced bank size must be divisible by four")
    per_shape = size // 4
    selected: list[np.ndarray] = []
    for shape in range(4):
        candidates = pool[((pool // 15) % 4) == shape]
        selected.append(_sample_uniform(candidates, per_shape, generator))
    return np.sort(np.concatenate(selected))


def build_quality_manifest(
    config: Shapes3DDataConfig,
    *,
    data_config_hash: str,
    manifest_seed: int,
    transformation_model_seeds: Sequence[int] = (),
    transformation_ref_size: int = 1024,
    bank_sizes: Mapping[str, int] | None = None,
) -> QualitySplitManifest:
    sizes = {
        "unlabeled_fit_bank": 8_000,
        "unlabeled_metric_bank": 2_000,
        "classifier_train": 8_000,
        "classifier_validation": 2_000,
        "classifier_test": 2_000,
        **(bank_sizes or {}),
    }
    excluded = set(reconstruct_static_indices(config))
    excluded.update(
        reconstruct_counterfactual_indices(
            config,
            transformation_model_seeds=transformation_model_seeds,
            ref_size=transformation_ref_size,
        )
    )
    generator = np.random.default_rng(manifest_seed)
    remaining = np.setdiff1d(
        np.arange(SHAPES3D_ROWS, dtype=np.int64), np.fromiter(excluded, dtype=np.int64)
    )
    banks: dict[str, tuple[int, ...]] = {}
    for name in ("unlabeled_fit_bank", "unlabeled_metric_bank"):
        selected = _sample_uniform(remaining, sizes[name], generator)
        banks[name] = tuple(int(value) for value in selected)
        remaining = np.setdiff1d(remaining, selected, assume_unique=True)
    for name in ("classifier_train", "classifier_validation", "classifier_test"):
        selected = _sample_shape_balanced(remaining, sizes[name], generator)
        banks[name] = tuple(int(value) for value in selected)
        remaining = np.setdiff1d(remaining, selected, assume_unique=True)

    metric = np.array(banks["unlabeled_metric_bank"], dtype=np.int64)
    cursor = 0
    for name, count in (
        ("metric_bank", min(1024, len(metric))),
        ("jacobian_bank", min(32, max(len(metric) - 1024, 0))),
        ("gradient_bank", min(32, max(len(metric) - 1056, 0))),
        ("virtual_conditioning_bank", min(16, max(len(metric) - 1088, 0))),
        ("replay_bank", min(128, max(len(metric) - 1104, 0))),
    ):
        banks[name] = tuple(int(value) for value in metric[cursor : cursor + count])
        cursor += count
    banks["integration_reserve"] = tuple(int(value) for value in metric[cursor:])
    fit_bank = np.array(banks["unlabeled_fit_bank"], dtype=np.int64)
    banks["decomposition_refit_bank"] = tuple(
        int(value) for value in _sample_uniform(fit_bank, min(256, len(fit_bank)), generator)
    )
    named_seeds = {
        name: derive_seed(manifest_seed, name)
        for name in (
            "label_free_mask_a_seed",
            "label_free_mask_b_seed",
            "q7_target_mask_seed",
            "q8_loss_mask_seed",
            "q14_loss_mask_seed",
            "q15_replay_seed",
        )
    }
    base = {
        "version": MANIFEST_VERSION,
        "manifest_seed": manifest_seed,
        "data_config_hash": data_config_hash,
        "excluded_indices": sorted(excluded),
        "banks": {name: list(values) for name, values in banks.items()},
        "named_seeds": named_seeds,
    }
    return QualitySplitManifest(
        version=MANIFEST_VERSION,
        manifest_seed=manifest_seed,
        data_config_hash=data_config_hash,
        excluded_indices=tuple(sorted(excluded)),
        banks=banks,
        named_seeds=named_seeds,
        content_hash=_manifest_hash(base),
    )


def iter_manifest_images(
    source: Shapes3DSource, indices: Sequence[int], *, batch_size: int
) -> Iterator[tuple[tuple[int, ...], torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered = tuple(int(index) for index in indices)
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        yield chunk, torch.from_numpy(source.images_batch(np.asarray(chunk, dtype=np.int64)))


__all__ = [
    "MANIFEST_VERSION",
    "QualitySplitManifest",
    "build_quality_manifest",
    "iter_manifest_images",
    "reconstruct_counterfactual_indices",
    "reconstruct_static_indices",
]
