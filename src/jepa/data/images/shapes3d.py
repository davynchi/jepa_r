"""Shapes3D-backed entity/context world: exact lookup, no rendering.

The hidden generative process is a slow entity (here, ``shape``) and a faster
context. Every observation is an *exact* image from
DeepMind's 3D Shapes dataset, indexed by a discrete factor combination. The
context process is therefore a discrete random walk over factor indices
(bounded step size, occasional zero-step) rather than a continuous AR(1)
process -- there is no snapping/interpolation error because every state the
walk can reach is already a real row in the dataset.

Built to satisfy the same duck-typed dataset interface
(``.observations``, ``.entities``, ``.contexts``, ``config.trajectory_length``,
``config.observation_dim``) as the vector world, so
``jepa.data.timeseries.entity_context.make_temporal_pairs`` / ``make_hierarchical_pairs`` and
``jepa.training.timeseries``'s per-epoch training helpers run unmodified here too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from jepa.configs.base import derive_seed
from jepa.configs.images.shapes3d import (
    SHAPES3D_CONTEXT_FACTORS,
    SHAPES3D_FACTOR_NAMES,
    SHAPES3D_FACTOR_SIZES,
    Shapes3DDataConfig,
)

Split = str

_STRIDES: dict[str, int] = {}
_base = 1
for _name in reversed(SHAPES3D_FACTOR_NAMES):
    _STRIDES[_name] = _base
    _base *= SHAPES3D_FACTOR_SIZES[_name]


def flat_index(factor_indices: dict[str, int]) -> int:
    """The dataset's own row-major index formula (see the 3d-shapes README)."""
    return sum(factor_indices[name] * _STRIDES[name] for name in SHAPES3D_FACTOR_NAMES)


_STRIDE_VECTOR = np.array([_STRIDES[name] for name in SHAPES3D_FACTOR_NAMES], dtype=np.int64)


def flat_index_batch(factor_matrix: np.ndarray) -> np.ndarray:
    """Vectorized ``flat_index``: ``factor_matrix`` is [N, len(SHAPES3D_FACTOR_NAMES)],
    columns in ``SHAPES3D_FACTOR_NAMES`` order. Returns [N] flat indices."""
    return (factor_matrix * _STRIDE_VECTOR).sum(axis=-1)


def factor_row_from_flat_index(index: int) -> torch.Tensor:
    """Invert the official row-major index without loading labels or images."""
    if index < 0 or index >= _base:
        raise ValueError(f"flat index must be in [0, {_base})")
    remainder = index
    values: list[int] = []
    for name in SHAPES3D_FACTOR_NAMES:
        stride = _STRIDES[name]
        value, remainder = divmod(remainder, stride)
        values.append(value)
    return torch.tensor(values, dtype=torch.long)


def _convert_to_memmap(h5_path: Path, npy_path: Path, *, batch_size: int = 20_000) -> None:
    """One-time sequential conversion of the images array to a local ``.npy``.

    ``3dshapes.h5`` is gzip-chunked as ``(15000, 4, 4, 1)`` -- each chunk spans
    15,000 images but only a 4x4x1 pixel patch, so reading *one full image*
    touches 768 chunks. Scattered random single-image reads are consequently
    catastrophic (~190s for one image remotely; tens of seconds for a few
    thousand images even locally with a large chunk cache). Sequential,
    contiguous reads hit each chunk exactly once and are ~2 orders of
    magnitude faster (measured: 20,000 images in ~1.6s). So we pay that
    sequential cost exactly once, cache the result as an uncompressed
    memory-mapped array, and every subsequent random access -- however
    scattered -- is then plain in-memory/OS-page-cache numpy indexing.
    """
    tmp_path = npy_path.with_suffix(".npy.tmp")
    with h5py.File(h5_path, "r") as source:
        images = source["images"]
        labels = source["labels"][:]
        memmap = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=images.dtype, shape=images.shape
        )
        for start in range(0, images.shape[0], batch_size):
            end = min(start + batch_size, images.shape[0])
            memmap[start:end] = images[start:end]
        memmap.flush()
        del memmap
    tmp_path.rename(npy_path)
    np.save(npy_path.with_name(npy_path.stem + "_labels.npy"), labels)


class Shapes3DSource:
    """A handle onto ``3dshapes.h5``, backed by a local memory-mapped cache.

    The file must be downloaded once
    (https://storage.googleapis.com/3d-shapes/3dshapes.h5, ~267 MB); on first
    use this class additionally converts it, once, into an uncompressed
    ``.npy`` memmap next to it (~5.9 GB, see :func:`_convert_to_memmap` for
    why) so all later random access is fast.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Shapes3D file not found at {self.path}. Download it once with:\n"
                f'  curl -o "{self.path}" https://storage.googleapis.com/3d-shapes/3dshapes.h5'
            )
        cache_path = self.path.with_suffix(".npy")
        if not cache_path.exists():
            _convert_to_memmap(self.path, cache_path)
        self.images = np.load(cache_path, mmap_mode="r")  # [480000, 64, 64, 3] uint8
        self.labels = np.load(cache_path.with_name(cache_path.stem + "_labels.npy"))

    def image_at(self, factor_indices: dict[str, int]) -> np.ndarray:
        """Returns float32 [3, 64, 64] in [0, 1]. Prefer :meth:`images_batch` when
        fetching more than a handful of images -- see its docstring for why."""
        row = self.images[flat_index(factor_indices)]
        return np.ascontiguousarray(row.transpose(2, 0, 1).astype(np.float32) / 255.0)

    def images_batch(self, flat_indices: np.ndarray) -> np.ndarray:
        """Fetch many images in one vectorized read. Against the memmap cache this
        is plain numpy fancy indexing (arbitrary order, duplicates allowed); the
        ``np.unique``/``inverse`` dance is kept anyway since it also minimizes
        redundant memory copies when the same combination is requested twice."""
        flat_indices = np.asarray(flat_indices)
        unique_sorted, inverse = np.unique(flat_indices, return_inverse=True)
        unique_images = self.images[unique_sorted]
        images = unique_images[inverse]
        return np.ascontiguousarray(images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)


def _cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _sample_factor_indices(generator: torch.Generator) -> dict[str, int]:
    return {
        name: int(torch.randint(SHAPES3D_FACTOR_SIZES[name], (), generator=generator).item())
        for name in SHAPES3D_FACTOR_NAMES
    }


def sample_shapes3d_static_factor_row(
    split_seed: int, sample_index: int
) -> tuple[torch.Tensor, int]:
    """Reproduce one static-dataset factor row without loading an image."""
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    generator = _cpu_generator(derive_seed(split_seed, "sample", sample_index))
    factors = _sample_factor_indices(generator)
    row = torch.tensor([factors[name] for name in SHAPES3D_FACTOR_NAMES], dtype=torch.long)
    return row, flat_index(factors)


def _step_context_indices(
    indices: dict[str, int], config: Shapes3DDataConfig, generator: torch.Generator
) -> dict[str, int]:
    updated = dict(indices)
    for name in SHAPES3D_CONTEXT_FACTORS:
        if torch.rand((), generator=generator).item() >= config.context_step_probability:
            continue
        step = int(
            torch.randint(
                -config.context_step_max, config.context_step_max + 1, (), generator=generator
            ).item()
        )
        size = SHAPES3D_FACTOR_SIZES[name]
        updated[name] = min(max(updated[name] + step, 0), size - 1)
    return updated


def _generate_factor_trajectory(config: Shapes3DDataConfig, *, seed: int) -> torch.Tensor:
    """Returns factor_sequence [T, len(SHAPES3D_FACTOR_NAMES)] (long), columns in
    ``SHAPES3D_FACTOR_NAMES`` order. Kept as the single source of truth for the
    per-step random draws, so entity/context/image lookups are always derived
    from the *same* materialized sequence rather than re-run against a second
    RNG stream that could silently drift out of sync (e.g. an entity switch
    consumes an extra draw that a naive replay could forget to mirror)."""
    generator = _cpu_generator(seed)
    length = config.trajectory_length
    factor_indices = _sample_factor_indices(generator)
    sequence = torch.empty(length, len(SHAPES3D_FACTOR_NAMES), dtype=torch.long)

    for t in range(length):
        for column, name in enumerate(SHAPES3D_FACTOR_NAMES):
            sequence[t, column] = factor_indices[name]
        if t + 1 < length:
            if torch.rand((), generator=generator).item() < config.entity_switch_probability:
                offset = 1 + int(
                    torch.randint(
                        SHAPES3D_FACTOR_SIZES["shape"] - 1, (), generator=generator
                    ).item()
                )
                factor_indices["shape"] = (
                    factor_indices["shape"] + offset
                ) % SHAPES3D_FACTOR_SIZES["shape"]
            factor_indices = _step_context_indices(factor_indices, config, generator)
    return sequence


_SHAPE_COLUMN = SHAPES3D_FACTOR_NAMES.index("shape")
_CONTEXT_COLUMNS = [SHAPES3D_FACTOR_NAMES.index(name) for name in SHAPES3D_CONTEXT_FACTORS]
_CONTEXT_SIZES = torch.tensor(
    [SHAPES3D_FACTOR_SIZES[name] for name in SHAPES3D_CONTEXT_FACTORS], dtype=torch.float32
)


def _entities_and_contexts(factor_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    entities = factor_sequence[..., _SHAPE_COLUMN]
    raw_context = factor_sequence[..., _CONTEXT_COLUMNS].to(torch.float32)
    contexts = raw_context / (_CONTEXT_SIZES - 1)
    return entities, contexts


def _split_size_and_seed(
    config: Shapes3DDataConfig, split: Split, *, static: bool
) -> tuple[int, int]:
    if split == "train":
        return (
            config.num_train_samples if static else config.num_train_trajectories
        ), config.train_sample_seed
    if split == "validation":
        return (
            config.num_val_samples if static else config.num_val_trajectories
        ), config.validation_sample_seed
    if split == "test":
        return (
            config.num_test_samples if static else config.num_test_trajectories
        ), config.test_sample_seed
    raise ValueError(f"unknown split: {split!r}")


class Shapes3DEntityContextTrajectoryDataset(Dataset):
    """Temporal Shapes3D analogue of ``EntityContextImageTrajectoryDataset``."""

    def __init__(
        self, config: Shapes3DDataConfig, split: Split, *, source: Shapes3DSource | None = None
    ) -> None:
        self.config = config
        self.split = split
        self.source = source if source is not None else Shapes3DSource(config.h5_path)
        size, split_seed = _split_size_and_seed(config, split, static=False)

        entities = torch.empty(size, config.trajectory_length, dtype=torch.long)
        contexts = torch.empty(size, config.trajectory_length, config.context_dim)
        all_factors = torch.empty(
            size, config.trajectory_length, len(SHAPES3D_FACTOR_NAMES), dtype=torch.long
        )
        for index in range(size):
            traj_seed = derive_seed(split_seed, "trajectory", index)
            factor_sequence = _generate_factor_trajectory(config, seed=traj_seed)
            entity_seq, context_seq = _entities_and_contexts(factor_sequence)
            entities[index] = entity_seq
            contexts[index] = context_seq
            all_factors[index] = factor_sequence

        flat_indices = flat_index_batch(all_factors.reshape(-1, len(SHAPES3D_FACTOR_NAMES)).numpy())
        images = self.source.images_batch(flat_indices)  # one bulk read for the whole split
        frames = torch.from_numpy(images).reshape(size, config.trajectory_length, 3, 64, 64)

        self.entities = entities
        self.contexts = contexts
        self.frames = frames
        self.observations = frames.reshape(size, config.trajectory_length, -1)

    def __len__(self) -> int:
        return self.entities.shape[0]

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "frames": self.frames[index],
            "entity": self.entities[index],
            "context": self.contexts[index],
            "trajectory_id": index,
        }


@dataclass(frozen=True, slots=True)
class Shapes3DDatasetSplits:
    train: Shapes3DEntityContextTrajectoryDataset
    validation: Shapes3DEntityContextTrajectoryDataset
    test: Shapes3DEntityContextTrajectoryDataset


def build_shapes3d_dataset_splits(config: Shapes3DDataConfig) -> Shapes3DDatasetSplits:
    source = Shapes3DSource(config.h5_path)
    return Shapes3DDatasetSplits(
        train=Shapes3DEntityContextTrajectoryDataset(config, "train", source=source),
        validation=Shapes3DEntityContextTrajectoryDataset(config, "validation", source=source),
        test=Shapes3DEntityContextTrajectoryDataset(config, "test", source=source),
    )


class Shapes3DStaticImageDataset(Dataset):
    """Independent-sample Shapes3D images (no temporal correlation)."""

    def __init__(
        self, config: Shapes3DDataConfig, split: Split, *, source: Shapes3DSource | None = None
    ) -> None:
        self.config = config
        self.split = split
        self.source = source if source is not None else Shapes3DSource(config.h5_path)
        size, split_seed = _split_size_and_seed(config, split, static=True)

        entities = torch.empty(size, dtype=torch.long)
        contexts = torch.empty(size, config.context_dim)
        all_factors = torch.empty(size, len(SHAPES3D_FACTOR_NAMES), dtype=torch.long)
        flat_indices = np.empty(size, dtype=np.int64)
        for index in range(size):
            factor_row, flat = sample_shapes3d_static_factor_row(split_seed, index)
            entities[index] = factor_row[_SHAPE_COLUMN]
            contexts[index] = torch.tensor(
                [
                    factor_row[SHAPES3D_FACTOR_NAMES.index(name)].item()
                    / (SHAPES3D_FACTOR_SIZES[name] - 1)
                    for name in SHAPES3D_CONTEXT_FACTORS
                ]
            )
            all_factors[index] = factor_row
            flat_indices[index] = flat

        images = torch.from_numpy(self.source.images_batch(flat_indices))

        self.entities = entities
        self.contexts = contexts
        self.images = images
        self.flat_indices = flat_indices
        self.observations = images.reshape(size, -1)

    def __len__(self) -> int:
        return self.entities.shape[0]

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": self.images[index],
            "entity": int(self.entities[index]),
            "context": self.contexts[index],
            "sample_id": index,
        }


@dataclass(frozen=True, slots=True)
class Shapes3DStaticDatasetSplits:
    train: Shapes3DStaticImageDataset
    validation: Shapes3DStaticImageDataset
    test: Shapes3DStaticImageDataset


def build_shapes3d_static_dataset_splits(config: Shapes3DDataConfig) -> Shapes3DStaticDatasetSplits:
    source = Shapes3DSource(config.h5_path)
    return Shapes3DStaticDatasetSplits(
        train=Shapes3DStaticImageDataset(config, "train", source=source),
        validation=Shapes3DStaticImageDataset(config, "validation", source=source),
        test=Shapes3DStaticImageDataset(config, "test", source=source),
    )


@dataclass(frozen=True, slots=True)
class Shapes3DCounterfactualPairs:
    same_entity: torch.Tensor
    same_entity_x1: torch.Tensor
    same_entity_x2: torch.Tensor
    diff_entity_entities: torch.Tensor
    diff_entity_x1: torch.Tensor
    diff_entity_x2: torch.Tensor


@dataclass(frozen=True, slots=True)
class Shapes3DCounterfactualFactorRows:
    same_entity: torch.Tensor
    same_factors_1: torch.Tensor
    same_factors_2: torch.Tensor
    diff_entities: torch.Tensor
    diff_factors_1: torch.Tensor
    diff_factors_2: torch.Tensor

    @property
    def same_flat_indices_1(self) -> np.ndarray:
        return flat_index_batch(self.same_factors_1.numpy())

    @property
    def same_flat_indices_2(self) -> np.ndarray:
        return flat_index_batch(self.same_factors_2.numpy())

    @property
    def diff_flat_indices_1(self) -> np.ndarray:
        return flat_index_batch(self.diff_factors_1.numpy())

    @property
    def diff_flat_indices_2(self) -> np.ndarray:
        return flat_index_batch(self.diff_factors_2.numpy())


def sample_shapes3d_counterfactual_factor_rows(
    config: Shapes3DDataConfig, *, num_pairs: int, seed: int
) -> Shapes3DCounterfactualFactorRows:
    """Generate the exact counterfactual factor rows without image I/O."""
    del config  # Reserved for future factor-policy fields; RNG contract is explicit.
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    generator = _cpu_generator(seed)
    num_shapes = SHAPES3D_FACTOR_SIZES["shape"]

    same_entity = torch.empty(num_pairs, dtype=torch.long)
    same_factors_1 = torch.empty(num_pairs, len(SHAPES3D_FACTOR_NAMES), dtype=torch.long)
    same_factors_2 = torch.empty_like(same_factors_1)
    for i in range(num_pairs):
        shape_idx = int(torch.randint(num_shapes, (), generator=generator).item())
        factors_1 = _sample_factor_indices(generator)
        factors_2 = _sample_factor_indices(generator)
        factors_1["shape"] = shape_idx
        factors_2["shape"] = shape_idx
        same_entity[i] = shape_idx
        same_factors_1[i] = torch.tensor(
            [factors_1[name] for name in SHAPES3D_FACTOR_NAMES], dtype=torch.long
        )
        same_factors_2[i] = torch.tensor(
            [factors_2[name] for name in SHAPES3D_FACTOR_NAMES], dtype=torch.long
        )

    diff_entities = torch.empty(num_pairs, 2, dtype=torch.long)
    diff_factors_1 = torch.empty_like(same_factors_1)
    diff_factors_2 = torch.empty_like(same_factors_1)
    for i in range(num_pairs):
        shape_1 = int(torch.randint(num_shapes, (), generator=generator).item())
        offset = 1 + int(torch.randint(num_shapes - 1, (), generator=generator).item())
        shape_2 = (shape_1 + offset) % num_shapes
        shared_factors = _sample_factor_indices(generator)
        factors_1 = dict(shared_factors, shape=shape_1)
        factors_2 = dict(shared_factors, shape=shape_2)
        diff_entities[i] = torch.tensor([shape_1, shape_2])
        diff_factors_1[i] = torch.tensor(
            [factors_1[name] for name in SHAPES3D_FACTOR_NAMES], dtype=torch.long
        )
        diff_factors_2[i] = torch.tensor(
            [factors_2[name] for name in SHAPES3D_FACTOR_NAMES], dtype=torch.long
        )
    return Shapes3DCounterfactualFactorRows(
        same_entity=same_entity,
        same_factors_1=same_factors_1,
        same_factors_2=same_factors_2,
        diff_entities=diff_entities,
        diff_factors_1=diff_factors_1,
        diff_factors_2=diff_factors_2,
    )


def build_shapes3d_counterfactual_pairs(
    config: Shapes3DDataConfig, source: Shapes3DSource, *, num_pairs: int, seed: int
) -> Shapes3DCounterfactualPairs:
    rows = sample_shapes3d_counterfactual_factor_rows(config, num_pairs=num_pairs, seed=seed)
    same_x1 = torch.from_numpy(source.images_batch(rows.same_flat_indices_1))
    same_x2 = torch.from_numpy(source.images_batch(rows.same_flat_indices_2))
    diff_x1 = torch.from_numpy(source.images_batch(rows.diff_flat_indices_1))
    diff_x2 = torch.from_numpy(source.images_batch(rows.diff_flat_indices_2))

    return Shapes3DCounterfactualPairs(
        same_entity=rows.same_entity,
        same_entity_x1=same_x1,
        same_entity_x2=same_x2,
        diff_entity_entities=rows.diff_entities,
        diff_entity_x1=diff_x1,
        diff_entity_x2=diff_x2,
    )


__all__ = [
    "Shapes3DCounterfactualPairs",
    "Shapes3DCounterfactualFactorRows",
    "Shapes3DDatasetSplits",
    "Shapes3DEntityContextTrajectoryDataset",
    "Shapes3DSource",
    "Shapes3DStaticDatasetSplits",
    "Shapes3DStaticImageDataset",
    "build_shapes3d_counterfactual_pairs",
    "build_shapes3d_dataset_splits",
    "build_shapes3d_static_dataset_splits",
    "flat_index",
    "flat_index_batch",
    "factor_row_from_flat_index",
    "sample_shapes3d_counterfactual_factor_rows",
    "sample_shapes3d_static_factor_row",
]
