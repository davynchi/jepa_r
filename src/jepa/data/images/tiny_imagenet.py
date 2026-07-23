"""Tiny ImageNet static image splits for spatial I-JEPA experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from jepa.configs.base import derive_seed

Split = str


@dataclass(frozen=True, slots=True)
class TinyImageNetDataConfig:
    root: str = "data/tiny-imagenet-200"
    num_train_samples: int = 16000
    num_val_samples: int = 1000
    num_test_samples: int = 1000
    train_sample_seed: int = 1000
    validation_sample_seed: int = 2000
    test_sample_seed: int = 3000

    @property
    def num_entities(self) -> int:
        return 200

    @property
    def context_dim(self) -> int:
        return 0

    @property
    def observation_dim(self) -> int:
        return 3 * 64 * 64


def _split_size_and_seed(config: TinyImageNetDataConfig, split: Split) -> tuple[int, int]:
    if split == "train":
        return config.num_train_samples, config.train_sample_seed
    if split == "validation":
        return config.num_val_samples, config.validation_sample_seed
    if split == "test":
        return config.num_test_samples, config.test_sample_seed
    raise ValueError(f"unknown split: {split!r}")


def _load_wnids(root: Path) -> list[str]:
    wnids_path = root / "wnids.txt"
    if wnids_path.exists():
        return [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]
    train_root = root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"missing Tiny ImageNet train directory: {train_root}")
    return sorted(path.name for path in train_root.iterdir() if path.is_dir())


def _load_class_names(root: Path, wnids: list[str]) -> dict[str, str]:
    names = {wnid: wnid for wnid in wnids}
    words_path = root / "words.txt"
    if not words_path.exists():
        return names
    wanted = set(wnids)
    for line in words_path.read_text().splitlines():
        fields = line.split("\t", maxsplit=1)
        if len(fields) != 2:
            continue
        wnid, name = fields
        if wnid in wanted:
            names[wnid] = name
    return names


def _read_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (64, 64):
            image = image.resize((64, 64), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32) / 255.0


def _sample_indices(count: int, sample_count: int, *, seed: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("cannot sample from an empty split")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    if sample_count <= count:
        return order[:sample_count]
    extra = torch.randint(count, (sample_count - count,), generator=generator)
    return torch.cat([order, extra])


def _selected_cache_path(root: Path, split: Split, *, sample_count: int, seed: int) -> Path:
    cache_dir = root / ".jepa_cache"
    return cache_dir / f"{split}_n{sample_count}_seed{seed}.pt"


def _train_items(root: Path, class_to_index: dict[str, int]) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for wnid, class_index in class_to_index.items():
        image_dir = root / "train" / wnid / "images"
        if not image_dir.exists():
            continue
        for path in sorted(image_dir.glob("*.JPEG")):
            items.append((path, class_index))
    if not items:
        raise FileNotFoundError(f"no Tiny ImageNet train images found under {root / 'train'}")
    return items


def _validation_items(root: Path, class_to_index: dict[str, int]) -> list[tuple[Path, int]]:
    annotations_path = root / "val" / "val_annotations.txt"
    image_dir = root / "val" / "images"
    if not annotations_path.exists() or not image_dir.exists():
        raise FileNotFoundError(f"missing Tiny ImageNet validation split under {root / 'val'}")
    items: list[tuple[Path, int]] = []
    for line in annotations_path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        filename, wnid = fields[0], fields[1]
        if wnid not in class_to_index:
            continue
        items.append((image_dir / filename, class_to_index[wnid]))
    if not items:
        raise FileNotFoundError(f"no Tiny ImageNet validation images found under {image_dir}")
    return items


class TinyImageNetStaticImageDataset(Dataset):
    """Eager static-image split for the official Tiny ImageNet directory layout."""

    def __init__(self, config: TinyImageNetDataConfig, split: Split) -> None:
        self.config = config
        self.split = split
        root = Path(config.root)
        if not root.exists():
            raise FileNotFoundError(
                f"Tiny ImageNet root not found at {root}. Expected tiny-imagenet-200/"
            )
        wnids = _load_wnids(root)
        class_names = _load_class_names(root, wnids)
        class_to_index = {wnid: index for index, wnid in enumerate(wnids)}
        items = (
            _train_items(root, class_to_index)
            if split == "train"
            else _validation_items(root, class_to_index)
        )
        sample_count, seed = _split_size_and_seed(config, split)
        if split == "test":
            seed = derive_seed(seed, "test-from-validation")
        indices = _sample_indices(len(items), sample_count, seed=seed)
        selected = [items[int(index)] for index in indices]

        self.paths = [path for path, _ in selected]
        self.wnids = tuple(wnids)
        self.class_names = tuple(class_names[wnid] for wnid in wnids)
        self.contexts = torch.empty(len(selected), 0)
        cache_path = _selected_cache_path(root, split, sample_count=sample_count, seed=seed)
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.entities = cached["entities"].to(torch.long)
            self.images = cached["images"].to(torch.float32)
        else:
            self.entities = torch.tensor([label for _, label in selected], dtype=torch.long)
            self.images = torch.stack([_read_image(path) for path in self.paths])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
            torch.save({"entities": self.entities, "images": self.images}, tmp_path)
            tmp_path.replace(cache_path)
        self.observations = self.images.reshape(len(selected), -1)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": self.images[index],
            "entity": int(self.entities[index]),
            "context": self.contexts[index],
            "path": str(self.paths[index]),
            "sample_id": index,
        }


@dataclass(frozen=True, slots=True)
class TinyImageNetStaticDatasetSplits:
    train: TinyImageNetStaticImageDataset
    validation: TinyImageNetStaticImageDataset
    test: TinyImageNetStaticImageDataset


def build_tiny_imagenet_static_dataset_splits(
    config: TinyImageNetDataConfig,
) -> TinyImageNetStaticDatasetSplits:
    return TinyImageNetStaticDatasetSplits(
        train=TinyImageNetStaticImageDataset(config, "train"),
        validation=TinyImageNetStaticImageDataset(config, "validation"),
        test=TinyImageNetStaticImageDataset(config, "test"),
    )


__all__ = [
    "TinyImageNetDataConfig",
    "TinyImageNetStaticDatasetSplits",
    "TinyImageNetStaticImageDataset",
    "build_tiny_imagenet_static_dataset_splits",
]
