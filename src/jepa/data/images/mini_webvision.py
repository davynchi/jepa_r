"""Mini-WebVision static image splits for spatial I-JEPA experiments."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Split = str


@dataclass(frozen=True, slots=True)
class MiniWebVisionDataConfig:
    root: str = "data/mini-webvision"
    image_size: int = 64
    num_train_samples: int = 65944
    num_val_samples: int = 1000
    num_test_samples: int = 1000
    train_sample_seed: int = 1000
    clean_split_seed: int = 2000
    file_backed: bool = False

    @property
    def num_entities(self) -> int:
        return 50

    @property
    def context_dim(self) -> int:
        return 0

    @property
    def observation_dim(self) -> int:
        return 3 * self.image_size * self.image_size


def _read_filelist(path: Path) -> list[tuple[Path, int]]:
    if not path.exists():
        raise FileNotFoundError(f"missing Mini-WebVision filelist: {path}")
    items: list[tuple[Path, int]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.rsplit(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"invalid filelist row at {path}:{line_number}: {line!r}")
        relative_path, label_text = fields
        label = int(label_text)
        if not 0 <= label < 50:
            raise ValueError(f"expected labels in [0, 49], got {label} at {path}:{line_number}")
        items.append((Path(relative_path), label))
    if not items:
        raise ValueError(f"empty Mini-WebVision filelist: {path}")
    return items


def _load_synsets(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    path = root / "info" / "synsets.txt"
    if not path.exists():
        labels = tuple(str(index) for index in range(50))
        return labels, labels
    rows = [
        line.strip().split(maxsplit=1)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) < 50:
        raise ValueError(f"expected at least 50 synsets in {path}, found {len(rows)}")
    rows = rows[:50]
    return (
        tuple(fields[0] for fields in rows),
        tuple(fields[1] if len(fields) > 1 else fields[0] for fields in rows),
    )


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


def _select_items(
    config: MiniWebVisionDataConfig,
    root: Path,
    split: Split,
) -> list[tuple[Path, int]]:
    if split == "train":
        items = _read_filelist(root / "train_filelist.txt")
        indices = _sample_indices(
            len(items),
            config.num_train_samples,
            seed=config.train_sample_seed,
        )
    elif split in {"validation", "test"}:
        items = _read_filelist(root / "val_filelist.txt")
        requested = config.num_val_samples + config.num_test_samples
        if requested > len(items):
            raise ValueError(
                "Mini-WebVision clean validation and test splits must be disjoint: "
                f"requested {requested} samples from {len(items)} available"
            )
        generator = torch.Generator().manual_seed(config.clean_split_seed)
        order = torch.randperm(len(items), generator=generator)
        if split == "validation":
            indices = order[: config.num_val_samples]
        else:
            start = config.num_val_samples
            indices = order[start : start + config.num_test_samples]
    else:
        raise ValueError(f"unknown split: {split!r}")
    return [(root / items[int(index)][0], items[int(index)][1]) for index in indices]


def _read_image(path: Path, *, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32) / 255.0


def _read_eval_image(path: Path, *, image_size: int) -> torch.Tensor:
    """Apply the deterministic resize/center-crop used for evaluation and scoring."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        resize_short = round(image_size / 0.875)
        width, height = image.size
        scale = resize_short / min(width, height)
        resized = (round(width * scale), round(height * scale))
        image = image.resize(resized, Image.Resampling.BICUBIC)
        left = max((image.width - image_size) // 2, 0)
        top = max((image.height - image_size) // 2, 0)
        image = image.crop((left, top, left + image_size, top + image_size))
        array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return tensor.sub_(mean).div_(std)


class FileBackedImageCollection:
    """Tensor-like, lazily decoded normalized image collection."""

    def __init__(self, paths: list[Path], *, image_size: int) -> None:
        self.paths = paths
        self.image_size = image_size
        self.shape = torch.Size((len(paths), 3, image_size, image_size))
        self.device = torch.device("cpu")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int | slice | torch.Tensor) -> torch.Tensor:
        if isinstance(index, slice):
            selected = list(range(len(self.paths)))[index]
        elif isinstance(index, torch.Tensor):
            selected = index.detach().cpu().reshape(-1).tolist()
        else:
            selected = [index]
        images = torch.stack(
            [
                _read_eval_image(self.paths[int(item)], image_size=self.image_size)
                for item in selected
            ]
        )
        return images[0] if isinstance(index, int) else images

    def split(self, batch_size: int) -> Iterator[torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            yield self[torch.arange(start, min(start + batch_size, len(self)))]


def _cache_path(root: Path, config: MiniWebVisionDataConfig, split: Split) -> Path:
    sample_count = {
        "train": config.num_train_samples,
        "validation": config.num_val_samples,
        "test": config.num_test_samples,
    }[split]
    seed = config.train_sample_seed if split == "train" else config.clean_split_seed
    return root / ".jepa_cache" / (
        f"{split}_size{config.image_size}_n{sample_count}_seed{seed}.pt"
    )


class MiniWebVisionStaticImageDataset(Dataset):
    """Image split backed by the standard Mini-WebVision filelists."""

    def __init__(self, config: MiniWebVisionDataConfig, split: Split) -> None:
        self.config = config
        self.split = split
        root = Path(config.root)
        if not root.exists():
            raise FileNotFoundError(f"Mini-WebVision root not found at {root}")
        selected = _select_items(config, root, split)
        missing = [path for path, _ in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} Mini-WebVision images are missing; "
                f"first missing path: {missing[0]}"
            )

        self.paths = [path for path, _ in selected]
        self.wnids, self.class_names = _load_synsets(root)
        self.contexts = torch.empty(len(selected), 0)
        self.images: torch.Tensor | FileBackedImageCollection
        self.observations: torch.Tensor | None
        if config.file_backed:
            self.entities = torch.tensor([label for _, label in selected], dtype=torch.long)
            self.images = FileBackedImageCollection(self.paths, image_size=config.image_size)
            self.observations = None
        else:
            cache_path = _cache_path(root, config, split)
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                self.entities = cached["entities"].to(torch.long)
                self.images = cached["images"].to(torch.float32)
            else:
                self.entities = torch.tensor([label for _, label in selected], dtype=torch.long)
                self.images = torch.stack(
                    [_read_image(path, image_size=config.image_size) for path in self.paths]
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
                torch.save({"entities": self.entities, "images": self.images}, tmp_path)
                tmp_path.replace(cache_path)
            assert isinstance(self.images, torch.Tensor)
            self.observations = self.images.reshape(len(selected), -1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": self.images[index],
            "entity": int(self.entities[index]),
            "context": self.contexts[index],
            "path": str(self.paths[index]),
            "sample_id": index,
        }


@dataclass(frozen=True, slots=True)
class MiniWebVisionStaticDatasetSplits:
    train: MiniWebVisionStaticImageDataset
    validation: MiniWebVisionStaticImageDataset
    test: MiniWebVisionStaticImageDataset


def build_mini_webvision_static_dataset_splits(
    config: MiniWebVisionDataConfig,
) -> MiniWebVisionStaticDatasetSplits:
    return MiniWebVisionStaticDatasetSplits(
        train=MiniWebVisionStaticImageDataset(config, "train"),
        validation=MiniWebVisionStaticImageDataset(config, "validation"),
        test=MiniWebVisionStaticImageDataset(config, "test"),
    )


__all__ = [
    "FileBackedImageCollection",
    "MiniWebVisionDataConfig",
    "MiniWebVisionStaticDatasetSplits",
    "MiniWebVisionStaticImageDataset",
    "build_mini_webvision_static_dataset_splits",
]
