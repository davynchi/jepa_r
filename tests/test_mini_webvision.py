from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jepa.data.images.mini_webvision import (
    MiniWebVisionDataConfig,
    build_mini_webvision_static_dataset_splits,
)
from jepa.training.images.ijepa_spatial import MaskConfig
from jepa.training.images.mask_loader import FileImageMaskLoader


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 10, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_builds_disjoint_filelist_splits(tmp_path: Path) -> None:
    train_rows = []
    val_rows = []
    for label in range(2):
        for index in range(3):
            relative = f"google/q{label:04d}/train_{label}_{index}.jpg"
            _write_image(tmp_path / relative, 20 + label * 40 + index)
            train_rows.append(f"{relative} {label}")
        for index in range(4):
            relative = f"val/val_{label}_{index}.jpg"
            _write_image(tmp_path / relative, 100 + label * 40 + index)
            val_rows.append(f"{relative} {label}")

    (tmp_path / "train_filelist.txt").write_text("\n".join(train_rows))
    (tmp_path / "val_filelist.txt").write_text("\n".join(val_rows))
    info = tmp_path / "info"
    info.mkdir()
    (info / "synsets.txt").write_text(
        "\n".join(f"n{index:08d} class {index}" for index in range(50))
    )

    config = MiniWebVisionDataConfig(
        root=str(tmp_path),
        image_size=8,
        num_train_samples=6,
        num_val_samples=3,
        num_test_samples=3,
    )
    datasets = build_mini_webvision_static_dataset_splits(config)

    assert datasets.train.images.shape == (6, 3, 8, 8)
    assert datasets.validation.images.shape == (3, 3, 8, 8)
    assert datasets.test.images.shape == (3, 3, 8, 8)
    assert set(datasets.validation.paths).isdisjoint(datasets.test.paths)
    assert len(datasets.train.wnids) == 50


def test_file_backed_split_and_upstream_loader(tmp_path: Path) -> None:
    train_rows = []
    val_rows = []
    for index in range(6):
        train_relative = f"google/q0000/train_{index}.jpg"
        val_relative = f"val/val_{index}.jpg"
        _write_image(tmp_path / train_relative, 20 + index)
        _write_image(tmp_path / val_relative, 80 + index)
        train_rows.append(f"{train_relative} 0")
        val_rows.append(f"{val_relative} 0")
    (tmp_path / "train_filelist.txt").write_text("\n".join(train_rows))
    (tmp_path / "val_filelist.txt").write_text("\n".join(val_rows))

    config = MiniWebVisionDataConfig(
        root=str(tmp_path),
        image_size=8,
        num_train_samples=6,
        num_val_samples=2,
        num_test_samples=2,
        file_backed=True,
    )
    datasets = build_mini_webvision_static_dataset_splits(config)
    assert datasets.train.images.shape == (6, 3, 8, 8)
    assert datasets.train.observations is None
    assert datasets.train.images[torch.tensor([0, 2])].shape == (2, 3, 8, 8)

    order = torch.tensor([5, 1, 1, 3])
    loader = FileImageMaskLoader(
        datasets.train.paths,
        num_draws=4,
        batch_size=2,
        image_size=8,
        patch_size=1,
        mask_config=MaskConfig(min_keep=1),
        crop_scale=(0.3, 1.0),
        horizontal_flip_probability=0.5,
        num_workers=0,
        pin_memory=False,
    )
    batches = list(loader.iter_epoch(order, seed=123))
    assert torch.cat([batch.indices for batch in batches]).tolist() == order.tolist()
    assert all(batch.images is not None and batch.images.shape == (2, 3, 8, 8) for batch in batches)
