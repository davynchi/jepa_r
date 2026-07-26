from __future__ import annotations

import torch

from jepa.moving_mnist import MovingMNISTDataset, MovingMNISTSpec, make_fake_digit_bank


def test_moving_mnist_is_deterministic_and_has_expected_labels(tmp_path) -> None:
    images, labels = make_fake_digit_bank(samples_per_class=2, size=16)
    spec = MovingMNISTSpec(
        num_samples=8,
        total_frames=8,
        context_frames=4,
        canvas_size=32,
        digit_size=16,
        min_speed=1.0,
        max_speed=2.0,
        num_directions=8,
        seed=7,
    )
    dataset = MovingMNISTDataset(
        tmp_path,
        "probe_evaluation",
        spec,
        download=False,
        digit_images=images,
        digit_labels=labels,
    )

    first = dataset[3]
    second = dataset[3]
    assert torch.equal(first["video"], second["video"])
    assert torch.equal(first["positions"], second["positions"])
    assert first["video"].shape == (8, 1, 32, 32)
    assert first["context"].shape == (4, 1, 32, 32)
    assert first["target"].shape == (4, 1, 32, 32)
    assert first["future_positions"].shape == (4, 2)
    assert 0 <= int(first["direction_label"]) < 8
    assert 0 <= int(first["digit_label"]) < 10
    assert int(first["bounce_target"]) in {0, 1}
    assert 1.0 <= float(first["speed"]) <= 2.0
    assert float(first["positions"].min()) >= 8.0
    assert float(first["positions"].max()) <= 24.0


def test_different_indices_generate_different_videos(tmp_path) -> None:
    images, labels = make_fake_digit_bank(samples_per_class=1, size=16)
    spec = MovingMNISTSpec(
        num_samples=4,
        total_frames=8,
        context_frames=4,
        canvas_size=32,
        digit_size=16,
        seed=13,
    )
    dataset = MovingMNISTDataset(
        tmp_path,
        "pretrain_train",
        spec,
        download=False,
        digit_images=images,
        digit_labels=labels,
    )
    assert not torch.equal(dataset[0]["video"], dataset[1]["video"])
