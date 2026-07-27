from __future__ import annotations

import torch

from scripts.images.train_ijepa_spatial import _select_probe_subset, _topk_accuracy


def test_select_probe_subset_is_seeded_and_keeps_labels_aligned() -> None:
    images = torch.arange(30).reshape(10, 3)
    labels = torch.arange(10)

    first_images, first_labels = _select_probe_subset(
        images,
        labels,
        size=4,
        seed=17,
    )
    second_images, second_labels = _select_probe_subset(
        images,
        labels,
        size=4,
        seed=17,
    )

    assert torch.equal(first_images, second_images)
    assert torch.equal(first_labels, second_labels)
    assert torch.equal(first_images[:, 0] // 3, first_labels)


def test_topk_accuracy_handles_fewer_than_five_classes() -> None:
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.1, 0.2],
        ]
    )
    labels = torch.tensor([0, 1])

    assert _topk_accuracy(scores, labels, k=1) == 0.5
    assert _topk_accuracy(scores, labels, k=5) == 1.0
