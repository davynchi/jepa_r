from __future__ import annotations

import torch
from torch import nn

from scripts.analysis.benchmark_tiny_imagenet_probes import (
    maybe_l2_normalize,
    topk_accuracy,
    train_frozen_head,
    train_full_finetune,
)


class _ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.flatten(1)).unsqueeze(1)


def test_l2_normalization_and_topk_accuracy() -> None:
    features = maybe_l2_normalize(
        torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
        enabled=True,
    )
    assert torch.allclose(features.norm(dim=-1), torch.ones(2))

    scores = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    labels = torch.tensor([0, 1])
    assert topk_accuracy(scores, labels, k=1) == 0.5
    assert topk_accuracy(scores, labels, k=5) == 1.0


def test_frozen_head_and_full_finetune_smoke() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(24, 4, generator=generator)
    labels = (features[:, 0] > 0).to(torch.long)
    frozen_metrics, frozen_history, frozen_state = train_frozen_head(
        nn.Linear(4, 2),
        train_features=features[:16],
        train_labels=labels[:16],
        validation_features=features[16:20],
        validation_labels=labels[16:20],
        test_features=features[20:],
        test_labels=labels[20:],
        epochs=2,
        batch_size=4,
        learning_rate=1.0e-2,
        weight_decay=0.0,
        device=torch.device("cpu"),
        seed=11,
    )
    assert len(frozen_history) == 2
    assert frozen_state
    assert 0 <= frozen_metrics["top1"] <= 1

    images = torch.randn(24, 1, 1, 2, generator=generator)
    image_labels = (images.flatten(1)[:, 0] > 0).to(torch.long)
    finetune_metrics, finetune_history, finetune_state = train_full_finetune(
        _ToyEncoder(),
        train_images=images[:16],
        train_labels=image_labels[:16],
        validation_images=images[16:20],
        validation_labels=image_labels[16:20],
        test_images=images[20:],
        test_labels=image_labels[20:],
        num_classes=2,
        embed_dim=4,
        epochs=2,
        batch_size=4,
        encoder_lr=1.0e-3,
        head_lr=1.0e-2,
        weight_decay=0.0,
        horizontal_flip_probability=0.0,
        device=torch.device("cpu"),
        amp_dtype=None,
        seed=13,
    )
    assert len(finetune_history) == 2
    assert set(finetune_state) == {"encoder", "head"}
    assert 0 <= finetune_metrics["top5"] <= 1
