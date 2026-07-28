from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

from jepa.training.images.barlow_twins import (
    BarlowTwinsProjector,
    OfficialBarlowTransform,
    off_diagonal,
)


def test_off_diagonal_matches_explicit_selection() -> None:
    matrix = torch.arange(16).reshape(4, 4)
    expected = matrix[~torch.eye(4, dtype=torch.bool)]

    assert torch.equal(off_diagonal(matrix), expected)


def test_barlow_projector_is_finite_and_differentiable() -> None:
    module = BarlowTwinsProjector(8, (16, 16), redundancy_weight=0.0051)
    first = torch.randn(8, 8, requires_grad=True)
    second = torch.randn(8, 8, requires_grad=True)

    loss, metadata = module(first, second)
    loss.backward()

    assert torch.isfinite(loss)
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()
    assert metadata["official_barlow/off_diagonal"] >= 0


def test_official_transform_produces_two_normalized_views() -> None:
    random.seed(7)
    pixels = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    first, second = OfficialBarlowTransform(32)(Image.fromarray(pixels))

    assert first.shape == (3, 32, 32)
    assert second.shape == (3, 32, 32)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert not torch.equal(first, second)
