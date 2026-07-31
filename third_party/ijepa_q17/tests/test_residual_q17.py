from __future__ import annotations

import torch

from src.residual_q17 import (
    ResidualQ17Regularizer,
    apply_fixed_transform,
    gradient_alignment_and_weight,
)


def _linear_residual_problem(
    *,
    samples: int = 128,
    dimensions: int = 8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(samples, dimensions, generator=generator)
    operator = torch.randn(dimensions, dimensions, generator=generator) / dimensions**0.5
    offset = torch.linspace(-2.0, 2.0, dimensions)
    residual = source @ operator.T + offset
    return source, {"flip": residual}


def test_centering_removes_constant_transformation_shift() -> None:
    source, residuals = _linear_residual_problem()
    shifted = {"flip": residuals["flip"] + 11.0}
    first = ResidualQ17Regularizer(8, ("flip",), statistics_decay=0.0)
    second = ResidualQ17Regularizer(8, ("flip",), statistics_decay=0.0)

    first.update_statistics(source, residuals)
    second.update_statistics(source, shifted)
    first_batch = first.center_batch(source, residuals)
    second_batch = second.center_batch(source, shifted)

    assert torch.allclose(
        first_batch.residuals["flip"],
        second_batch.residuals["flip"],
        atol=2.0e-6,
    )


def test_full_operator_generalizes_and_shuffling_destroys_pairing() -> None:
    source, residuals = _linear_residual_problem()
    regularizer = ResidualQ17Regularizer(8, ("flip",), statistics_decay=0.0)
    regularizer.update_statistics(source, residuals)
    batch = regularizer.center_batch(source, residuals)
    optimizer = torch.optim.AdamW(
        regularizer.operators.parameters(),
        lr=0.05,
        weight_decay=0.0,
    )
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = regularizer.operator_loss(batch, mode="matched")
        loss.backward()
        optimizer.step()

    matched_loss, matched = regularizer.encoder_loss(batch, mode="matched")
    shuffled_loss, shuffled = regularizer.encoder_loss(batch, mode="shuffled")

    assert matched_loss.item() < 1.0e-4
    assert matched["residual_q17/flip/gain"] > 0.999
    assert shuffled_loss.item() > matched_loss.item() * 1000
    assert shuffled["residual_q17/flip/gain"] < 0.5


def test_operator_and_encoder_steps_have_separate_gradient_paths() -> None:
    source, residuals = _linear_residual_problem(samples=16, dimensions=4)
    source.requires_grad_(True)
    regularizer = ResidualQ17Regularizer(4, ("flip",), statistics_decay=0.0)
    regularizer.update_statistics(source, residuals)
    batch = regularizer.center_batch(source, residuals)

    operator_loss = regularizer.operator_loss(batch, mode="matched")
    operator_loss.backward()
    assert source.grad is None
    assert regularizer.operators["flip"].weight.grad is not None

    source.grad = None
    regularizer.operators["flip"].weight.grad = None
    encoder_loss, _ = regularizer.encoder_loss(batch, mode="matched")
    encoder_loss.backward()
    assert source.grad is not None
    assert regularizer.operators["flip"].weight.grad is None


def test_gradient_weight_matches_requested_ratio_and_cosine() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    jepa_loss = parameter.square().sum()
    auxiliary_loss = 3.0 * parameter.square().sum()

    alignment = gradient_alignment_and_weight(
        jepa_loss,
        auxiliary_loss,
        (parameter,),
        target_ratio=0.05,
    )

    assert abs(alignment.cosine - 1.0) < 1.0e-7
    assert abs(alignment.actual_ratio - 0.05) < 1.0e-7
    assert abs(alignment.loss_weight - 0.05 / 3.0) < 1.0e-7


def test_fixed_transforms_are_deterministic_and_shape_preserving() -> None:
    images = torch.randn(4, 3, 16, 16, generator=torch.Generator().manual_seed(9))
    outputs = {}
    for transform in ("flip", "blur", "color"):
        first = apply_fixed_transform(
            images,
            transform,
            blur_sigma=1.0,
            color_strength=0.2,
        )
        second = apply_fixed_transform(
            images,
            transform,
            blur_sigma=1.0,
            color_strength=0.2,
        )
        assert first.shape == images.shape
        assert torch.equal(first, second)
        outputs[transform] = first
    assert not torch.equal(outputs["flip"], outputs["blur"])


def test_state_dict_restores_statistics_and_operators() -> None:
    source, residuals = _linear_residual_problem(samples=16, dimensions=4)
    regularizer = ResidualQ17Regularizer(4, ("flip",), statistics_decay=0.9)
    regularizer.update_statistics(source, residuals)
    regularizer.operators["flip"].weight.data.fill_(0.25)

    restored = ResidualQ17Regularizer(4, ("flip",), statistics_decay=0.9)
    restored.load_state_dict(regularizer.state_dict())

    assert torch.equal(restored.source_mean, regularizer.source_mean)
    assert torch.equal(restored.residual_means, regularizer.residual_means)
    assert torch.equal(restored.residual_energies, regularizer.residual_energies)
    assert torch.equal(
        restored.operators["flip"].weight,
        regularizer.operators["flip"].weight,
    )
