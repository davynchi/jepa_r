from __future__ import annotations

import torch

from jepa.analysis.joint_transform_factorization import (
    apply_whitening,
    fit_joint_block_diagonalization,
    fit_linear_operator,
    fit_whitening_projection,
    remove_isotropic_component,
)


def test_whitening_and_linear_operator_generalize() -> None:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(500, 12, generator=generator)
    whitening = fit_whitening_projection(features[:350], dimension=8)
    source = apply_whitening(features, whitening)
    operator = torch.randn(8, 8, generator=generator, dtype=source.dtype) / 4
    target = source @ operator

    fit = fit_linear_operator(
        source[:350],
        target[:350],
        source[350:],
        target[350:],
        ridge=1e-6,
    )

    assert whitening.retained_variance_fraction > 0.6
    assert fit.r_squared > 0.999


def test_joint_block_diagonalization_recovers_shared_blocks() -> None:
    generator = torch.Generator().manual_seed(7)
    dimension = 8
    hidden_basis, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    operators = []
    for _ in range(4):
        blocks = torch.block_diag(
            torch.randn(4, 4, generator=generator, dtype=torch.float64),
            torch.randn(4, 4, generator=generator, dtype=torch.float64),
        )
        operators.append(hidden_basis @ blocks @ hidden_basis.T)
    train = torch.stack(operators)
    validation = train + 0.002 * torch.randn(
        train.shape,
        generator=generator,
        dtype=train.dtype,
    )

    fit = fit_joint_block_diagonalization(
        train,
        validation,
        num_blocks=2,
        restarts=3,
        steps=500,
        learning_rate=0.03,
        seed=11,
    )

    assert fit.validation_factorization > 0.97
    assert fit.validation_factorization > fit.random_validation_factorization + 0.2


def test_joint_block_diagonalization_does_not_invent_perfect_structure() -> None:
    generator = torch.Generator().manual_seed(17)
    operators = torch.randn(5, 8, 8, generator=generator, dtype=torch.float64)
    validation = torch.randn(5, 8, 8, generator=generator, dtype=torch.float64)

    fit = fit_joint_block_diagonalization(
        operators,
        validation,
        num_blocks=2,
        restarts=2,
        steps=200,
        learning_rate=0.03,
        seed=19,
    )

    assert fit.validation_factorization < 0.8


def test_isotropic_component_is_removed_exactly() -> None:
    operators = torch.stack((2 * torch.eye(6), -3 * torch.eye(6)))

    residuals = remove_isotropic_component(operators)

    assert torch.allclose(residuals, torch.zeros_like(residuals), atol=1e-7)
