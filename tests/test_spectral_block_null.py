from __future__ import annotations

import torch

from jepa.analysis.spectral_block_null import (
    fit_variance_whitening_projection,
    haar_orthogonal,
    independently_rotate_operators,
    optimal_projector_agreement,
    optimized_factorizations_batched,
    spectral_null_test,
)


def test_variance_whitening_uses_smallest_admissible_multiple() -> None:
    generator = torch.Generator().manual_seed(3)
    scales = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.2, 0.1, 0.05, 0.01])
    features = torch.randn(2000, 8, generator=generator) * scales

    whitening = fit_variance_whitening_projection(
        features,
        minimum_variance_fraction=0.95,
        dimension_multiple=2,
    )

    assert whitening.projection.shape == (8, 4)
    assert whitening.retained_variance_fraction >= 0.95


def test_haar_matrix_is_orthogonal() -> None:
    basis = haar_orthogonal(
        16,
        generator=torch.Generator().manual_seed(5),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert torch.allclose(basis.T @ basis, torch.eye(16, dtype=torch.float64), atol=1e-10)


def test_independent_rotation_preserves_each_operator_spectrum_and_norm() -> None:
    generator = torch.Generator().manual_seed(7)
    operators = torch.randn(3, 8, 8, generator=generator, dtype=torch.float64)

    rotated = independently_rotate_operators(operators, seed=11)

    for original, null in zip(operators, rotated, strict=True):
        assert torch.allclose(original.norm(), null.norm(), atol=1e-10)
        assert torch.allclose(
            torch.linalg.svdvals(original),
            torch.linalg.svdvals(null),
            atol=1e-10,
        )
        original_power = torch.eye(original.shape[0], dtype=original.dtype)
        null_power = torch.eye(null.shape[0], dtype=null.dtype)
        for _ in range(original.shape[0]):
            original_power = original_power @ original
            null_power = null_power @ null
            assert torch.allclose(
                original_power.trace(),
                null_power.trace(),
                atol=1e-7,
                rtol=1e-9,
            )


def test_spectral_null_detects_shared_hidden_blocks() -> None:
    generator = torch.Generator().manual_seed(13)
    basis, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator, dtype=torch.float64))
    operators = []
    for _ in range(3):
        blocks = torch.block_diag(
            torch.randn(4, 4, generator=generator, dtype=torch.float64),
            torch.randn(4, 4, generator=generator, dtype=torch.float64),
        )
        operators.append(basis @ blocks @ basis.T)

    result = spectral_null_test(
        torch.stack(operators).float(),
        num_blocks=2,
        null_samples=12,
        restarts=2,
        steps=250,
        learning_rate=0.03,
        seed=17,
    )

    assert result.real_factorization > 0.98
    assert result.delta > 0.1
    assert result.z_score > 2.0


def test_batched_optimizer_matches_independent_optimizer_quality() -> None:
    generator = torch.Generator().manual_seed(23)
    operator_sets = torch.randn(3, 2, 4, 4, generator=generator)

    together = optimized_factorizations_batched(
        operator_sets,
        num_blocks=2,
        restarts=2,
        steps=40,
        learning_rate=0.03,
        seeds=[31, 32, 33],
        batch_size=3,
    )
    separately = optimized_factorizations_batched(
        operator_sets,
        num_blocks=2,
        restarts=2,
        steps=40,
        learning_rate=0.03,
        seeds=[31, 32, 33],
        batch_size=1,
    )

    assert torch.allclose(torch.tensor(together), torch.tensor(separately), atol=1e-6)


def test_projector_agreement_ignores_block_permutations_and_internal_rotations() -> None:
    first = torch.eye(8, dtype=torch.float64)
    generator = torch.Generator().manual_seed(41)
    rotations = []
    for _ in range(4):
        block, _ = torch.linalg.qr(torch.randn(2, 2, generator=generator, dtype=torch.float64))
        rotations.append(block)
    internal = torch.block_diag(*rotations)
    permutation = torch.tensor([4, 5, 0, 1, 6, 7, 2, 3])
    second = (first @ internal)[:, permutation]

    agreement = optimal_projector_agreement(first, second, num_blocks=4)

    assert abs(agreement.mean_overlap - 1.0) < 1e-10
    assert abs(agreement.minimum_overlap - 1.0) < 1e-10
    assert sorted(agreement.assignment) == [0, 1, 2, 3]
