from __future__ import annotations

import torch

from jepa.analysis.factorization_v2 import (
    factorization_metrics,
    fit_view_factorization,
    lda_spectrum_metrics,
    principal_subspace_similarity,
    supervised_alignment_metrics,
)


def test_view_factorization_separates_stable_and_perturbed_coordinates() -> None:
    generator = torch.Generator().manual_seed(4)
    stable = torch.randn(5, 512, 3, generator=generator)
    stable[:] = stable.mean(dim=0, keepdim=True)
    unstable = torch.randn(5, 512, 3, generator=generator)
    unstable -= unstable.mean(dim=0, keepdim=True)
    views = torch.cat((stable, unstable), dim=-1)

    result = fit_view_factorization(views)

    assert torch.all(result.eigenvalues[:3] > 0.95)
    assert torch.all(result.eigenvalues[-3:] < 0.1)
    metrics = factorization_metrics(result)
    assert 0.4 < metrics["invariance_mean"] < 0.6


def test_supervised_alignment_finds_stable_class_direction() -> None:
    generator = torch.Generator().manual_seed(8)
    labels = torch.arange(600) % 3
    class_signal = torch.nn.functional.one_hot(labels, 3).float() * 3
    stable = class_signal + 0.05 * torch.randn(600, 3, generator=generator)
    stable_views = stable.unsqueeze(0).repeat(4, 1, 1)
    unstable = torch.randn(4, 600, 3, generator=generator)
    views = torch.cat((stable_views, unstable), dim=-1)
    result = fit_view_factorization(views)

    metrics = supervised_alignment_metrics(views.mean(dim=0), labels, result)

    assert metrics["class_weighted_invariance"] > 0.95
    assert metrics["class_invariant_overlap"] > 0.9
    assert metrics["audit_rank"] == 2


def test_subspace_similarity_is_basis_rotation_invariant() -> None:
    generator = torch.Generator().manual_seed(12)
    views = torch.randn(4, 300, 8, generator=generator)
    result = fit_view_factorization(views)

    assert principal_subspace_similarity(result, result, rank=4) > 0.999


def test_lda_metrics_detect_separated_classes() -> None:
    generator = torch.Generator().manual_seed(19)
    labels = torch.arange(900) % 3
    separated = torch.nn.functional.one_hot(labels, 3).float() * 4
    separated += 0.1 * torch.randn(900, 3, generator=generator)
    noise = torch.randn(900, 5, generator=generator)

    metrics = lda_spectrum_metrics(torch.cat((separated, noise), dim=1), labels)

    assert metrics["between_total_trace_ratio"] > 0.5
    assert metrics["lda_effective_rank"] > 1.8
    assert metrics["num_classes_present"] == 3
