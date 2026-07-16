from __future__ import annotations

import numpy as np
import torch

from jepa.metrics import compute_representation_metrics
from jepa.temporal_analysis import (
    classifier_accuracy,
    compute_scatter_matrices,
    context_r2_mean,
    counterfactual_invariance,
    entity_subspace_projection,
    fit_context_regressor,
    fit_entity_classifier,
    project,
    solve_generalized_eigenproblem,
)


def _synthetic_latents(num_entities=3, per_entity=40, dim=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    entity = torch.arange(num_entities).repeat_interleave(per_entity)
    centers = torch.randn(num_entities, dim, generator=generator) * 5.0
    noise = torch.randn(entity.shape[0], dim, generator=generator) * 0.1
    latents = centers[entity] + noise
    return latents, entity


def test_generalized_eigensolver_handles_singular_within_class_covariance() -> None:
    # Only one sample per entity: within-class covariance is exactly singular.
    latents = torch.randn(3, 4)
    entity = torch.arange(3)
    scatter = compute_scatter_matrices(latents, entity, num_entities=3)
    assert np.allclose(scatter.within, 0.0)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=1e-6)
    assert np.isfinite(eigen.eigenvalues).all()
    assert np.isfinite(eigen.eigenvectors).all()


def test_entity_subspace_projection_is_a_projector() -> None:
    latents, entity = _synthetic_latents()
    scatter = compute_scatter_matrices(latents, entity, num_entities=3)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=1e-6)
    projection = entity_subspace_projection(eigen.eigenvectors, 2)
    assert np.allclose(projection @ projection, projection, atol=1e-6)
    assert np.allclose(projection, projection.T, atol=1e-6)


def test_entity_classifier_and_context_regressor_recover_signal() -> None:
    latents, entity = _synthetic_latents()
    scatter = compute_scatter_matrices(latents, entity, num_entities=3)
    eigen = solve_generalized_eigenproblem(scatter, epsilon=1e-6)
    projection = entity_subspace_projection(eigen.eigenvectors, 2)
    projected = project(latents, projection)
    classifier = fit_entity_classifier(projected, entity, num_classes=3, ridge=1e-6)
    accuracy = classifier_accuracy(classifier, projected, entity)
    assert accuracy > 0.9

    context = torch.randn(latents.shape[0], 2)
    probe = fit_context_regressor(latents, context, ridge=1e-6)
    r2 = context_r2_mean(probe, latents, context)
    assert r2.value is not None


def test_counterfactual_invariance_orders_same_vs_diff_entity() -> None:
    same_1 = torch.zeros(20, 2)
    same_2 = torch.zeros(20, 2) + 0.01
    diff_1 = torch.zeros(20, 2)
    diff_2 = torch.ones(20, 2) * 5.0
    result = counterfactual_invariance(same_1, same_2, diff_1, diff_2)
    assert result.d_same < result.d_diff
    assert result.q_entity > 1.0


def test_evaluation_metrics_remain_finite_for_nearly_collapsed_representations() -> None:
    representations = torch.zeros(50, 4) + torch.randn(50, 4) * 1e-9
    metrics = compute_representation_metrics(representations)
    assert metrics.collapsed
    if metrics.effective_rank.value is not None:
        assert np.isfinite(metrics.effective_rank.value)
    if metrics.latent_std_mean.value is not None:
        assert np.isfinite(metrics.latent_std_mean.value)
