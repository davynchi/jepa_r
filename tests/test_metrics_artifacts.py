from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from jepa.config import DataConfig, EvaluationConfig, ExperimentConfig, validate_config
from jepa.data import LatentDynamicsDataset
from jepa.metrics import (
    MetricValue,
    WeightedMean,
    compute_representation_metrics,
    fit_ridge_probe,
    global_gradient_norm,
    ridge_r2_score,
)
from jepa.models import LinearEncoder


def test_metric_value_requires_exactly_one_value_or_reason() -> None:
    with pytest.raises(ValueError, match="either a value"):
        MetricValue(None)
    with pytest.raises(ValueError, match="either a value"):
        MetricValue(1.0, "unexpected")
    with pytest.raises(ValueError, match="finite"):
        MetricValue(float("inf"))


def test_isotropic_representations_have_full_effective_rank() -> None:
    representations = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    metrics = compute_representation_metrics(representations)

    assert metrics.collapsed is False
    assert metrics.collapse_reasons == ()
    assert metrics.effective_rank.value == pytest.approx(2.0)
    assert metrics.effective_rank_normalized.value == pytest.approx(1.0)
    assert metrics.top_eigen_fraction.value == pytest.approx(0.5)
    assert metrics.latent_std_mean.value == pytest.approx((2.0 / 3.0) ** 0.5)
    assert metrics.vector_norm_mean.value == pytest.approx(1.0)
    assert metrics.eigenvalues == pytest.approx((2.0 / 3.0, 2.0 / 3.0))


def test_rank_one_representations_report_effective_rank_one() -> None:
    representations = torch.tensor([[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]])

    metrics = compute_representation_metrics(representations)

    assert metrics.effective_rank.value == pytest.approx(1.0)
    assert metrics.effective_rank_normalized.value == pytest.approx(0.5)
    assert metrics.top_eigen_fraction.value == pytest.approx(1.0)


def test_thresholds_flag_low_std_and_low_rank_without_changing_values() -> None:
    representations = torch.tensor([[-0.001, -0.001], [0.001, 0.001]])
    config = EvaluationConfig(collapse_std_threshold=0.01, collapse_rank_threshold=0.75)

    metrics = compute_representation_metrics(representations, config)

    assert metrics.collapsed is True
    assert metrics.collapse_reasons == ("low_latent_std", "low_effective_rank")
    assert metrics.effective_rank.value == pytest.approx(1.0)


def test_zero_spectrum_is_null_and_explicitly_collapsed() -> None:
    metrics = compute_representation_metrics(torch.ones(5, 3))

    assert metrics.collapsed is True
    assert metrics.collapse_reasons == ("zero_spectrum",)
    assert metrics.effective_rank == MetricValue(None, "zero_spectrum")
    assert metrics.latent_std_mean == MetricValue(None, "zero_spectrum")
    assert metrics.vector_norm_mean.value == pytest.approx(3.0**0.5)
    assert metrics.eigenvalues is None


def test_single_sample_keeps_vector_norm_but_nulls_covariance_metrics() -> None:
    metrics = compute_representation_metrics(torch.tensor([[3.0, 4.0]]))

    assert metrics.vector_norm_mean == MetricValue(5.0)
    assert metrics.effective_rank == MetricValue(None, "insufficient_samples")
    assert metrics.collapsed is True


def test_empty_and_non_finite_representations_return_stable_reasons() -> None:
    empty = compute_representation_metrics(torch.empty(0, 2))
    non_finite = compute_representation_metrics(torch.tensor([[0.0, float("nan")]]))

    assert empty.vector_norm_mean == MetricValue(None, "no_samples")
    assert empty.collapse_reasons == ("no_samples",)
    assert non_finite.vector_norm_mean == MetricValue(None, "non_finite")
    assert non_finite.collapse_reasons == ("non_finite",)


def test_finite_inputs_with_overflowing_norm_return_non_finite_reason() -> None:
    metrics = compute_representation_metrics(torch.full((2, 2), 1.0e308, dtype=torch.float64))

    assert metrics.vector_norm_mean == MetricValue(None, "non_finite")
    assert metrics.collapse_reasons == ("non_finite",)


def test_representation_metrics_reject_invalid_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_representation_metrics(torch.ones(3))
    with pytest.raises(ValueError, match="one dimension"):
        compute_representation_metrics(torch.empty(3, 0))


def test_weighted_mean_is_invariant_to_partial_batch_weighting() -> None:
    metric = WeightedMean()
    metric.update(2.0, 4)
    metric.update(6.0, 1)

    assert metric.compute() == MetricValue(2.8)
    assert WeightedMean().compute() == MetricValue(None, "no_samples")


def test_weighted_mean_rejects_invalid_values_and_weights() -> None:
    metric = WeightedMean()
    with pytest.raises(ValueError, match="finite"):
        metric.update(float("nan"), 1)
    with pytest.raises(ValueError, match="positive integers"):
        metric.update(1.0, 0)


def test_global_gradient_norm_matches_manual_l2_and_handles_missing() -> None:
    encoder = LinearEncoder(2, 1)
    assert global_gradient_norm(encoder.parameters()) == MetricValue(None, "missing_gradient")

    loss = encoder(torch.tensor([[2.0, -1.0]])).square().sum()
    loss.backward()
    expected = sum(parameter.grad.square().sum() for parameter in encoder.parameters()).sqrt()

    assert global_gradient_norm(encoder.parameters()).value == pytest.approx(expected.item())


def test_global_gradient_norm_reports_non_finite_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])

    assert global_gradient_norm([parameter]) == MetricValue(None, "non_finite")


def test_ridge_probe_recovers_offset_multi_output_mapping() -> None:
    features = torch.tensor([[-2.0, 1.0], [-1.0, 0.0], [0.0, 2.0], [1.0, -1.0], [2.0, 3.0]])
    true_weights = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    true_intercept = torch.tensor([4.0, -2.0])
    targets = features @ true_weights + true_intercept

    probe = fit_ridge_probe(features, targets, alpha=1.0e-10)
    predictions = probe.predict(features)

    assert probe.weights.dtype == torch.float64
    assert probe.weights.device.type == "cpu"
    torch.testing.assert_close(predictions, targets.to(torch.float64), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        probe.intercept, true_intercept.to(torch.float64), rtol=1e-8, atol=1e-8
    )
    assert ridge_r2_score(probe, features, targets).value == pytest.approx(1.0)


def test_ridge_probe_handles_singular_features_via_regularization() -> None:
    features = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    targets = torch.tensor([[1.0], [2.0], [3.0]])

    probe = fit_ridge_probe(features, targets, alpha=1.0e-3)

    assert torch.isfinite(probe.weights).all()
    assert ridge_r2_score(probe, features, targets).value > 0.99


def test_probe_uses_only_train_means_for_validation_predictions() -> None:
    train_features = torch.tensor([[0.0], [1.0], [2.0]])
    train_targets = 3 * train_features + 5
    validation_features = torch.tensor([[10.0], [11.0]])
    validation_targets = 3 * validation_features + 5
    probe = fit_ridge_probe(train_features, train_targets, alpha=1.0e-10)
    frozen_feature_mean = probe.feature_mean.clone()
    frozen_target_mean = probe.target_mean.clone()

    score = ridge_r2_score(probe, validation_features, validation_targets)

    assert score.value == pytest.approx(1.0)
    torch.testing.assert_close(probe.feature_mean, frozen_feature_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(probe.target_mean, frozen_target_mean, rtol=0.0, atol=0.0)
    assert probe.feature_mean.item() == pytest.approx(1.0)
    assert probe.target_mean.item() == pytest.approx(8.0)


def test_constant_train_target_returns_null_r2() -> None:
    features = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    targets = torch.full((4, 2), 7.0)
    probe = fit_ridge_probe(features, targets, alpha=1.0e-6)

    torch.testing.assert_close(
        probe.predict(torch.tensor([[100.0]])), torch.tensor([[7.0, 7.0]], dtype=torch.float64)
    )
    assert ridge_r2_score(probe, features, targets) == MetricValue(None, "constant_target")


@pytest.mark.parametrize(
    "features,targets,match",
    [
        (torch.ones(2, 2), torch.ones(3, 1), "same number"),
        (torch.ones(2), torch.ones(2, 1), "shape"),
        (torch.ones(2, 2), torch.tensor([[1.0], [float("inf")]]), "non-finite"),
    ],
)
def test_ridge_probe_rejects_invalid_inputs(features, targets, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fit_ridge_probe(features, targets, alpha=1.0e-6)


def test_evaluation_config_rejects_rank_threshold_above_one() -> None:
    config = ExperimentConfig(
        evaluation=replace(ExperimentConfig().evaluation, collapse_rank_threshold=1.1)
    )
    with pytest.raises(ValueError, match="collapse_rank_threshold"):
        validate_config(config)


def test_metrics_reject_invalid_evaluation_config() -> None:
    with pytest.raises(ValueError, match="covariance_epsilon"):
        compute_representation_metrics(torch.eye(2), EvaluationConfig(covariance_epsilon=0.0))


def test_synthetic_windows_flow_through_encoder_metrics_and_probe() -> None:
    data_config = DataConfig(
        num_samples=12,
        validation_samples=4,
        test_samples=4,
        sequence_length=12,
        window_size=3,
        burn_in_steps=4,
        latent_state_dim=2,
        observation_dim=3,
    )
    dataset = LatentDynamicsDataset(data_config, "train")
    encoder = LinearEncoder(data_config.window_size * data_config.observation_dim, 4)
    with torch.no_grad():
        representations = encoder(dataset.contexts.flatten(start_dim=1))

    metrics = compute_representation_metrics(representations)
    probe = fit_ridge_probe(representations, dataset.context_states, alpha=1.0e-6)
    score = ridge_r2_score(probe, representations, dataset.context_states)

    assert metrics.effective_rank.value is not None
    assert score.value is not None
    assert math.isfinite(score.value)
