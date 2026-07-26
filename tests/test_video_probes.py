from __future__ import annotations

import torch

from jepa.video_diagnostics import token_sequence_diagnostics
from jepa.video_probes import (
    ProbeConfig,
    evaluate_low_shot_feature_variant,
    evaluate_probe_suite,
    fit_attentive_classifier,
    predict_attentive_classifier,
    raw_context_features,
    spatially_pool_tokens,
    stratified_subset_indices,
    temporal_pool_tokens,
)


def make_split(samples: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(samples, 12, generator=generator)
    temporal_features = torch.cat([features, features[:, :4]], dim=1)
    direction = (features[:, 0] > 0).long() + 2 * (features[:, 1] > 0).long()
    digit = torch.remainder((features[:, 2] * 3).floor().long().abs(), 10)
    bounce = (features[:, 3] > 0).long()
    velocity_x = features[:, 7]
    velocity_y = features[:, 8]
    boundary_velocity = torch.stack([velocity_x, velocity_y], dim=-1)
    speed = 2.5 + 0.8 * features[:, 4]
    base_x = 16.0 + 2.0 * features[:, 5]
    base_y = 16.0 + 2.0 * features[:, 6]
    time = torch.arange(4, dtype=torch.float32)[None, :, None]
    start = torch.stack([base_x, base_y], dim=-1)[:, None, :]
    velocity = boundary_velocity[:, None, :]
    future = start + time * velocity
    return {
        "features": features,
        "temporal_features": temporal_features,
        "direction_label": direction,
        "speed": speed,
        "boundary_velocity": boundary_velocity,
        "future_positions": future,
        "bounce_target": bounce,
        "digit_label": digit,
        "sample_id": torch.arange(samples),
    }


def test_probe_suite_recovers_linear_signals() -> None:
    train = make_split(256, 1)
    validation = make_split(128, 2)
    test = make_split(128, 3)
    metrics, hyperparameters = evaluate_probe_suite(
        train,
        validation,
        test,
        num_directions=4,
        config=ProbeConfig(
            classifier_epochs=20,
            classifier_batch_size=64,
            classifier_lr=0.05,
            seed=5,
        ),
    )
    assert metrics["direction_accuracy"] > 0.85
    assert metrics["bounce_accuracy"] > 0.85
    assert metrics["speed_rmse"] < 0.2
    assert metrics["velocity_vector_rmse"] < 0.25
    assert metrics["future_fde"] < 0.5
    assert hyperparameters["speed_ridge_lambda"] > 0.0


def test_spatial_pool_keeps_time_axis() -> None:
    tokens = torch.arange(2 * 3 * 4 * 4 * 8, dtype=torch.float32).reshape(2, 48, 8)
    pooled = spatially_pool_tokens(
        tokens,
        time_tokens=3,
        spatial_tokens_per_side=4,
        output_side=2,
    )
    assert pooled.shape == (2, 12, 8)
    temporal = temporal_pool_tokens(
        tokens,
        time_tokens=3,
        spatial_tokens_per_side=4,
    )
    assert temporal.shape == (2, 3, 8)


def test_raw_context_features_have_compact_shapes() -> None:
    context = torch.rand(5, 4, 1, 32, 32)
    features = raw_context_features(context, output_side=4)
    assert features["raw_temporal_features"].shape == (5, 64)
    assert features["raw_last_frame_features"].shape == (5, 16)
    assert features["raw_difference_features"].shape == (5, 48)


def test_token_diagnostics_separate_pooled_and_position_residual_rank() -> None:
    generator = torch.Generator().manual_seed(4)
    tokens = torch.randn(32, 6, 12, generator=generator)
    diagnostics = token_sequence_diagnostics(tokens, "context")
    assert diagnostics["context_pooled_effective_rank"] > 1.0
    assert diagnostics["context_token_effective_rank"] > 1.0
    assert diagnostics["context_token_variance_across_samples"] > 0.0


def test_attentive_probe_recovers_position_specific_signal() -> None:
    generator = torch.Generator().manual_seed(7)

    def make_tokens(samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(0, 2, (samples,), generator=generator)
        sign = labels.float().mul(2).sub(1)
        tokens = 0.03 * torch.randn(samples, 4, 8, generator=generator)
        tokens[:, 0, 0] += 2.0
        tokens[:, 1, 1] += 2.0
        tokens[:, 0, 2] += sign
        tokens[:, 1, 2] -= sign
        return tokens, labels

    train_tokens, train_labels = make_tokens(192)
    validation_tokens, validation_labels = make_tokens(96)
    test_tokens, test_labels = make_tokens(96)
    config = ProbeConfig(
        attentive_epochs=20,
        attentive_batch_size=32,
        attentive_lr=0.003,
        attentive_heads=2,
        seed=11,
    )
    model, _ = fit_attentive_classifier(
        train_tokens,
        train_labels,
        validation_tokens,
        validation_labels,
        num_classes=2,
        config=config,
        device=torch.device("cpu"),
    )
    predictions = predict_attentive_classifier(
        model,
        test_tokens,
        batch_size=32,
        device=torch.device("cpu"),
    )
    assert float((predictions == test_labels).float().mean()) > 0.9


def test_stratified_subset_and_lowshot_probe() -> None:
    train = make_split(256, 10)
    validation = make_split(128, 11)
    evaluation = make_split(128, 12)
    indices = stratified_subset_indices(train["direction_label"], 64, seed=3)
    assert len(indices) == 64
    counts = torch.bincount(train["direction_label"][indices], minlength=4)
    assert int(counts.max() - counts.min()) <= 1

    metrics, _ = evaluate_low_shot_feature_variant(
        train,
        validation,
        evaluation,
        feature_key="temporal_features",
        budget=128,
        split_seed=4,
        num_directions=4,
        config=ProbeConfig(
            classifier_epochs=20,
            classifier_batch_size=64,
            classifier_lr=0.05,
            seed=4,
        ),
    )
    assert metrics["direction_accuracy"] > 0.8
    assert metrics["velocity_vector_rmse"] < 0.3
    assert metrics["future_fde"] < 0.6
