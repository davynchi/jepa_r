from __future__ import annotations

import torch

from jepa.video_diagnostics import pairing_context_batch
from jepa.video_jepa import (
    CausalVideoJEPA,
    CausalVideoJEPAConfig,
    make_3d_sincos_position_embedding,
)


def tiny_config(**overrides: object) -> CausalVideoJEPAConfig:
    values: dict[str, object] = {
        "image_size": 32,
        "total_frames": 8,
        "context_frames": 4,
        "patch_size": 8,
        "tubelet_size": 2,
        "embed_dim": 32,
        "encoder_depth": 1,
        "predictor_depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
    }
    values.update(overrides)
    return CausalVideoJEPAConfig(**values)  # type: ignore[arg-type]


def test_3d_positions_are_unique_and_axis_separated() -> None:
    positions = make_3d_sincos_position_embedding(5, 8, 8, 128)
    assert positions.shape == (320, 128)
    assert len(torch.unique(positions, dim=0)) == 320
    centered_rank = int(torch.linalg.matrix_rank(positions - positions.mean(dim=0)))
    assert centered_rank >= 15

    grid = positions.reshape(5, 8, 8, 128)
    assert not torch.equal(grid[0, 3, 4], grid[3, 0, 4])
    assert not torch.equal(grid[0, 3, 4], grid[4, 3, 0])


def test_causal_video_jepa_forward_shapes_and_gradients() -> None:
    model = CausalVideoJEPA(tiny_config())
    context = torch.rand(3, 4, 1, 32, 32)
    target = torch.rand(3, 4, 1, 32, 32)
    output = model(context, target)

    assert output["per_sample_loss"].shape == (3,)
    assert output["per_token_loss"].shape == (3, 32)
    assert output["context_tokens"].shape == (3, 32, 32)
    assert output["predicted_tokens"].shape == output["target_tokens"].shape
    assert output["motion_token_weights"].shape == (3, 32)
    assert torch.isfinite(output["loss"])

    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_pairing_controls_return_matched_temporal_variants() -> None:
    model = CausalVideoJEPA(tiny_config()).eval()
    context = torch.rand(4, 4, 1, 32, 32)
    target = torch.rand(4, 4, 1, 32, 32)
    losses = pairing_context_batch(model, context, target)
    assert set(losses) == {
        "correct",
        "shuffled_context",
        "zero_context",
        "reversed_context",
        "shuffled_time_context",
        "last_frame_context",
        "shuffled_target",
        "reversed_target",
        "shuffled_time_target",
    }
    assert all(value.shape == (4,) for value in losses.values())
    assert all(torch.isfinite(value).all() for value in losses.values())


def test_group_matched_pairing_marks_singletons_as_nan() -> None:
    model = CausalVideoJEPA(tiny_config()).eval()
    context = torch.rand(4, 4, 1, 32, 32)
    target = torch.rand(4, 4, 1, 32, 32)
    losses = pairing_context_batch(
        model,
        context,
        target,
        digit_labels=torch.tensor([1, 1, 2, 3]),
        direction_labels=torch.tensor([0, 0, 1, 1]),
    )
    assert torch.isfinite(losses["same_digit_shuffled_context"][:2]).all()
    assert torch.isnan(losses["same_digit_shuffled_context"][2:]).all()
    assert torch.isfinite(losses["same_direction_shuffled_context"]).all()


def test_ema_updates_target_encoder() -> None:
    model = CausalVideoJEPA(tiny_config())
    target_before = [parameter.detach().clone() for parameter in model.target_encoder.parameters()]
    with torch.no_grad():
        for parameter in model.context_encoder.parameters():
            parameter.add_(0.25)
    model.update_target_encoder(0.5)
    changed = [
        not torch.equal(before, after)
        for before, after in zip(target_before, model.target_encoder.parameters(), strict=True)
    ]
    assert any(changed)


def test_foreground_mask_matches_tubelet_geometry() -> None:
    config = tiny_config()
    model = CausalVideoJEPA(config)
    target = torch.zeros(1, config.target_frames, 1, 32, 32)
    target[:, :2, :, :8, :8] = 1.0
    mask = model.target_foreground_mask(target)
    assert mask.shape == (1, config.target_token_count)
    assert int(mask.sum()) == 1


def test_foreground_weighted_loss_is_normalized() -> None:
    predictions = torch.zeros(1, 2, 1)
    targets = torch.tensor([[[1.0], [3.0]]])
    unweighted, _, _ = CausalVideoJEPA.feature_prediction_loss(predictions, targets)
    weighted, _, _ = CausalVideoJEPA.feature_prediction_loss(
        predictions,
        targets,
        torch.tensor([[1.0, 4.0]]),
    )
    assert torch.isclose(unweighted, torch.tensor(2.0))
    assert torch.isclose(weighted, torch.tensor(2.6))


def test_motion_weights_focus_on_changed_tubelets_and_keep_mean_one() -> None:
    config = tiny_config(target_weighting_mode="motion")
    model = CausalVideoJEPA(config)
    context = torch.zeros(1, config.context_frames, 1, 32, 32)
    target = torch.zeros(1, config.target_frames, 1, 32, 32)
    target[:, :, :, :8, :8] = torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 4, 1, 1, 1)
    activity = model.target_motion_activity(context, target)
    weights = model.motion_token_weights(activity)
    assert activity.shape == (1, config.target_token_count)
    assert weights.shape == activity.shape
    assert torch.allclose(weights.mean(dim=1), torch.ones(1), atol=1e-6)
    assert float(weights.max()) > float(weights.min())


def test_motion_objective_uses_motion_weighted_loss() -> None:
    model = CausalVideoJEPA(tiny_config(target_weighting_mode="motion"))
    context = torch.rand(2, 4, 1, 32, 32)
    target = torch.rand(2, 4, 1, 32, 32)
    output = model(context, target)
    assert torch.allclose(output["loss"], output["motion_weighted_loss"])
