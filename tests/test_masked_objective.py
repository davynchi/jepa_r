from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

import jepa.training as training_module
from jepa.config import (
    DataConfig,
    EMAConfig,
    ExperimentConfig,
    MaskedPatchesConfig,
    ModelConfig,
    ObjectiveConfig,
    OutputConfig,
    TrainingConfig,
)
from jepa.data import build_dataset_bundle
from jepa.training import (
    build_masked_jepa_core,
    compute_masked_loss,
    ema_update,
    masked_patch_indices,
    optimizer_parameters,
    parameter_counts,
    train_experiment,
)


def _config(root: Path, *, epochs: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            num_samples=8,
            validation_samples=4,
            test_samples=4,
            sequence_length=64,
            window_size=32,
            burn_in_steps=2,
            latent_state_dim=2,
            observation_dim=2,
        ),
        model=ModelConfig(latent_dim=3, hidden_dim=5, hidden_layers=1),
        training=TrainingConfig(
            seed=17,
            learning_rate=0.005,
            batch_size=4,
            epochs=epochs,
            device="cpu",
            evaluation_every_epochs=1,
            checkpoint_every_epochs=1,
            ema=EMAConfig(enabled=True, decay=0.9),
        ),
        output=OutputConfig(root=str(root)),
        objective=ObjectiveConfig(
            kind="masked_patches",
            masked_patches=MaskedPatchesConfig(
                patch_size=8,
                mask_ratio=0.5,
                position_dim=16,
            ),
        ),
    )


def _masks(indices: torch.Tensor, epoch: int | None) -> torch.Tensor:
    return masked_patch_indices(
        replicate_seed=7,
        dataset_fingerprint="a" * 64,
        split="train" if epoch is not None else "validation",
        sample_indices=indices,
        num_patches=8,
        masked_count=4,
        epoch=epoch,
    )


def test_masks_are_sorted_unique_order_independent_and_epoch_aware() -> None:
    indices = torch.arange(12)
    first = _masks(indices, 1)
    shuffled_indices = indices[torch.tensor([7, 1, 9, 0, 5, 3, 11, 2, 4, 8, 6, 10])]
    shuffled = _masks(shuffled_indices, 1)
    restored = {int(index): mask for index, mask in zip(shuffled_indices, shuffled, strict=True)}

    assert first.shape == (12, 4)
    assert torch.all(first[:, 1:] > first[:, :-1])
    assert all(torch.equal(first[index], restored[index]) for index in range(12))
    assert torch.any(first != _masks(indices, 2))
    assert torch.equal(_masks(indices, None), _masks(indices, None))


def test_masked_values_cannot_change_context_and_loss_is_vectorized_mse() -> None:
    torch.manual_seed(3)
    core = build_masked_jepa_core(
        "linear",
        patch_input_dim=8 * 2,
        num_patches=8,
        latent_dim=4,
        position_dim=16,
        stop_gradient=True,
        ema_enabled=False,
    )
    sequences = torch.randn(3, 64, 2)
    targets = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7], [0, 1, 6, 7]])
    original = compute_masked_loss(core, sequences, targets, patch_size=8)
    changed = sequences.clone()
    patches = changed.reshape(3, 8, 8, 2)
    patches.scatter_(
        1,
        targets[:, :, None, None].expand(-1, -1, 8, 2),
        torch.full((3, 4, 8, 2), 1000.0),
    )
    modified = compute_masked_loss(core, changed, targets, patch_size=8)

    assert original.prediction.shape == (3, 4, 4)
    assert original.target_latent.shape == (3, 4, 4)
    torch.testing.assert_close(original.context_latent, modified.context_latent)
    torch.testing.assert_close(
        original.loss,
        torch.mean((original.prediction - original.target_latent.detach()) ** 2),
    )


def test_target_position_changes_predictor_query() -> None:
    torch.manual_seed(5)
    core = build_masked_jepa_core(
        "linear",
        patch_input_dim=16,
        num_patches=8,
        latent_dim=4,
        position_dim=16,
        stop_gradient=True,
        ema_enabled=False,
    )
    grid = torch.randn(2, 8, 4)
    visibility = torch.ones(2, 8, dtype=torch.bool)
    first = core.predictor(grid, visibility, torch.tensor([[0, 1], [0, 1]]))
    second = core.predictor(grid, visibility, torch.tensor([[0, 2], [0, 2]]))

    torch.testing.assert_close(first[:, 0], second[:, 0])
    assert not torch.equal(first[:, 1], second[:, 1])


@pytest.mark.parametrize("stop_gradient", [False, True])
@pytest.mark.parametrize("ema_enabled", [False, True])
def test_masked_policy_gradients_and_ema_updates(stop_gradient: bool, ema_enabled: bool) -> None:
    torch.manual_seed(11)
    core = build_masked_jepa_core(
        "nonlinear",
        patch_input_dim=16,
        num_patches=8,
        latent_dim=4,
        position_dim=16,
        stop_gradient=stop_gradient,
        ema_enabled=ema_enabled,
        hidden_dim=7,
    )
    optimizer = torch.optim.Adam(optimizer_parameters(core), lr=0.01)
    before_target = [parameter.detach().clone() for parameter in core.target_encoder.parameters()]
    forward = compute_masked_loss(
        core,
        torch.randn(4, 64, 2),
        torch.tensor([[0, 2, 4, 6]] * 4),
        patch_size=8,
    )
    if forward.target_latent.requires_grad:
        forward.target_latent.retain_grad()
    forward.loss.backward()

    assert (forward.target_latent.grad is None) is stop_gradient
    optimizer.step()
    after_adam = [parameter.detach().clone() for parameter in core.target_encoder.parameters()]
    if ema_enabled:
        ema_update(core.target_encoder, core.context_encoder, 0.9)
        assert any(
            not torch.equal(after, current)
            for after, current in zip(after_adam, core.target_encoder.parameters(), strict=True)
        )
    elif core.target_encoder is core.context_encoder:
        assert any(
            not torch.equal(before, current)
            for before, current in zip(before_target, core.target_encoder.parameters(), strict=True)
        )


def test_parameter_counts_do_not_duplicate_shared_encoder() -> None:
    core = build_masked_jepa_core(
        "linear",
        patch_input_dim=16,
        num_patches=8,
        latent_dim=4,
        position_dim=16,
        stop_gradient=False,
        ema_enabled=False,
    )
    counts = parameter_counts(core)
    expected = sum(parameter.numel() for parameter in core.context_encoder.parameters()) + sum(
        parameter.numel() for parameter in core.predictor.parameters()
    )
    assert counts == {
        "unique_total": expected,
        "requires_grad": expected,
        "optimizer": expected,
    }


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_masked_cpu_resume_is_bitwise_and_rejects_objective_or_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = train_experiment(_config(tmp_path / "baseline", epochs=3), seed_label=0)
    interrupted_config = _config(tmp_path / "resumed", epochs=3)
    original = training_module._train_epoch

    def interrupt_at_three(*args: Any, **kwargs: Any) -> Any:
        if args[-1] == 3:
            raise RuntimeError("interrupt")
        return original(*args, **kwargs)

    monkeypatch.setattr(training_module, "_train_epoch", interrupt_at_three)
    with pytest.raises(RuntimeError, match="interrupt"):
        train_experiment(interrupted_config, seed_label=0)
    run_dir = next((tmp_path / "resumed").iterdir())
    monkeypatch.setattr(training_module, "_train_epoch", original)
    resumed = train_experiment(interrupted_config, resume_from=run_dir / "checkpoint.pt")

    baseline_checkpoint = torch.load(
        baseline.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    resumed_checkpoint = torch.load(
        resumed.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    for key in ("context_encoder", "predictor", "target_encoder", "optimizer"):
        _assert_nested_equal(baseline_checkpoint[key], resumed_checkpoint[key])
    assert baseline.metrics["objective"] == resumed.metrics["objective"]

    status = json.loads((run_dir / "status.json").read_text())
    status["state"] = "failed"
    (run_dir / "status.json").write_text(json.dumps(status))
    future = replace(interrupted_config, objective=ObjectiveConfig(kind="future_window"))
    with pytest.raises(ValueError, match="differs"):
        train_experiment(future, resume_from=run_dir / "checkpoint.pt")

    changed_bundle = deepcopy(build_dataset_bundle(interrupted_config.data))
    changed_bundle = replace(changed_bundle, fingerprint="f" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        train_experiment(
            interrupted_config,
            resume_from=run_dir / "checkpoint.pt",
            datasets=changed_bundle,
        )
