from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from jepa.models import build_model_pair
from jepa.training import (
    build_jepa_core,
    compute_loss,
    optimizer_parameters,
    resolve_policy,
    train_step,
)


def _module_types(module: nn.Module, module_type: type[nn.Module]) -> list[nn.Module]:
    return [child for child in module.modules() if isinstance(child, module_type)]


def _all_parameter_ids(module: nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def _nonzero_gradient_exists(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def test_linear_family_contains_only_one_affine_layer_per_module() -> None:
    encoder, predictor = build_model_pair("linear", input_dim=12, latent_dim=5)

    assert len(_module_types(encoder, nn.Linear)) == 1
    assert len(_module_types(predictor, nn.Linear)) == 1
    assert not _module_types(encoder, nn.Tanh)
    assert not _module_types(predictor, nn.Tanh)
    assert encoder(torch.randn(3, 12)).shape == (3, 5)
    assert predictor(torch.randn(3, 5)).shape == (3, 5)


@pytest.mark.parametrize("hidden_layers", [1, 3])
def test_nonlinear_family_has_exact_tanh_structure(hidden_layers: int) -> None:
    encoder, predictor = build_model_pair(
        "nonlinear",
        input_dim=12,
        latent_dim=5,
        hidden_dim=7,
        hidden_layers=hidden_layers,
    )

    for module in (encoder, predictor):
        assert len(_module_types(module, nn.Tanh)) == hidden_layers
        assert len(_module_types(module, nn.Linear)) == hidden_layers + 1
    assert encoder(torch.randn(3, 12)).shape == (3, 5)
    assert predictor(torch.randn(3, 5)).shape == (3, 5)


def test_nonlinear_family_rejects_zero_hidden_layers() -> None:
    with pytest.raises(ValueError, match="hidden_layers must be positive"):
        build_model_pair("nonlinear", input_dim=12, latent_dim=5, hidden_dim=7, hidden_layers=0)


@pytest.mark.parametrize(
    (
        "stop_gradient",
        "ema_enabled",
        "separate_target",
        "optimize_target",
        "target_receives_gradient",
        "variant",
    ),
    [
        (False, False, False, False, True, "shared-gradient-target"),
        (True, False, False, False, False, "shared-stop-gradient-target"),
        (True, True, True, False, False, "ema-stop-gradient-target"),
        (False, True, True, True, True, "hybrid-gradient-ema-target"),
    ],
)
def test_gradient_policy_truth_table(
    stop_gradient: bool,
    ema_enabled: bool,
    separate_target: bool,
    optimize_target: bool,
    target_receives_gradient: bool,
    variant: str,
) -> None:
    torch.manual_seed(7)
    core = build_jepa_core(
        "linear",
        input_dim=6,
        latent_dim=4,
        stop_gradient=stop_gradient,
        ema_enabled=ema_enabled,
    )
    policy = resolve_policy(stop_gradient=stop_gradient, ema_enabled=ema_enabled)

    assert core.policy == policy
    assert core.policy.variant == variant
    assert (core.target_encoder is not core.context_encoder) is separate_target

    optimizer_ids = {id(parameter) for parameter in optimizer_parameters(core)}
    context_ids = _all_parameter_ids(core.context_encoder)
    predictor_ids = _all_parameter_ids(core.predictor)
    target_ids = _all_parameter_ids(core.target_encoder)
    assert context_ids <= optimizer_ids
    assert predictor_ids <= optimizer_ids
    if optimize_target:
        assert target_ids <= optimizer_ids
    elif separate_target:
        assert target_ids.isdisjoint(optimizer_ids)
        assert not any(parameter.requires_grad for parameter in core.target_encoder.parameters())

    context = torch.randn(8, 6)
    target = torch.randn(8, 6)
    forward = compute_loss(core, context, target)
    if forward.target_latent.requires_grad:
        forward.target_latent.retain_grad()
    forward.loss.backward()

    assert _nonzero_gradient_exists(core.context_encoder)
    assert _nonzero_gradient_exists(core.predictor)
    if separate_target:
        assert _nonzero_gradient_exists(core.target_encoder) is target_receives_gradient
    assert (forward.target_latent.grad is not None) is target_receives_gradient


@pytest.mark.parametrize("stop_gradient", [False, True])
def test_ema_target_starts_as_an_exact_context_copy(stop_gradient: bool) -> None:
    torch.manual_seed(11)
    core = build_jepa_core(
        "nonlinear",
        input_dim=6,
        latent_dim=4,
        hidden_dim=8,
        hidden_layers=2,
        stop_gradient=stop_gradient,
        ema_enabled=True,
    )

    assert core.target_encoder is not core.context_encoder
    for context_parameter, target_parameter in zip(
        core.context_encoder.parameters(), core.target_encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(target_parameter, context_parameter, rtol=0.0, atol=0.0)
        assert target_parameter is not context_parameter


def test_hybrid_policy_applies_adam_before_ema_and_retains_moments() -> None:
    torch.manual_seed(19)
    actual = build_jepa_core(
        "linear",
        input_dim=6,
        latent_dim=4,
        stop_gradient=False,
        ema_enabled=True,
    )
    reference = deepcopy(actual)
    actual_optimizer = torch.optim.Adam(optimizer_parameters(actual), lr=0.01)
    reference_optimizer = torch.optim.Adam(optimizer_parameters(reference), lr=0.01)
    context = torch.randn(8, 6)
    target = torch.randn(8, 6)
    decay = 0.8

    reference_optimizer.zero_grad(set_to_none=True)
    reference_forward = compute_loss(reference, context, target)
    reference_forward.loss.backward()
    reference_optimizer.step()
    expected_targets = [
        decay * target_parameter.detach().clone()
        + (1.0 - decay) * context_parameter.detach().clone()
        for target_parameter, context_parameter in zip(
            reference.target_encoder.parameters(),
            reference.context_encoder.parameters(),
            strict=True,
        )
    ]

    train_step(actual, actual_optimizer, context, target, ema_decay=decay)

    for actual_parameter, expected_parameter in zip(
        actual.target_encoder.parameters(), expected_targets, strict=True
    ):
        torch.testing.assert_close(actual_parameter, expected_parameter)
        state = actual_optimizer.state[actual_parameter]
        assert "exp_avg" in state
        assert "exp_avg_sq" in state


def test_ema_decay_validation() -> None:
    core = build_jepa_core(
        "linear",
        input_dim=6,
        latent_dim=4,
        stop_gradient=True,
        ema_enabled=True,
    )
    optimizer = torch.optim.Adam(optimizer_parameters(core), lr=0.01)
    parameters_before = [
        parameter.detach().clone() for parameter in core.context_encoder.parameters()
    ]

    with pytest.raises(ValueError, match="EMA decay"):
        train_step(core, optimizer, torch.randn(2, 6), torch.randn(2, 6), ema_decay=1.0)

    for parameter, before in zip(core.context_encoder.parameters(), parameters_before, strict=True):
        torch.testing.assert_close(parameter, before, rtol=0.0, atol=0.0)


def test_model_factory_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model_pair("transformer", input_dim=12, latent_dim=5)  # type: ignore[arg-type]
