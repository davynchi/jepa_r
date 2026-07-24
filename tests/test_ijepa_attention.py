from __future__ import annotations

import torch

from jepa.models.ijepa import Attention


def test_sdpa_matches_explicit_attention_in_eval_mode() -> None:
    torch.manual_seed(7)
    attention = Attention(
        dim=24,
        num_heads=3,
        qkv_bias=True,
        qk_scale=0.17,
        attn_drop=0.2,
        proj_drop=0.0,
    ).eval()
    inputs = torch.randn(2, 11, 24)

    sdpa_output, no_weights = attention(inputs)
    explicit_output, weights = attention(inputs, return_attention=True)

    assert no_weights is None
    assert weights is not None
    assert weights.shape == (2, 3, 11, 11)
    torch.testing.assert_close(sdpa_output, explicit_output, rtol=1.0e-5, atol=1.0e-6)


def test_sdpa_attention_backward_is_finite() -> None:
    attention = Attention(dim=24, num_heads=3, qkv_bias=True)
    inputs = torch.randn(2, 11, 24, requires_grad=True)

    output, _ = attention(inputs)
    output.square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in attention.parameters()
    )
