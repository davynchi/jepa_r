"""Patchify utility and a positionally-conditioned patch predictor.

Minimal V-JEPA-style collapse fix: instead of one global vector predicting
another global vector (which admits a trivial constant-output solution under
plain MSE + stop-gradient/EMA -- see the investigation that motivated this),
split each frame into a grid of patches, encode each patch independently
(reusing the existing linear/nonlinear/cnn encoders unmodified -- they already
operate generically on any ``[..., input_dim]`` leading-batch shape), and
predict *several* target patches, each conditioned on a learned embedding for
*which* patch position is being asked for. A collapsed encoder cannot satisfy
this: different images' patches at the same position genuinely differ, and
position alone can't explain that variance, so the trivial "always output a
constant" solution no longer gives zero loss. No self-attention/ViT needed --
that mechanism, not cross-patch attention, is what breaks the collapse.
"""

from __future__ import annotations

import torch
from torch import nn


def patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[..., C, H, W] -> [..., num_patches, C*patch_size*patch_size].

    H and W must be divisible by ``patch_size``. Patches are taken in
    row-major (top-to-bottom, left-to-right) order.
    """
    *lead, channels, height, width = frames.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"H={height}, W={width} must be divisible by patch_size={patch_size}")
    patches_h, patches_w = height // patch_size, width // patch_size
    n = len(lead)
    x = frames.reshape(*lead, channels, patches_h, patch_size, patches_w, patch_size)
    # -> [..., patches_h, patches_w, channels, patch_size, patch_size]
    x = x.permute(*range(n), n + 1, n + 3, n, n + 2, n + 4)
    return x.reshape(*lead, patches_h * patches_w, channels * patch_size * patch_size).contiguous()


class PositionalPatchPredictor(nn.Module):
    """Predicts K target patches' latents from a pooled context summary,
    each conditioned on a learned embedding for its patch index."""

    def __init__(
        self,
        num_patches: int,
        context_dim: int,
        patch_latent_dim: int,
        *,
        position_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(num_patches, position_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + position_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, patch_latent_dim),
        )

    def forward(self, context_summary: torch.Tensor, patch_indices: torch.Tensor) -> torch.Tensor:
        """``context_summary``: [batch, context_dim]. ``patch_indices``: [k]
        (the same k target positions for the whole batch). Returns [batch, k, patch_latent_dim]."""
        batch = context_summary.shape[0]
        k = patch_indices.shape[0]
        position = self.position_embedding(patch_indices)  # [k, position_dim]
        context_expanded = context_summary.unsqueeze(1).expand(-1, k, -1)
        position_expanded = position.unsqueeze(0).expand(batch, -1, -1)
        combined = torch.cat([context_expanded, position_expanded], dim=-1)
        return self.mlp(combined)
