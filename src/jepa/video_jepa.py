"""A compact causal Video-JEPA model for controlled Moving-MNIST experiments."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CausalVideoJEPAConfig:
    image_size: int = 64
    total_frames: int = 20
    context_frames: int = 10
    in_channels: int = 1
    patch_size: int = 8
    tubelet_size: int = 2
    embed_dim: int = 128
    encoder_depth: int = 4
    predictor_depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    position_embedding_type: str = "separate_3d"
    target_weighting_mode: str = "uniform"
    foreground_weight: float = 1.0
    foreground_threshold: float = 0.05
    motion_weight_min: float = 0.25
    motion_weight_max: float = 4.0
    motion_weight_power: float = 1.0
    motion_weight_epsilon: float = 1e-6

    @property
    def target_frames(self) -> int:
        return self.total_frames - self.context_frames

    @property
    def spatial_tokens_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def context_time_tokens(self) -> int:
        return self.context_frames // self.tubelet_size

    @property
    def target_time_tokens(self) -> int:
        return self.target_frames // self.tubelet_size

    @property
    def context_token_count(self) -> int:
        side = self.spatial_tokens_per_side
        return self.context_time_tokens * side * side

    @property
    def target_token_count(self) -> int:
        side = self.spatial_tokens_per_side
        return self.target_time_tokens * side * side

    def validate(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.context_frames % self.tubelet_size != 0:
            raise ValueError("context_frames must be divisible by tubelet_size")
        if self.target_frames % self.tubelet_size != 0:
            raise ValueError("target_frames must be divisible by tubelet_size")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.embed_dim < 6:
            raise ValueError("embed_dim must be at least 6 for independent 3D positions")
        if self.position_embedding_type not in {"separate_3d", "legacy_sum"}:
            raise ValueError("unknown position_embedding_type")
        if self.encoder_depth <= 0 or self.predictor_depth <= 0:
            raise ValueError("encoder and predictor depths must be positive")
        if self.target_weighting_mode not in {"uniform", "foreground", "motion"}:
            raise ValueError("target_weighting_mode must be uniform, foreground, or motion")
        if self.foreground_weight < 1.0:
            raise ValueError("foreground_weight must be at least 1.0")
        if not 0.0 <= self.foreground_threshold <= 1.0:
            raise ValueError("foreground_threshold must be in [0,1]")
        if not 0.0 < self.motion_weight_min <= self.motion_weight_max:
            raise ValueError("expected 0 < motion_weight_min <= motion_weight_max")
        if self.motion_weight_power <= 0.0:
            raise ValueError("motion_weight_power must be positive")
        if self.motion_weight_epsilon <= 0.0:
            raise ValueError("motion_weight_epsilon must be positive")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def _axis_sincos(length: int, dim: int, *, offset: int = 0) -> Tensor:
    if length <= 0:
        raise ValueError("axis length must be positive")
    if dim < 0:
        raise ValueError("axis embedding dimension cannot be negative")
    if dim == 0:
        return torch.empty(length, 0, dtype=torch.float32)

    positions = torch.arange(offset, offset + length, dtype=torch.float32)[:, None]
    pair_count = dim // 2
    if pair_count == 0:
        return torch.zeros(length, dim, dtype=torch.float32)

    frequencies = torch.arange(pair_count, dtype=torch.float32)
    frequencies = torch.exp(
        -torch.log(torch.tensor(10_000.0))
        * frequencies
        / max(pair_count - 1, 1)
    )
    angles = positions * frequencies[None]
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    if embedding.shape[1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[1]))
    return embedding


def _axis_dimensions(embed_dim: int) -> tuple[int, int, int, int]:
    """Split channels into independent even-dimensional t/h/w subspaces."""

    pair_count = embed_dim // 2
    base, remainder = divmod(pair_count, 3)
    pairs = [base + int(index < remainder) for index in range(3)]
    dimensions = [2 * value for value in pairs]
    padding = embed_dim - sum(dimensions)
    return dimensions[0], dimensions[1], dimensions[2], padding


def make_3d_sincos_position_embedding(
    time_tokens: int,
    height_tokens: int,
    width_tokens: int,
    embed_dim: int,
    *,
    time_offset: int = 0,
) -> Tensor:
    """Return flattened [T*H*W,D] positions with separate t/h/w channels."""

    time_dim, height_dim, width_dim, padding_dim = _axis_dimensions(embed_dim)
    time = _axis_sincos(time_tokens, time_dim, offset=time_offset)
    height = _axis_sincos(height_tokens, height_dim)
    width = _axis_sincos(width_tokens, width_dim)

    time = time[:, None, None, :].expand(
        time_tokens, height_tokens, width_tokens, time_dim
    )
    height = height[None, :, None, :].expand(
        time_tokens, height_tokens, width_tokens, height_dim
    )
    width = width[None, None, :, :].expand(
        time_tokens, height_tokens, width_tokens, width_dim
    )
    parts = [time, height, width]
    if padding_dim > 0:
        parts.append(
            torch.zeros(
                time_tokens,
                height_tokens,
                width_tokens,
                padding_dim,
                dtype=torch.float32,
            )
        )
    positions = torch.cat(parts, dim=-1)
    return positions.reshape(time_tokens * height_tokens * width_tokens, embed_dim)


def make_legacy_3d_sincos_position_embedding(
    time_tokens: int,
    height_tokens: int,
    width_tokens: int,
    embed_dim: int,
    *,
    time_offset: int = 0,
) -> Tensor:
    """Original ambiguous t+h+w embedding retained for checkpoint diagnosis."""

    time = _axis_sincos(time_tokens, embed_dim, offset=time_offset)[:, None, None, :]
    height = _axis_sincos(height_tokens, embed_dim)[None, :, None, :]
    width = _axis_sincos(width_tokens, embed_dim)[None, None, :, :]
    positions = time + height + width
    return positions.reshape(time_tokens * height_tokens * width_tokens, embed_dim)


def make_video_position_embedding(
    time_tokens: int,
    height_tokens: int,
    width_tokens: int,
    embed_dim: int,
    *,
    time_offset: int,
    embedding_type: str,
) -> Tensor:
    if embedding_type == "separate_3d":
        return make_3d_sincos_position_embedding(
            time_tokens,
            height_tokens,
            width_tokens,
            embed_dim,
            time_offset=time_offset,
        )
    if embedding_type == "legacy_sum":
        return make_legacy_3d_sincos_position_embedding(
            time_tokens,
            height_tokens,
            width_tokens,
            embed_dim,
            time_offset=time_offset,
        )
    raise ValueError(f"unknown position embedding type: {embedding_type}")


class VideoTransformerEncoder(nn.Module):
    """Tubelet embedding followed by a pre-norm transformer encoder."""

    def __init__(self, config: CausalVideoJEPAConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.patch_embed = nn.Conv3d(
            config.in_channels,
            config.embed_dim,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, video: Tensor, *, time_offset: int) -> Tensor:
        if video.ndim != 5:
            raise ValueError("video must have shape [B,T,C,H,W]")
        _, frames, channels, height, width = video.shape
        if channels != self.config.in_channels:
            raise ValueError(f"expected {self.config.in_channels} channels, got {channels}")
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                f"expected {self.config.image_size}x{self.config.image_size}, "
                f"got {height}x{width}"
            )
        if frames % self.config.tubelet_size != 0:
            raise ValueError("frame count must be divisible by tubelet_size")

        features = self.patch_embed(video.permute(0, 2, 1, 3, 4).contiguous())
        _, _, time_tokens, height_tokens, width_tokens = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        positions = make_video_position_embedding(
            time_tokens,
            height_tokens,
            width_tokens,
            self.config.embed_dim,
            time_offset=time_offset,
            embedding_type=self.config.position_embedding_type,
        ).to(device=tokens.device, dtype=tokens.dtype)
        tokens = tokens + positions.unsqueeze(0)
        tokens = self.blocks(tokens)
        return self.norm(tokens)


class CausalFuturePredictor(nn.Module):
    """Predict future tubelet embeddings with learned positional queries."""

    def __init__(self, config: CausalVideoJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerDecoder(layer, num_layers=config.predictor_depth)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, context_tokens: Tensor) -> Tensor:
        batch = context_tokens.shape[0]
        side = self.config.spatial_tokens_per_side
        positions = make_video_position_embedding(
            self.config.target_time_tokens,
            side,
            side,
            self.config.embed_dim,
            time_offset=self.config.context_time_tokens,
            embedding_type=self.config.position_embedding_type,
        ).to(device=context_tokens.device, dtype=context_tokens.dtype)
        queries = self.mask_token.expand(batch, len(positions), -1) + positions.unsqueeze(0)
        predictions = self.blocks(tgt=queries, memory=context_tokens)
        return self.norm(predictions)


class CausalVideoJEPA(nn.Module):
    """EMA target encoder plus stop-gradient future feature prediction."""

    def __init__(self, config: CausalVideoJEPAConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.context_encoder = VideoTransformerEncoder(config)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = CausalFuturePredictor(config)
        self.target_encoder.requires_grad_(False)

    def train(self, mode: bool = True) -> "CausalVideoJEPA":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def encode_context(self, context: Tensor, *, return_tokens: bool = False) -> Tensor:
        tokens = self.context_encoder(context, time_offset=0)
        return tokens if return_tokens else tokens.mean(dim=1)

    def predict_from_context_tokens(self, context_tokens: Tensor) -> Tensor:
        return self.predictor(context_tokens)

    @torch.no_grad()
    def encode_target(self, target: Tensor) -> Tensor:
        return self.target_encoder(target, time_offset=self.config.context_time_tokens)

    @staticmethod
    def feature_prediction_loss(
        predictions: Tensor,
        targets: Tensor,
        token_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"prediction/target shape mismatch: {predictions.shape} vs {targets.shape}"
            )
        per_token = (predictions - targets).abs().mean(dim=-1)
        if token_weights is None:
            per_sample = per_token.mean(dim=-1)
        else:
            if token_weights.shape != per_token.shape:
                raise RuntimeError(
                    "token weight shape mismatch: "
                    f"{token_weights.shape} vs {per_token.shape}"
                )
            weights = token_weights.to(device=per_token.device, dtype=per_token.dtype)
            per_sample = (per_token * weights).sum(dim=-1) / weights.sum(
                dim=-1
            ).clamp_min(1e-8)
        return per_sample.mean(), per_sample, per_token

    def target_foreground_mask(self, target: Tensor) -> Tensor:
        """Identify target tubelets intersecting non-background pixels."""

        if target.ndim != 5:
            raise ValueError("target must have shape [B,T,C,H,W]")
        activity = F.max_pool3d(
            target.permute(0, 2, 1, 3, 4).contiguous(),
            kernel_size=(
                self.config.tubelet_size,
                self.config.patch_size,
                self.config.patch_size,
            ),
            stride=(
                self.config.tubelet_size,
                self.config.patch_size,
                self.config.patch_size,
            ),
        )
        activity = activity.flatten(2).amax(dim=1)
        return activity > self.config.foreground_threshold

    def target_motion_activity(self, context: Tensor, target: Tensor) -> Tensor:
        """Compute label-free pixel motion for each future target tubelet.

        The first target-frame difference is measured against the last context frame.
        Subsequent differences are measured between adjacent target frames. The result
        follows the exact temporal and spatial geometry of the target encoder tokens.
        """

        if context.ndim != 5 or target.ndim != 5:
            raise ValueError("context and target must have shape [B,T,C,H,W]")
        if context.shape[0] != target.shape[0] or context.shape[2:] != target.shape[2:]:
            raise ValueError("context and target batch/channel/spatial shapes must match")
        sequence = torch.cat([context[:, -1:], target], dim=1)
        frame_difference = (sequence[:, 1:] - sequence[:, :-1]).abs()
        pooled = F.avg_pool3d(
            frame_difference.permute(0, 2, 1, 3, 4).contiguous(),
            kernel_size=(
                self.config.tubelet_size,
                self.config.patch_size,
                self.config.patch_size,
            ),
            stride=(
                self.config.tubelet_size,
                self.config.patch_size,
                self.config.patch_size,
            ),
        )
        return pooled.mean(dim=1).flatten(1)

    def motion_token_weights(self, activity: Tensor) -> Tensor:
        if activity.ndim != 2:
            raise ValueError("motion activity must have shape [B,L]")
        epsilon = self.config.motion_weight_epsilon
        scale = activity.mean(dim=1, keepdim=True)
        normalized = activity / (scale + epsilon)
        weights = normalized.clamp(
            min=self.config.motion_weight_min,
            max=self.config.motion_weight_max,
        )
        if self.config.motion_weight_power != 1.0:
            weights = weights.pow(self.config.motion_weight_power)
        return weights / weights.mean(dim=1, keepdim=True).clamp_min(epsilon)

    @staticmethod
    def _masked_token_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(self, context: Tensor, target: Tensor) -> dict[str, Tensor]:
        context_tokens = self.context_encoder(context, time_offset=0)
        predictions = self.predict_from_context_tokens(context_tokens)
        with torch.no_grad():
            target_tokens = self.target_encoder(
                target,
                time_offset=self.config.context_time_tokens,
            )
            foreground_mask = self.target_foreground_mask(target)
            motion_activity = self.target_motion_activity(context, target)

        uniform_weights = torch.ones_like(motion_activity, dtype=predictions.dtype)
        foreground_weights = uniform_weights + foreground_mask.to(predictions.dtype) * (
            self.config.foreground_weight - 1.0
        )
        motion_weights = self.motion_token_weights(motion_activity).to(predictions.dtype)
        objective_weights = {
            "uniform": uniform_weights,
            "foreground": foreground_weights,
            "motion": motion_weights,
        }[self.config.target_weighting_mode]

        loss, per_sample_loss, per_token_loss = self.feature_prediction_loss(
            predictions,
            target_tokens,
            objective_weights,
        )
        unweighted_loss, per_sample_unweighted_loss, _ = self.feature_prediction_loss(
            predictions,
            target_tokens,
        )
        foreground_weighted_loss, per_sample_foreground_loss, _ = (
            self.feature_prediction_loss(
                predictions,
                target_tokens,
                foreground_weights,
            )
        )
        motion_weighted_loss, per_sample_motion_loss, _ = self.feature_prediction_loss(
            predictions,
            target_tokens,
            motion_weights,
        )
        foreground_loss = self._masked_token_mean(per_token_loss, foreground_mask)
        background_loss = self._masked_token_mean(per_token_loss, ~foreground_mask)
        active_threshold = motion_activity.mean(dim=1, keepdim=True)
        motion_active_mask = motion_activity > active_threshold
        motion_active_loss = self._masked_token_mean(per_token_loss, motion_active_mask)
        motion_inactive_loss = self._masked_token_mean(per_token_loss, ~motion_active_mask)

        return {
            "loss": loss,
            "unweighted_loss": unweighted_loss,
            "foreground_weighted_loss": foreground_weighted_loss,
            "motion_weighted_loss": motion_weighted_loss,
            "foreground_loss": foreground_loss,
            "background_loss": background_loss,
            "motion_active_loss": motion_active_loss,
            "motion_inactive_loss": motion_inactive_loss,
            "foreground_fraction": foreground_mask.to(torch.float32).mean(),
            "motion_active_fraction": motion_active_mask.to(torch.float32).mean(),
            "motion_activity_mean": motion_activity.mean(),
            "motion_weight_std": motion_weights.float().std(unbiased=False),
            "motion_weight_max": motion_weights.float().amax(),
            "per_sample_loss": per_sample_loss,
            "per_sample_unweighted_loss": per_sample_unweighted_loss,
            "per_sample_foreground_loss": per_sample_foreground_loss,
            "per_sample_motion_loss": per_sample_motion_loss,
            "per_token_loss": per_token_loss,
            "foreground_token_mask": foreground_mask,
            "motion_activity": motion_activity,
            "motion_token_weights": motion_weights,
            "context_tokens": context_tokens,
            "predicted_tokens": predictions,
            "target_tokens": target_tokens,
        }

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0,1]")
        for online, target in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
        for online, target in zip(
            self.context_encoder.buffers(),
            self.target_encoder.buffers(),
            strict=True,
        ):
            target.copy_(online)


def cosine_ema_momentum(
    step: int,
    total_steps: int,
    start: float,
    end: float,
) -> float:
    if total_steps <= 1:
        return end
    progress = min(max(step / (total_steps - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 - torch.cos(torch.tensor(progress * torch.pi)).item())
    return start + (end - start) * cosine
