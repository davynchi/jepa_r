from __future__ import annotations

import torch
from torch import nn

from jepa.training.images.ijepa_spatial import (
    MaskConfig,
    SpatialIJEPACore,
    build_spatial_ijepa_core,
)
from jepa.training.images.spatial_curriculum import (
    adamw_update_direction,
    richness_from_images,
    score_frames_by_ras,
)


class _PatchMeanEncoder(nn.Module):
    def __init__(self, *, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(3, embed_dim, bias=False)

    def forward(
        self,
        images: torch.Tensor,
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.mean(dim=(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2)
        encoded = self.projection(patches)
        if masks is None:
            return encoded
        return torch.cat(
            [
                torch.gather(
                    encoded,
                    dim=1,
                    index=mask.to(encoded.device).unsqueeze(-1).expand(-1, -1, encoded.shape[-1]),
                )
                for mask in masks
            ],
            dim=0,
        )


class _DeviceCheckingEncoder(_PatchMeanEncoder):
    def forward(
        self,
        images: torch.Tensor,
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if masks is not None:
            assert all(mask.device == images.device for mask in masks)
        return super().forward(images, masks)


def _fake_core() -> SpatialIJEPACore:
    encoder = _PatchMeanEncoder(patch_size=2, embed_dim=3)
    return SpatialIJEPACore(
        context_encoder=encoder,
        predictor=nn.Identity(),
        target_encoder=encoder,
        policy=None,  # type: ignore[arg-type]
        model_name="vit_tiny",
        image_size=16,
        patch_size=2,
        embed_dim=3,
    )


def _device_checking_core() -> SpatialIJEPACore:
    core = _fake_core()
    encoder = _DeviceCheckingEncoder(patch_size=2, embed_dim=3)
    core.context_encoder = encoder
    core.target_encoder = encoder
    return core


def _mask_config() -> MaskConfig:
    return MaskConfig(
        enc_mask_scale=(0.5, 0.7),
        pred_mask_scale=(0.1, 0.2),
        num_enc_masks=1,
        num_pred_masks=1,
        min_keep=0,
    )


def test_adamw_update_direction_matches_optimizer_step_without_mutating_state() -> None:
    parameter = nn.Parameter(torch.tensor([0.25, -0.75], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=0.03,
        betas=(0.8, 0.95),
        eps=1.0e-7,
        weight_decay=0.1,
        amsgrad=True,
    )
    parameter.grad = torch.tensor([0.2, -0.4], dtype=torch.float64)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    gradient = torch.tensor([-0.3, 0.7], dtype=torch.float64)
    state_before = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[parameter].items()
    }
    predicted = adamw_update_direction(
        (parameter,),
        (gradient,),
        optimizer,
    )[0]

    assert predicted is not None
    for key, value in optimizer.state[parameter].items():
        expected = state_before[key]
        if torch.is_tensor(value):
            assert torch.equal(value, expected)
        else:
            assert value == expected

    previous = parameter.detach().clone()
    parameter.grad = gradient.clone()
    optimizer.step()
    actual = parameter.detach() - previous
    assert torch.allclose(predicted, actual, atol=1.0e-12, rtol=1.0e-10)


def test_optimizer_aware_ras_runs_through_batch_scoring() -> None:
    core = build_spatial_ijepa_core(
        "vit_tiny",
        image_size=16,
        patch_size=4,
        predictor_embed_dim=24,
        predictor_depth=1,
    )
    optimizer = torch.optim.AdamW(
        tuple(core.context_encoder.parameters()) + tuple(core.predictor.parameters()),
        lr=1.0e-3,
        weight_decay=0.05,
    )
    images = torch.randn(4, 3, 16, 16)
    scores, metadata = score_frames_by_ras(
        core,
        images,
        ref_indices=torch.arange(4),
        grid=4,
        mask_config=_mask_config(),
        batch_size=2,
        seed=13,
        device=torch.device("cpu"),
        richness_functional="predictive-barlow",
        richness_delta=1.0e-4,
        richness_trace_target=1.0,
        richness_trace_beta=0.0,
        score_granularity="batch",
        alignment="adamw-cosine",
        optimizer=optimizer,
    )

    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert metadata["ras/alignment_cosine"] == 1.0
    assert metadata["ras/alignment_adamw"] == 1.0


def test_predictive_barlow_prefers_spatially_stable_signal_to_pixel_noise() -> None:
    generator = torch.Generator().manual_seed(11)
    sample_colors = torch.randn(32, 3, 1, 1, generator=generator)
    stable_images = sample_colors.expand(-1, -1, 16, 16).clone()
    noisy_images = torch.randn(32, 3, 16, 16, generator=generator)
    core = _fake_core()

    stable_richness, stable_metadata = richness_from_images(
        core,
        stable_images,
        functional="predictive-barlow",
        delta=1.0e-4,
        trace_target=1.0,
        trace_beta=0.0,
        grid=8,
        mask_config=_mask_config(),
        mask_seed=7,
        predictive_redundancy_weight=0.0,
    )
    noisy_richness, noisy_metadata = richness_from_images(
        core,
        noisy_images,
        functional="predictive-barlow",
        delta=1.0e-4,
        trace_target=1.0,
        trace_beta=0.0,
        grid=8,
        mask_config=_mask_config(),
        mask_seed=7,
        predictive_redundancy_weight=0.0,
    )

    assert stable_richness > noisy_richness
    assert stable_metadata["ras/predictive_diag_mean"] > 0.99
    assert stable_metadata["ras/predictive_diag_mean"] > (
        noisy_metadata["ras/predictive_diag_mean"] + 0.1
    )
    stable_richness.backward()
    assert core.context_encoder.projection.weight.grad is not None
    assert torch.isfinite(core.context_encoder.projection.weight.grad).all()


def test_predictive_barlow_runs_through_batch_ras() -> None:
    core = build_spatial_ijepa_core(
        "vit_tiny",
        image_size=16,
        patch_size=4,
        predictor_embed_dim=24,
        predictor_depth=1,
    )
    images = torch.randn(4, 3, 16, 16)
    scores, metadata = score_frames_by_ras(
        core,
        images,
        ref_indices=torch.arange(4),
        grid=4,
        mask_config=_mask_config(),
        batch_size=2,
        seed=17,
        device=torch.device("cpu"),
        richness_functional="predictive-barlow",
        richness_delta=1.0e-4,
        richness_trace_target=1.0,
        richness_trace_beta=0.0,
        predictive_redundancy_weight=0.005,
        score_granularity="batch",
    )

    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(torch.tensor(metadata["ras/grad_richness_norm"]))
    assert "ras/predictive_invariance_loss" in metadata


def test_predictive_barlow_moves_sampled_masks_to_image_device() -> None:
    richness, _ = richness_from_images(
        _device_checking_core(),
        torch.randn(8, 3, 16, 16),
        functional="predictive-barlow",
        delta=1.0e-4,
        trace_target=1.0,
        trace_beta=0.0,
        grid=8,
        mask_config=_mask_config(),
        mask_seed=9,
    )

    assert torch.isfinite(richness)


def test_predictive_spectral_is_finite_and_differentiable() -> None:
    core = build_spatial_ijepa_core(
        "vit_tiny",
        image_size=16,
        patch_size=4,
        predictor_embed_dim=24,
        predictor_depth=1,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        richness, metadata = richness_from_images(
            core,
            torch.randn(8, 3, 16, 16),
            functional="predictive-spectral",
            delta=1.0e-3,
            trace_target=1.0,
            trace_beta=0.0,
            grid=4,
            mask_config=_mask_config(),
            mask_seed=19,
            predictive_kappa=1.0,
        )

    assert torch.isfinite(richness)
    assert metadata["ras/predictive_spectral_energy"] >= 0
    assert metadata["ras/predictive_spectral_effective_rank"] >= 1
    gradients = torch.autograd.grad(
        richness,
        tuple(
            parameter
            for parameter in core.context_encoder.parameters()
            if parameter.requires_grad
        ),
        allow_unused=True,
    )
    finite_gradients = [
        gradient for gradient in gradients if gradient is not None and gradient.numel() > 0
    ]
    assert finite_gradients
    assert all(torch.isfinite(gradient).all() for gradient in finite_gradients)


def test_predictive_spectral_runs_through_batch_ras() -> None:
    core = build_spatial_ijepa_core(
        "vit_tiny",
        image_size=16,
        patch_size=4,
        predictor_embed_dim=24,
        predictor_depth=1,
    )
    images = torch.randn(4, 3, 16, 16)
    scores, metadata = score_frames_by_ras(
        core,
        images,
        ref_indices=torch.arange(4),
        grid=4,
        mask_config=_mask_config(),
        batch_size=2,
        seed=23,
        device=torch.device("cpu"),
        richness_functional="predictive-spectral",
        richness_delta=1.0e-3,
        richness_trace_target=1.0,
        richness_trace_beta=0.0,
        predictive_kappa=1.0,
        score_granularity="batch",
        alignment="cosine",
    )

    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert metadata["ras/alignment_cosine"] == 1.0
    assert "ras/predictive_spectral_effective_rank" in metadata


def test_new_predictive_richness_variants_are_finite_and_differentiable() -> None:
    for functional in (
        "predictive-covariance",
        "predictive-energy",
        "predictive-dimension",
        "predictive-combined",
    ):
        core = build_spatial_ijepa_core(
            "vit_tiny",
            image_size=16,
            patch_size=4,
            predictor_embed_dim=24,
            predictor_depth=1,
        )
        richness, metadata = richness_from_images(
            core,
            torch.randn(8, 3, 16, 16),
            functional=functional,
            delta=1.0e-3,
            trace_target=1.0,
            trace_beta=0.0,
            grid=4,
            mask_config=_mask_config(),
            mask_seed=29,
            predictive_kappa=1.0,
        )

        assert torch.isfinite(richness), functional
        assert torch.isfinite(torch.tensor(metadata["ras/richness_value"])), functional
        assert "ras/predictive_covariance_logdet" in metadata
        assert "ras/predictive_spectral_combined" in metadata
        gradients = torch.autograd.grad(
            richness,
            tuple(
                parameter
                for parameter in core.context_encoder.parameters()
                if parameter.requires_grad
            ),
            allow_unused=True,
        )
        finite_gradients = [
            gradient
            for gradient in gradients
            if gradient is not None and gradient.numel() > 0
        ]
        assert finite_gradients, functional
        assert all(torch.isfinite(gradient).all() for gradient in finite_gradients), functional


def test_predictive_combined_runs_through_batch_ras() -> None:
    core = build_spatial_ijepa_core(
        "vit_tiny",
        image_size=16,
        patch_size=4,
        predictor_embed_dim=24,
        predictor_depth=1,
    )
    images = torch.randn(4, 3, 16, 16)
    scores, metadata = score_frames_by_ras(
        core,
        images,
        ref_indices=torch.arange(4),
        grid=4,
        mask_config=_mask_config(),
        batch_size=2,
        seed=31,
        device=torch.device("cpu"),
        richness_functional="predictive-combined",
        richness_delta=1.0e-3,
        richness_trace_target=1.0,
        richness_trace_beta=0.0,
        predictive_kappa=1.0,
        score_granularity="batch",
        alignment="cosine",
    )

    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert "ras/predictive_spectral_combined" in metadata
