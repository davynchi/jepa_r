"""Representation geometry and context-sensitivity diagnostics for Video-JEPA."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from .video_jepa import CausalVideoJEPA


def representation_diagnostics(features: Tensor) -> dict[str, float]:
    """Compute collapse diagnostics for a matrix of sample features [N,D]."""

    features = features.to(torch.float64)
    if features.ndim != 2:
        raise ValueError("features must have shape [N,D]")
    if len(features) == 0:
        raise ValueError("features cannot be empty")

    centered = features - features.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    singular_values = eigenvalues.sqrt()
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    maximum_rank = min(features.shape[0] - 1, features.shape[1])
    return {
        "effective_rank": float(effective_rank),
        "effective_rank_fraction": float(effective_rank / max(maximum_rank, 1)),
        "latent_std_mean": float(std.mean()),
        "latent_std_min": float(std.min()),
        "dead_dimension_fraction": float((std < 1e-3).to(torch.float64).mean()),
        "covariance_offdiag_rms": float(off_diagonal.square().mean().sqrt()),
        "feature_norm_mean": float(features.norm(dim=1).mean()),
    }


def _prefix(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def token_sequence_diagnostics(tokens: Tensor, prefix: str) -> dict[str, float]:
    """Measure pooled and position-controlled token geometry for [N,L,D]."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [N,L,D]")
    tokens = tokens.to(torch.float64)
    pooled = tokens.mean(dim=1)
    position_residual = tokens - tokens.mean(dim=0, keepdim=True)
    residual_matrix = position_residual.reshape(-1, tokens.shape[-1])

    result = _prefix(representation_diagnostics(pooled), f"{prefix}_pooled")
    result.update(
        _prefix(representation_diagnostics(residual_matrix), f"{prefix}_token")
    )
    result[f"{prefix}_token_variance_across_samples"] = float(
        position_residual.square().mean()
    )
    return result


def deterministic_time_permutation(frame_count: int) -> Tensor:
    """Return a stable non-identity permutation for matched temporal controls."""

    if frame_count <= 1:
        return torch.arange(frame_count)
    generator = torch.Generator().manual_seed(17_071 + 97 * frame_count)
    permutation = torch.randperm(frame_count, generator=generator)
    identity = torch.arange(frame_count)
    if torch.equal(permutation, identity) or torch.equal(permutation, identity.flip(0)):
        permutation = identity.roll(1)
    return permutation


def _group_roll_indices(groups: Tensor) -> tuple[Tensor, Tensor]:
    """Map each member to another member of the same group when possible."""

    groups = groups.detach().cpu().reshape(-1)
    indices = torch.arange(len(groups), dtype=torch.long)
    valid = torch.zeros(len(groups), dtype=torch.bool)
    for value in torch.unique(groups, sorted=True):
        members = torch.nonzero(groups == value, as_tuple=False).flatten()
        if len(members) < 2:
            continue
        indices[members] = members.roll(1)
        valid[members] = True
    return indices, valid


def _masked_with_nan(values: Tensor, valid: Tensor) -> Tensor:
    valid = valid.to(device=values.device)
    return torch.where(valid, values, torch.full_like(values, float("nan")))


@torch.no_grad()
def pairing_context_batch(
    model: CausalVideoJEPA,
    context: Tensor,
    target: Tensor,
    *,
    digit_labels: Tensor | None = None,
    direction_labels: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return matched and coarse counterfactual JEPA losses.

    Matched temporal controls keep the same frames and alter only their order or
    temporal information. Grouped controls exchange contexts only between clips
    with the same digit identity or the same direction label; singleton groups
    are marked with NaN and excluded from aggregation.
    """

    context_tokens = model.encode_context(context, return_tokens=True)
    target_tokens = model.encode_target(target)
    correct_predictions = model.predict_from_context_tokens(context_tokens)
    _, correct_loss, _ = model.feature_prediction_loss(
        correct_predictions,
        target_tokens,
    )

    batch_size = len(context)
    if batch_size > 1:
        global_indices = torch.arange(batch_size, device=context.device).roll(1)
    else:
        global_indices = torch.arange(batch_size, device=context.device)

    context_permutation = deterministic_time_permutation(context.shape[1]).to(
        context.device
    )
    target_permutation = deterministic_time_permutation(target.shape[1]).to(
        target.device
    )
    transformed_contexts = {
        "zero_context": torch.zeros_like(context),
        "reversed_context": context.flip(dims=(1,)),
        "shuffled_time_context": context.index_select(1, context_permutation),
        "last_frame_context": context[:, -1:].expand_as(context),
    }
    transformed_predictions = {
        name: model.predict_from_context_tokens(
            model.encode_context(value, return_tokens=True)
        )
        for name, value in transformed_contexts.items()
    }

    variants: dict[str, tuple[Tensor, Tensor]] = {
        "correct": (correct_predictions, target_tokens),
        "shuffled_context": (
            model.predict_from_context_tokens(context_tokens.index_select(0, global_indices)),
            target_tokens,
        ),
        "shuffled_target": (
            correct_predictions,
            target_tokens.index_select(0, global_indices),
        ),
        "reversed_target": (
            correct_predictions,
            model.encode_target(target.flip(dims=(1,))),
        ),
        "shuffled_time_target": (
            correct_predictions,
            model.encode_target(target.index_select(1, target_permutation)),
        ),
    }
    variants.update(
        {
            name: (prediction, target_tokens)
            for name, prediction in transformed_predictions.items()
        }
    )

    losses: dict[str, Tensor] = {}
    for name, (predictions, targets) in variants.items():
        _, per_sample, _ = model.feature_prediction_loss(predictions, targets)
        losses[name] = per_sample

    if digit_labels is not None:
        indices, valid = _group_roll_indices(digit_labels)
        indices = indices.to(context_tokens.device)
        valid = valid.to(context_tokens.device)
        predictions = model.predict_from_context_tokens(
            context_tokens.index_select(0, indices)
        )
        _, values, _ = model.feature_prediction_loss(predictions, target_tokens)
        losses["correct_same_digit"] = _masked_with_nan(correct_loss, valid)
        losses["same_digit_shuffled_context"] = _masked_with_nan(values, valid)

    if direction_labels is not None:
        indices, valid = _group_roll_indices(direction_labels)
        indices = indices.to(context_tokens.device)
        valid = valid.to(context_tokens.device)
        predictions = model.predict_from_context_tokens(
            context_tokens.index_select(0, indices)
        )
        _, values, _ = model.feature_prediction_loss(predictions, target_tokens)
        losses["correct_same_direction"] = _masked_with_nan(correct_loss, valid)
        losses["same_direction_shuffled_context"] = _masked_with_nan(values, valid)

    return losses


def accumulate_pairing_losses(
    sums: dict[str, float],
    counts: dict[str, int],
    losses: Mapping[str, Tensor],
) -> None:
    for name, values in losses.items():
        finite = torch.isfinite(values)
        if not finite.any():
            continue
        sums[name] = sums.get(name, 0.0) + float(values[finite].float().sum())
        counts[name] = counts.get(name, 0) + int(finite.sum())


def finalize_pairing_metrics(
    sums: Mapping[str, float],
    counts: Mapping[str, int] | int,
) -> dict[str, float]:
    if isinstance(counts, int):
        count_lookup = {name: counts for name in sums}
    else:
        count_lookup = dict(counts)

    means: dict[str, float] = {}
    for name, value in sums.items():
        count = int(count_lookup.get(name, 0))
        if count > 0:
            means[f"pairing_loss_{name}"] = value / count
    if "pairing_loss_correct" not in means:
        return means

    references = {
        "shuffled_context": "correct",
        "zero_context": "correct",
        "reversed_context": "correct",
        "shuffled_time_context": "correct",
        "last_frame_context": "correct",
        "shuffled_target": "correct",
        "reversed_target": "correct",
        "shuffled_time_target": "correct",
        "same_digit_shuffled_context": "correct_same_digit",
        "same_direction_shuffled_context": "correct_same_direction",
    }
    for name, reference in references.items():
        key = f"pairing_loss_{name}"
        reference_key = f"pairing_loss_{reference}"
        if key not in means or reference_key not in means:
            continue
        loss = means[key]
        correct = means[reference_key]
        means[f"pairing_gap_{name}"] = loss - correct
        means[f"pairing_relative_gap_{name}"] = (
            loss - correct
        ) / max(abs(correct), 1e-12)
        means[f"pairing_ratio_{name}"] = loss / max(correct, 1e-12)

    for key, value in means.items():
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite pairing metric {key}: {value}")
    return means
