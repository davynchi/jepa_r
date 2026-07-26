"""Frozen representation extraction, linear probes, and attentive token probes."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .video_diagnostics import representation_diagnostics
from .video_jepa import CausalVideoJEPA


@dataclass(frozen=True)
class ProbeConfig:
    classifier_epochs: int = 80
    classifier_batch_size: int = 256
    classifier_lr: float = 1e-2
    classifier_weight_decay: float = 1e-4
    ridge_lambdas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    attentive_epochs: int = 30
    attentive_batch_size: int = 128
    attentive_lr: float = 1e-3
    attentive_weight_decay: float = 1e-4
    attentive_heads: int = 4
    attentive_mlp_ratio: float = 2.0
    attentive_spatial_pool: int = 4
    raw_spatial_pool: int = 8
    seed: int = 0


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, features: Tensor) -> "FeatureNormalizer":
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def transform(self, features: Tensor) -> Tensor:
        return (features - self.mean) / self.std


@dataclass
class LinearClassifierResult:
    model: nn.Linear
    normalizer: FeatureNormalizer
    best_validation_loss: float

    def predict(self, features: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.model(self.normalizer.transform(features))
        return logits.argmax(dim=-1)


@dataclass(frozen=True)
class RidgeResult:
    weights: Tensor
    normalizer: FeatureNormalizer
    regularization: float

    def predict(self, features: Tensor) -> Tensor:
        normalized = self.normalizer.transform(features).to(self.weights.dtype)
        design = torch.cat(
            [normalized, torch.ones(len(normalized), 1, dtype=self.weights.dtype)],
            dim=1,
        )
        return design @ self.weights


class AttentiveTokenClassifier(nn.Module):
    """Cross-attention pooling followed by a small classification head."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        *,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("token embed_dim must be divisible by attentive_heads")
        hidden_dim = max(int(embed_dim * mlp_ratio), embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.token_norm(tokens)
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        pooled = query + pooled
        pooled = pooled + self.mlp(self.query_norm(pooled))
        return self.classifier(self.output_norm(pooled[:, 0]))


def _class_weights(labels: Tensor, num_classes: int) -> Tensor:
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float32)
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean().clamp_min(1e-8)


def fit_linear_classifier(
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    *,
    num_classes: int,
    config: ProbeConfig,
) -> LinearClassifierResult:
    torch.manual_seed(config.seed)
    normalizer = FeatureNormalizer.fit(train_features)
    train_x = normalizer.transform(train_features).to(torch.float32)
    validation_x = normalizer.transform(validation_features).to(torch.float32)
    train_y = train_labels.to(torch.long)
    validation_y = validation_labels.to(torch.long)

    model = nn.Linear(train_x.shape[1], num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.classifier_lr,
        weight_decay=config.classifier_weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_y, num_classes))
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.classifier_batch_size,
        shuffle=True,
        generator=generator,
    )

    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    for _ in range(config.classifier_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_y))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("classifier probe did not train")
    model.load_state_dict(best_state)
    model.eval()
    return LinearClassifierResult(
        model=model,
        normalizer=normalizer,
        best_validation_loss=best_loss,
    )


def _classification_loss(
    model: nn.Module,
    tokens: Tensor,
    labels: Tensor,
    criterion: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            end = min(start + batch_size, len(tokens))
            batch_tokens = tokens[start:end].to(device=device, dtype=torch.float32)
            batch_labels = labels[start:end].to(device=device, dtype=torch.long)
            loss = criterion(model(batch_tokens), batch_labels)
            total += float(loss) * len(batch_tokens)
            count += len(batch_tokens)
    return total / max(count, 1)


def fit_attentive_classifier(
    train_tokens: Tensor,
    train_labels: Tensor,
    validation_tokens: Tensor,
    validation_labels: Tensor,
    *,
    num_classes: int,
    config: ProbeConfig,
    device: torch.device,
) -> tuple[AttentiveTokenClassifier, float]:
    torch.manual_seed(config.seed)
    model = AttentiveTokenClassifier(
        train_tokens.shape[-1],
        num_classes,
        num_heads=config.attentive_heads,
        mlp_ratio=config.attentive_mlp_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.attentive_lr,
        weight_decay=config.attentive_weight_decay,
    )
    class_weights = _class_weights(train_labels.to(torch.long), num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train_tokens, train_labels.to(torch.long)),
        batch_size=config.attentive_batch_size,
        shuffle=True,
        generator=generator,
    )

    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    for _ in range(config.attentive_epochs):
        model.train()
        for batch_tokens, batch_labels in loader:
            batch_tokens = batch_tokens.to(device=device, dtype=torch.float32)
            batch_labels = batch_labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_tokens), batch_labels)
            loss.backward()
            optimizer.step()
        validation_loss = _classification_loss(
            model,
            validation_tokens,
            validation_labels,
            criterion,
            batch_size=config.attentive_batch_size,
            device=device,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("attentive classifier did not train")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


@torch.no_grad()
def predict_attentive_classifier(
    model: AttentiveTokenClassifier,
    tokens: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    predictions: list[Tensor] = []
    model.eval()
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start : start + batch_size].to(
            device=device,
            dtype=torch.float32,
        )
        predictions.append(model(batch).argmax(dim=-1).cpu())
    return torch.cat(predictions, dim=0)


def _augment_intercept(features: Tensor) -> Tensor:
    return torch.cat(
        [features, torch.ones(len(features), 1, dtype=features.dtype)],
        dim=1,
    )


def _ridge_weights(design: Tensor, targets: Tensor, regularization: float) -> Tensor:
    gram = design.T @ design
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    identity[-1, -1] = 0.0
    rhs = design.T @ targets
    return torch.linalg.solve(gram + regularization * identity, rhs)


def fit_ridge_regression(
    train_features: Tensor,
    train_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    regularizations: Iterable[float],
) -> RidgeResult:
    normalizer = FeatureNormalizer.fit(train_features)
    train_design = _augment_intercept(
        normalizer.transform(train_features).to(torch.float64)
    )
    validation_design = _augment_intercept(
        normalizer.transform(validation_features).to(torch.float64)
    )
    train_y = train_targets.to(torch.float64)
    validation_y = validation_targets.to(torch.float64)
    if train_y.ndim == 1:
        train_y = train_y[:, None]
        validation_y = validation_y[:, None]

    best_regularization: float | None = None
    best_weights: Tensor | None = None
    best_loss = float("inf")
    for regularization in regularizations:
        weights = _ridge_weights(train_design, train_y, float(regularization))
        prediction = validation_design @ weights
        loss = float((prediction - validation_y).square().mean())
        if loss < best_loss:
            best_loss = loss
            best_regularization = float(regularization)
            best_weights = weights
    if best_regularization is None or best_weights is None:
        raise ValueError("regularizations must contain at least one value")
    return RidgeResult(
        weights=best_weights,
        normalizer=normalizer,
        regularization=best_regularization,
    )


def classification_metrics(
    predictions: Tensor,
    labels: Tensor,
    num_classes: int,
) -> dict[str, float]:
    predictions = predictions.to(torch.long)
    labels = labels.to(torch.long)
    accuracy = float((predictions == labels).to(torch.float64).mean())
    f1_values: list[float] = []
    for class_index in range(num_classes):
        predicted = predictions == class_index
        actual = labels == class_index
        true_positive = int((predicted & actual).sum())
        false_positive = int((predicted & ~actual).sum())
        false_negative = int((~predicted & actual).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": accuracy,
        "macro_f1": float(sum(f1_values) / len(f1_values)),
    }


def spatially_pool_tokens(
    tokens: Tensor,
    *,
    time_tokens: int,
    spatial_tokens_per_side: int,
    output_side: int,
) -> Tensor:
    """Pool an ordered [B,T*H*W,D] token grid without removing time."""

    if output_side <= 0 or output_side > spatial_tokens_per_side:
        raise ValueError("attentive spatial pool must be in [1, spatial token side]")
    batch, token_count, embed_dim = tokens.shape
    expected = time_tokens * spatial_tokens_per_side * spatial_tokens_per_side
    if token_count != expected:
        raise ValueError(f"expected {expected} context tokens, got {token_count}")
    grid = tokens.reshape(
        batch,
        time_tokens,
        spatial_tokens_per_side,
        spatial_tokens_per_side,
        embed_dim,
    )
    grid = grid.permute(0, 1, 4, 2, 3).reshape(
        batch * time_tokens,
        embed_dim,
        spatial_tokens_per_side,
        spatial_tokens_per_side,
    )
    pooled = F.adaptive_avg_pool2d(grid, (output_side, output_side))
    pooled = pooled.reshape(batch, time_tokens, embed_dim, output_side, output_side)
    pooled = pooled.permute(0, 1, 3, 4, 2)
    return pooled.reshape(batch, time_tokens * output_side * output_side, embed_dim)


def temporal_pool_tokens(
    tokens: Tensor,
    *,
    time_tokens: int,
    spatial_tokens_per_side: int,
) -> Tensor:
    """Spatially average each time slice while preserving temporal order."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B,T*H*W,D]")
    batch, token_count, embed_dim = tokens.shape
    expected = time_tokens * spatial_tokens_per_side * spatial_tokens_per_side
    if token_count != expected:
        raise ValueError(f"expected {expected} tokens, got {token_count}")
    grid = tokens.reshape(
        batch,
        time_tokens,
        spatial_tokens_per_side,
        spatial_tokens_per_side,
        embed_dim,
    )
    return grid.mean(dim=(2, 3))


def raw_context_features(context: Tensor, output_side: int) -> dict[str, Tensor]:
    """Build compact raw-pixel controls without a large pixel-space Gram matrix."""

    if context.ndim != 5:
        raise ValueError("context must have shape [B,T,C,H,W]")
    if output_side <= 0:
        raise ValueError("raw spatial pool must be positive")
    batch, frames, channels, height, width = context.shape
    flattened = context.reshape(batch * frames, channels, height, width)
    pooled = F.adaptive_avg_pool2d(flattened, (output_side, output_side))
    pooled = pooled.reshape(batch, frames, channels * output_side * output_side)
    differences = (context[:, 1:] - context[:, :-1]).abs()
    difference_flat = differences.reshape(
        batch * max(frames - 1, 1), channels, height, width
    )
    difference_pooled = F.adaptive_avg_pool2d(
        difference_flat, (output_side, output_side)
    )
    difference_pooled = difference_pooled.reshape(
        batch, max(frames - 1, 1), channels * output_side * output_side
    )
    return {
        "raw_temporal_features": pooled.flatten(1),
        "raw_last_frame_features": pooled[:, -1],
        "raw_difference_features": difference_pooled.flatten(1),
    }


@torch.no_grad()
def extract_frozen_features(
    model: CausalVideoJEPA,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    include_attentive_tokens: bool = False,
    attentive_spatial_pool: int = 4,
    include_temporal_features: bool = False,
    include_raw_features: bool = False,
    include_shuffled_context_features: bool = False,
    raw_spatial_pool: int = 8,
) -> dict[str, Tensor]:
    model.eval()
    outputs: dict[str, list[Tensor]] = {
        "features": [],
        "direction_label": [],
        "speed": [],
        "boundary_velocity": [],
        "future_positions": [],
        "bounce_target": [],
        "digit_label": [],
        "sample_id": [],
    }
    if include_attentive_tokens:
        outputs["attentive_tokens"] = []
    if include_temporal_features:
        outputs["temporal_features"] = []
        outputs["last_token_features"] = []
    if include_shuffled_context_features:
        outputs["shuffled_temporal_features"] = []
    if include_raw_features:
        outputs["raw_temporal_features"] = []
        outputs["raw_last_frame_features"] = []
        outputs["raw_difference_features"] = []

    from .video_diagnostics import deterministic_time_permutation

    for batch in loader:
        context = batch["context"].to(device, non_blocking=True)
        tokens = model.encode_context(context, return_tokens=True)
        outputs["features"].append(tokens.mean(dim=1).float().cpu())
        temporal = temporal_pool_tokens(
            tokens,
            time_tokens=model.config.context_time_tokens,
            spatial_tokens_per_side=model.config.spatial_tokens_per_side,
        )
        if include_temporal_features:
            outputs["temporal_features"].append(temporal.flatten(1).float().cpu())
            outputs["last_token_features"].append(temporal[:, -1].float().cpu())
        if include_attentive_tokens:
            pooled_tokens = spatially_pool_tokens(
                tokens,
                time_tokens=model.config.context_time_tokens,
                spatial_tokens_per_side=model.config.spatial_tokens_per_side,
                output_side=attentive_spatial_pool,
            )
            outputs["attentive_tokens"].append(pooled_tokens.to(torch.float16).cpu())
        if include_shuffled_context_features:
            permutation = deterministic_time_permutation(context.shape[1]).to(context.device)
            shuffled = context.index_select(1, permutation)
            shuffled_tokens = model.encode_context(shuffled, return_tokens=True)
            shuffled_temporal = temporal_pool_tokens(
                shuffled_tokens,
                time_tokens=model.config.context_time_tokens,
                spatial_tokens_per_side=model.config.spatial_tokens_per_side,
            )
            outputs["shuffled_temporal_features"].append(
                shuffled_temporal.flatten(1).float().cpu()
            )
        if include_raw_features:
            for key, value in raw_context_features(
                batch["context"].to(torch.float32), raw_spatial_pool
            ).items():
                outputs[key].append(value.cpu())
        for key in [
            "direction_label",
            "speed",
            "boundary_velocity",
            "future_positions",
            "bounce_target",
            "digit_label",
            "sample_id",
        ]:
            outputs[key].append(batch[key].cpu())
    return {key: torch.cat(values, dim=0) for key, values in outputs.items()}


def evaluate_probe_suite(
    train: dict[str, Tensor],
    validation: dict[str, Tensor],
    evaluation: dict[str, Tensor],
    *,
    num_directions: int,
    config: ProbeConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train mean-pooled linear probes and score one evaluation split."""

    metrics: dict[str, float] = {}
    hyperparameters: dict[str, float] = {}

    direction_probe = fit_linear_classifier(
        train["features"],
        train["direction_label"],
        validation["features"],
        validation["direction_label"],
        num_classes=num_directions,
        config=config,
    )
    direction_prediction = direction_probe.predict(evaluation["features"])
    direction_metrics = classification_metrics(
        direction_prediction,
        evaluation["direction_label"],
        num_directions,
    )
    metrics["mean_direction_accuracy"] = direction_metrics["accuracy"]
    metrics["mean_direction_macro_f1"] = direction_metrics["macro_f1"]
    metrics["direction_accuracy"] = direction_metrics["accuracy"]
    metrics["direction_macro_f1"] = direction_metrics["macro_f1"]

    bounce_probe = fit_linear_classifier(
        train["features"],
        train["bounce_target"],
        validation["features"],
        validation["bounce_target"],
        num_classes=2,
        config=config,
    )
    bounce_prediction = bounce_probe.predict(evaluation["features"])
    bounce_metrics = classification_metrics(
        bounce_prediction,
        evaluation["bounce_target"],
        2,
    )
    metrics["bounce_accuracy"] = bounce_metrics["accuracy"]
    metrics["bounce_macro_f1"] = bounce_metrics["macro_f1"]

    digit_probe = fit_linear_classifier(
        train["features"],
        train["digit_label"],
        validation["features"],
        validation["digit_label"],
        num_classes=10,
        config=config,
    )
    digit_prediction = digit_probe.predict(evaluation["features"])
    digit_metrics = classification_metrics(
        digit_prediction,
        evaluation["digit_label"],
        10,
    )
    metrics["digit_accuracy"] = digit_metrics["accuracy"]

    speed_probe = fit_ridge_regression(
        train["features"],
        train["speed"],
        validation["features"],
        validation["speed"],
        regularizations=config.ridge_lambdas,
    )
    speed_prediction = speed_probe.predict(evaluation["features"]).squeeze(-1)
    metrics["speed_rmse"] = float(
        (speed_prediction - evaluation["speed"]).square().mean().sqrt()
    )
    hyperparameters["speed_ridge_lambda"] = speed_probe.regularization

    velocity_probe = fit_ridge_regression(
        train["features"],
        train["boundary_velocity"],
        validation["features"],
        validation["boundary_velocity"],
        regularizations=config.ridge_lambdas,
    )
    velocity_prediction = velocity_probe.predict(evaluation["features"])
    velocity_error = velocity_prediction - evaluation["boundary_velocity"]
    metrics["velocity_vector_rmse"] = float(
        velocity_error.square().sum(dim=-1).mean().sqrt()
    )
    hyperparameters["velocity_ridge_lambda"] = velocity_probe.regularization

    train_trajectory = train["future_positions"].flatten(1)
    validation_trajectory = validation["future_positions"].flatten(1)
    trajectory_probe = fit_ridge_regression(
        train["features"],
        train_trajectory,
        validation["features"],
        validation_trajectory,
        regularizations=config.ridge_lambdas,
    )
    future_prediction = trajectory_probe.predict(evaluation["features"])
    future_prediction = future_prediction.reshape_as(evaluation["future_positions"])
    distances = (
        (future_prediction - evaluation["future_positions"])
        .square()
        .sum(dim=-1)
        .sqrt()
    )
    metrics["future_ade"] = float(distances.mean())
    metrics["future_fde"] = float(distances[:, -1].mean())
    hyperparameters["trajectory_ridge_lambda"] = trajectory_probe.regularization

    metrics.update(representation_diagnostics(evaluation["features"]))
    for key, value in metrics.items():
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite probe metric {key}: {value}")
    return metrics, hyperparameters


def evaluate_attentive_classification_probes(
    train: dict[str, Tensor],
    validation: dict[str, Tensor],
    evaluation: dict[str, Tensor],
    *,
    num_directions: int,
    tasks: Sequence[str],
    config: ProbeConfig,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train supervised attentive pooling probes over ordered context tokens."""

    for split in [train, validation, evaluation]:
        if "attentive_tokens" not in split:
            raise KeyError("attentive_tokens were not extracted")

    specifications = {
        "direction": ("direction_label", num_directions),
        "bounce": ("bounce_target", 2),
        "digit": ("digit_label", 10),
    }
    metrics: dict[str, float] = {}
    hyperparameters: dict[str, float] = {}
    for task in tasks:
        if task not in specifications:
            raise ValueError(f"unknown attentive probe task: {task}")
        label_key, num_classes = specifications[task]
        model, best_loss = fit_attentive_classifier(
            train["attentive_tokens"],
            train[label_key],
            validation["attentive_tokens"],
            validation[label_key],
            num_classes=num_classes,
            config=config,
            device=device,
        )
        predictions = predict_attentive_classifier(
            model,
            evaluation["attentive_tokens"],
            batch_size=config.attentive_batch_size,
            device=device,
        )
        values = classification_metrics(
            predictions,
            evaluation[label_key],
            num_classes,
        )
        metrics[f"attentive_{task}_accuracy"] = values["accuracy"]
        metrics[f"attentive_{task}_macro_f1"] = values["macro_f1"]
        hyperparameters[f"attentive_{task}_best_validation_loss"] = best_loss

    if "attentive_direction_accuracy" in metrics:
        metrics["direction_accuracy"] = metrics["attentive_direction_accuracy"]
        metrics["direction_macro_f1"] = metrics["attentive_direction_macro_f1"]
    return metrics, hyperparameters


def stratified_subset_indices(labels: Tensor, budget: int, seed: int) -> Tensor:
    """Choose a deterministic approximately class-balanced subset."""

    labels = labels.to(torch.long).cpu()
    if budget <= 0 or budget > len(labels):
        raise ValueError("budget must be in [1, number of samples]")
    classes = torch.unique(labels, sorted=True)
    generator = torch.Generator().manual_seed(seed)
    base, remainder = divmod(budget, len(classes))
    selected: list[Tensor] = []
    for class_offset, class_index in enumerate(classes):
        candidates = torch.nonzero(labels == class_index, as_tuple=False).flatten()
        take = base + int(class_offset < remainder)
        if take > len(candidates):
            raise ValueError(
                f"not enough samples for class {int(class_index)}: {len(candidates)} < {take}"
            )
        permutation = torch.randperm(len(candidates), generator=generator)
        selected.append(candidates[permutation[:take]])
    indices = torch.cat(selected)
    return indices[torch.randperm(len(indices), generator=generator)]


def _direction_from_velocity(velocity: Tensor, num_directions: int) -> Tensor:
    angles = torch.atan2(velocity[:, 1], velocity[:, 0]).remainder(2.0 * torch.pi)
    scaled = angles * num_directions / (2.0 * torch.pi)
    return torch.floor(scaled + 0.5).to(torch.long).remainder(num_directions)


def evaluate_low_shot_feature_variant(
    train: dict[str, Tensor],
    validation: dict[str, Tensor],
    evaluation: dict[str, Tensor],
    *,
    feature_key: str,
    budget: int,
    split_seed: int,
    num_directions: int,
    config: ProbeConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate a deterministic low-shot linear/ridge protocol."""

    for split in (train, validation, evaluation):
        if feature_key not in split:
            raise KeyError(f"feature key {feature_key!r} is missing")
    indices = stratified_subset_indices(train["direction_label"], budget, split_seed)
    train_features = train[feature_key][indices]
    train_direction = train["direction_label"][indices]
    local_config = replace(config, seed=split_seed)

    direction_probe = fit_linear_classifier(
        train_features,
        train_direction,
        validation[feature_key],
        validation["direction_label"],
        num_classes=num_directions,
        config=local_config,
    )
    direction_prediction = direction_probe.predict(evaluation[feature_key])
    direction_values = classification_metrics(
        direction_prediction, evaluation["direction_label"], num_directions
    )

    velocity_probe = fit_ridge_regression(
        train_features,
        train["boundary_velocity"][indices],
        validation[feature_key],
        validation["boundary_velocity"],
        regularizations=config.ridge_lambdas,
    )
    velocity_prediction = velocity_probe.predict(evaluation[feature_key])
    velocity_error = velocity_prediction - evaluation["boundary_velocity"]

    trajectory_probe = fit_ridge_regression(
        train_features,
        train["future_positions"][indices].flatten(1),
        validation[feature_key],
        validation["future_positions"].flatten(1),
        regularizations=config.ridge_lambdas,
    )
    future_prediction = trajectory_probe.predict(evaluation[feature_key])
    future_prediction = future_prediction.reshape_as(evaluation["future_positions"])
    distances = (
        (future_prediction - evaluation["future_positions"])
        .square()
        .sum(dim=-1)
        .sqrt()
    )
    velocity_direction = _direction_from_velocity(velocity_prediction, num_directions)
    metrics = {
        "direction_accuracy": direction_values["accuracy"],
        "direction_macro_f1": direction_values["macro_f1"],
        "velocity_vector_rmse": float(
            velocity_error.square().sum(dim=-1).mean().sqrt()
        ),
        "velocity_direction_accuracy": float(
            (velocity_direction == evaluation["direction_label"])
            .to(torch.float64)
            .mean()
        ),
        "future_ade": float(distances.mean()),
        "future_fde": float(distances[:, -1].mean()),
    }
    hyperparameters = {
        "velocity_ridge_lambda": velocity_probe.regularization,
        "trajectory_ridge_lambda": trajectory_probe.regularization,
        "direction_best_validation_loss": direction_probe.best_validation_loss,
    }
    return metrics, hyperparameters
