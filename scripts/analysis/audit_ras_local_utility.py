#!/usr/bin/env python3
"""Audit whether batch RAS predicts realized richness and frozen-probe utility."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jepa.analysis.subspace import fit_entity_classifier  # noqa: E402
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    build_tiny_imagenet_static_dataset_splits,
)
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    MaskConfig,
    build_spatial_ijepa_core_from_metadata,
    load_spatial_checkpoint,
    normalize_ijepa_images,
    sample_masks,
    spatial_ijepa_loss,
)
from jepa.training.images.spatial_curriculum import (  # noqa: E402
    adamw_update_direction,
    richness_from_images,
)

RICHNESS_FUNCTIONALS = (
    "predictive-spectral",
    "predictive-barlow",
    "predictive-covariance",
    "predictive-energy",
    "predictive-dimension",
    "predictive-combined",
)
PROBE_MODES = ("ridge", "linear", "linear-l2", "mlp")
FUNCTIONAL_SLUG = {
    functional: functional.removeprefix("predictive-").replace("-", "_")
    for functional in RICHNESS_FUNCTIONALS
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="epoch_0700.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--candidate-batch-size", type=int, default=128)
    parser.add_argument(
        "--mask-repeats",
        type=int,
        default=1,
        help="Repeat every candidate with independently sampled JEPA masks.",
    )
    parser.add_argument("--virtual-steps", type=int, default=3)
    parser.add_argument(
        "--step-multiplier",
        type=float,
        default=1.0,
        help="Multiply every virtual optimizer learning rate by this factor.",
    )
    parser.add_argument(
        "--virtual-optimizer",
        choices=("adamw", "sgd-small"),
        default="adamw",
        help="Optimizer used for realized post-update utility.",
    )
    parser.add_argument(
        "--sgd-learning-rate",
        type=float,
        default=1.0e-5,
        help="Learning rate for --virtual-optimizer sgd-small.",
    )
    parser.add_argument(
        "--reuse-score-masks-first-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the exact masks from RAS scoring for the first virtual update.",
    )
    parser.add_argument(
        "--virtual-ema-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the training EMA target update after each virtual step.",
    )
    parser.add_argument("--reference-size", type=int, default=1024)
    parser.add_argument("--probe-train-size", type=int, default=5000)
    parser.add_argument("--probe-test-size", type=int, default=1000)
    parser.add_argument("--probe-validation-size", type=int, default=1000)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--probe-learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0)
    parser.add_argument("--probe-ridge", type=float, default=1.0e-6)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--mlp-hidden-dim", type=int, default=1024)
    parser.add_argument("--mlp-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--probe-modes",
        nargs="+",
        choices=PROBE_MODES,
        default=PROBE_MODES,
    )
    parser.add_argument(
        "--probe-encoders",
        nargs="+",
        choices=("context", "target"),
        default=("context", "target"),
    )
    parser.add_argument("--encode-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "none", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--richness-delta", type=float, default=1.0e-3)
    parser.add_argument("--predictive-kappa", type=float, default=1.0)
    parser.add_argument("--predictive-redundancy-weight", type=float, default=0.005)
    parser.add_argument(
        "--richness-functionals",
        nargs="+",
        choices=RICHNESS_FUNCTIONALS,
        default=("predictive-spectral", "predictive-barlow"),
        help="Richness functionals scored and measured in the audit.",
    )
    return parser.parse_args()


def resolve_amp_dtype(value: str, spatial_config: dict[str, Any]) -> torch.dtype | None:
    if value == "auto":
        value = str(spatial_config.get("amp_dtype", ""))
        if not value:
            value = "bfloat16" if spatial_config.get("bfloat16", False) else "none"
    return {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype or torch.float32,
        enabled=dtype is not None and device.type == "cuda",
    )


def encode_pooled(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    chunks = []
    encoder.eval()
    with torch.inference_mode():
        for batch in images.split(batch_size):
            with autocast(device, amp_dtype):
                chunks.append(
                    encoder(batch.to(device, non_blocking=True)).mean(dim=1).float().cpu()
                )
    return torch.cat(chunks)


class MLPProbe(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(slots=True)
class FrozenProbe:
    name: str
    encoder_name: str
    normalize: bool
    head: torch.nn.Module | None = None
    ridge_classifier: Any | None = None


def probe_metrics(
    probe: FrozenProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    if probe.normalize:
        features = F.normalize(features, dim=-1)
    if probe.ridge_classifier is not None:
        scores = probe.ridge_classifier.probe.predict(features).float()
    else:
        assert probe.head is not None
        probe.head.eval()
        scores = []
        with torch.inference_mode():
            for batch in features.split(batch_size):
                scores.append(
                    probe.head(batch.to(device, non_blocking=True)).float().cpu()
                )
        scores = torch.cat(scores)
    loss = F.cross_entropy(scores, labels).item()
    accuracy = scores.argmax(dim=-1).eq(labels).float().mean().item()
    return loss, accuracy


def train_neural_probe(
    head: torch.nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> torch.nn.Module:
    del num_classes
    head.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_state = None
    evaluation_probe = FrozenProbe(
        name="training",
        encoder_name="",
        normalize=False,
        head=head,
    )
    for _ in range(epochs):
        head.train()
        order = torch.randperm(len(train_features), generator=generator)
        for indices in order.split(args.probe_batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_features[indices].to(device, non_blocking=True))
            loss = F.cross_entropy(
                logits,
                train_labels[indices].to(device, non_blocking=True),
            )
            loss.backward()
            optimizer.step()
        validation_loss, _ = probe_metrics(
            evaluation_probe,
            validation_features,
            validation_labels,
            device=device,
            batch_size=args.probe_batch_size,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
    assert best_state is not None
    head.load_state_dict(best_state)
    head.requires_grad_(False)
    return head


def build_probe_matrix(
    encoded: dict[str, dict[str, torch.Tensor]],
    labels: dict[str, torch.Tensor],
    *,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, FrozenProbe]:
    probes = {}
    for encoder_name in args.probe_encoders:
        features = encoded[encoder_name]
        for mode in args.probe_modes:
            name = f"{encoder_name}_{mode.replace('-', '_')}"
            normalize = mode == "linear-l2"
            train_features = (
                F.normalize(features["train"], dim=-1)
                if normalize
                else features["train"]
            )
            validation_features = (
                F.normalize(features["validation"], dim=-1)
                if normalize
                else features["validation"]
            )
            if mode == "ridge":
                classifier = fit_entity_classifier(
                    train_features,
                    labels["train"],
                    num_classes,
                    ridge=args.probe_ridge,
                )
                probes[name] = FrozenProbe(
                    name=name,
                    encoder_name=encoder_name,
                    normalize=False,
                    ridge_classifier=classifier,
                )
                continue
            if mode == "mlp":
                head: torch.nn.Module = MLPProbe(
                    train_features.shape[1],
                    args.mlp_hidden_dim,
                    num_classes,
                )
                epochs = args.mlp_epochs
                learning_rate = args.mlp_learning_rate
                weight_decay = args.mlp_weight_decay
            else:
                head = torch.nn.Linear(train_features.shape[1], num_classes)
                epochs = args.probe_epochs
                learning_rate = args.probe_learning_rate
                weight_decay = args.probe_weight_decay
            head = train_neural_probe(
                head,
                train_features,
                labels["train"],
                validation_features,
                labels["validation"],
                num_classes=num_classes,
                args=args,
                device=device,
                seed=derive_seed(args.seed, "probe-head", encoder_name, mode),
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            probes[name] = FrozenProbe(
                name=name,
                encoder_name=encoder_name,
                normalize=normalize,
                head=head,
            )
    return probes


def optimizer_for_checkpoint(core, checkpoint: dict[str, Any]) -> torch.optim.AdamW:
    groups = [
        {
            "params": [
                parameter
                for name, parameter in core.context_encoder.named_parameters()
                if "bias" not in name and parameter.ndim != 1
            ]
        },
        {
            "params": [
                parameter
                for name, parameter in core.predictor.named_parameters()
                if "bias" not in name and parameter.ndim != 1
            ]
        },
        {
            "params": [
                parameter
                for name, parameter in core.context_encoder.named_parameters()
                if "bias" in name or parameter.ndim == 1
            ],
            "WD_exclude": True,
            "weight_decay": 0,
        },
        {
            "params": [
                parameter
                for name, parameter in core.predictor.named_parameters()
                if "bias" in name or parameter.ndim == 1
            ],
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return optimizer


def clone_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def restore_module_state(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(state)


@torch.no_grad()
def ema_update(target: torch.nn.Module, online: torch.nn.Module, momentum: float) -> None:
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters(), strict=True
    ):
        target_parameter.mul_(momentum).add_(online_parameter, alpha=1.0 - momentum)


def gradient_norm(gradients: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    device = next(gradient.device for gradient in gradients if gradient is not None)
    total = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in gradients:
        if gradient is not None:
            total += gradient.detach().double().square().sum()
    return total.sqrt()


def gradient_alignment(
    loss_gradients: tuple[torch.Tensor | None, ...],
    richness_gradients: tuple[torch.Tensor, ...],
) -> tuple[float, float]:
    dot = torch.zeros((), dtype=torch.float64, device=richness_gradients[0].device)
    for loss_gradient, richness_gradient in zip(
        loss_gradients, richness_gradients, strict=True
    ):
        if loss_gradient is not None:
            dot += (loss_gradient.double() * richness_gradient.double()).sum()
    loss_norm = gradient_norm(loss_gradients)
    richness_norm = gradient_norm(richness_gradients)
    ras_dot = -dot
    denominator = loss_norm * richness_norm
    ras_cosine = (
        torch.zeros_like(ras_dot)
        if denominator <= torch.finfo(torch.float64).eps
        else ras_dot / denominator
    )
    return float(ras_dot.item()), float(ras_cosine.item())


def update_alignment(
    updates: tuple[torch.Tensor | None, ...],
    richness_gradients: tuple[torch.Tensor, ...],
) -> tuple[float, float]:
    dot = torch.zeros((), dtype=torch.float64, device=richness_gradients[0].device)
    for update, richness_gradient in zip(updates, richness_gradients, strict=True):
        if update is not None:
            dot += (update.double() * richness_gradient.double()).sum()
    update_norm = gradient_norm(updates)
    richness_norm = gradient_norm(richness_gradients)
    denominator = update_norm * richness_norm
    cosine = (
        torch.zeros_like(dot)
        if denominator <= torch.finfo(torch.float64).eps
        else dot / denominator
    )
    return float(dot.item()), float(cosine.item())


def richness(
    core,
    reference_images: torch.Tensor,
    *,
    functional: str,
    grid: int,
    mask_config: MaskConfig,
    mask_seed: int,
    args: argparse.Namespace,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    with autocast(reference_images.device, amp_dtype):
        value, _ = richness_from_images(
            core,
            reference_images,
            functional=functional,
            delta=args.richness_delta,
            trace_target=1.0,
            trace_beta=0.01,
            grid=grid,
            mask_config=mask_config,
            mask_seed=mask_seed,
            predictive_redundancy_weight=args.predictive_redundancy_weight,
            predictive_kappa=args.predictive_kappa,
        )
    return value


def rankdata(values: list[float]) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64)
    order = torch.argsort(tensor)
    ranks = torch.empty_like(tensor)
    sorted_values = tensor[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def correlation(left: list[float], right: list[float], *, ranks: bool) -> float:
    x = rankdata(left) if ranks else torch.tensor(left, dtype=torch.float64)
    y = rankdata(right) if ranks else torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if denominator <= torch.finfo(torch.float64).eps:
        return float("nan")
    return float((x @ y / denominator).item())


def summarize(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    score_keys = sorted(key for key in records[0] if key.startswith("ras_")) + [
        "jepa_loss",
        "gradient_norm",
        "random_score",
    ]
    utility_keys = sorted(
        key
        for key in records[0]
        if key.startswith(("delta_richness_", "probe_"))
    )
    summary = {}
    for score_key in score_keys:
        summary[score_key] = {}
        scores = [record[score_key] for record in records]
        for utility_key in utility_keys:
            utilities = [record[utility_key] for record in records]
            summary[score_key][f"pearson/{utility_key}"] = correlation(
                scores, utilities, ranks=False
            )
            summary[score_key][f"spearman/{utility_key}"] = correlation(
                scores, utilities, ranks=True
            )
    for richness_key in sorted(
        key for key in records[0] if key.startswith("delta_richness_")
    ):
        summary[f"realized/{richness_key}"] = {}
        richness_values = [record[richness_key] for record in records]
        for utility_key in sorted(
            key for key in records[0] if key.startswith("probe_")
        ):
            utilities = [record[utility_key] for record in records]
            summary[f"realized/{richness_key}"][f"pearson/{utility_key}"] = correlation(
                richness_values, utilities, ranks=False
            )
            summary[f"realized/{richness_key}"][f"spearman/{utility_key}"] = correlation(
                richness_values, utilities, ranks=True
            )
    return summary


def plot_summary(
    records: list[dict[str, float]],
    output: Path,
    functionals: tuple[str, ...] | list[str],
) -> None:
    fig, axes = plt.subplots(
        2,
        len(functionals),
        figsize=(6 * len(functionals), 9),
        squeeze=False,
    )
    for column, functional in enumerate(functionals):
        slug = FUNCTIONAL_SLUG[functional]
        optimizer_score_key = f"ras_{slug}_adamw_cosine"
        score_key = (
            optimizer_score_key
            if optimizer_score_key in records[0]
            else f"ras_{slug}_cosine"
        )
        panels = (
            (
                axes[0, column],
                score_key,
                f"delta_richness_{slug}",
                f"{functional}\nRAS vs realized ΔR",
            ),
            (
                axes[1, column],
                f"delta_richness_{slug}",
                "negative_delta_probe_loss",
                f"{functional}\nrealized ΔR vs probe utility",
            ),
        )
        for axis, x_key, y_key, title in panels:
            x = [record[x_key] for record in records]
            y = [record[y_key] for record in records]
            pearson = correlation(x, y, ranks=False)
            spearman = correlation(x, y, ranks=True)
            axis.scatter(x, y, s=15, alpha=0.55)
            axis.set_title(
                f"{title}\nPearson r={pearson:.3f} · Spearman ρ={spearman:.3f}"
            )
            axis.set_xlabel(x_key)
            axis.set_ylabel(y_key)
            axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if min(
        args.num_candidates,
        args.candidate_batch_size,
        args.mask_repeats,
        args.virtual_steps,
        args.reference_size,
        args.probe_train_size,
        args.probe_validation_size,
        args.probe_test_size,
        args.probe_epochs,
        args.probe_batch_size,
        args.mlp_epochs,
        args.mlp_hidden_dim,
    ) <= 0:
        raise ValueError("all audit sizes and --virtual-steps must be positive")
    if not math.isfinite(args.sgd_learning_rate) or args.sgd_learning_rate <= 0:
        raise ValueError("--sgd-learning-rate must be finite and positive")
    if not math.isfinite(args.step_multiplier) or args.step_multiplier <= 0:
        raise ValueError("--step-multiplier must be finite and positive")
    for name in ("probe_learning_rate", "mlp_learning_rate", "probe_ridge"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("probe_weight_decay", "mlp_weight_decay"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = run_dir / "network" / args.checkpoint
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    config = json.loads((run_dir / "config.json").read_text())
    spatial_config = config["spatial"]
    data_config = TinyImageNetDataConfig(**config["tiny_imagenet"]["data"])
    datasets = build_tiny_imagenet_static_dataset_splits(data_config)
    train_images = normalize_ijepa_images(datasets.train.images, inplace=True)
    validation_images = normalize_ijepa_images(
        datasets.validation.images,
        inplace=True,
    )
    test_images = normalize_ijepa_images(datasets.test.images, inplace=True)
    train_labels = datasets.train.entities
    validation_labels = datasets.validation.entities
    test_labels = datasets.test.entities
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = resolve_amp_dtype(args.amp_dtype, spatial_config)
    checkpoint = load_spatial_checkpoint(checkpoint_path)
    core = build_spatial_ijepa_core_from_metadata(checkpoint["metadata"])
    core.context_encoder.load_state_dict(checkpoint["context_encoder"])
    core.predictor.load_state_dict(checkpoint["predictor"])
    core.target_encoder.load_state_dict(checkpoint["target_encoder"])
    core.context_encoder.to(device)
    core.predictor.to(device)
    core.target_encoder.to(device)
    if args.virtual_optimizer == "adamw":
        optimizer: torch.optim.Optimizer = optimizer_for_checkpoint(core, checkpoint)
    else:
        optimizer = torch.optim.SGD(
            tuple(core.context_encoder.parameters()) + tuple(core.predictor.parameters()),
            lr=args.sgd_learning_rate,
        )
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] *= args.step_multiplier
    grid = int(spatial_config["image_size"]) // int(spatial_config["patch_size"])
    min_keep = int(spatial_config.get("mask_min_keep") or (10 if grid >= 16 else 4))
    mask_config = MaskConfig(min_keep=min_keep)

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_images), generator=generator)
    required = (
        args.reference_size
        + args.probe_train_size
        + args.num_candidates * args.candidate_batch_size
    )
    if required > len(order):
        raise ValueError(
            f"audit requests {required} distinct train samples, only {len(order)} exist"
        )
    reference_indices = order[: args.reference_size]
    probe_indices = order[args.reference_size : args.reference_size + args.probe_train_size]
    candidate_indices = order[
        args.reference_size
        + args.probe_train_size : required
    ].reshape(args.num_candidates, args.candidate_batch_size)
    test_indices = torch.randperm(
        len(test_images),
        generator=torch.Generator().manual_seed(derive_seed(args.seed, "probe-test")),
    )[: args.probe_test_size]
    validation_indices = torch.randperm(
        len(validation_images),
        generator=torch.Generator().manual_seed(
            derive_seed(args.seed, "probe-validation")
        ),
    )[: args.probe_validation_size]
    reference_images = train_images[reference_indices].to(device)
    probe_validation_images = validation_images[validation_indices]
    probe_validation_labels = validation_labels[validation_indices]
    probe_test_images = test_images[test_indices]
    probe_test_labels = test_labels[test_indices]
    mask_seed = derive_seed(args.seed, "audit-richness")

    encoders = {
        "context": core.context_encoder,
        "target": core.target_encoder,
    }
    encoded = {}
    for encoder_name in args.probe_encoders:
        encoder = encoders[encoder_name]
        encoded[encoder_name] = {
            "train": encode_pooled(
                encoder,
                train_images[probe_indices],
                device=device,
                batch_size=args.encode_batch_size,
                amp_dtype=amp_dtype,
            ),
            "validation": encode_pooled(
                encoder,
                probe_validation_images,
                device=device,
                batch_size=args.encode_batch_size,
                amp_dtype=amp_dtype,
            ),
            "test": encode_pooled(
                encoder,
                probe_test_images,
                device=device,
                batch_size=args.encode_batch_size,
                amp_dtype=amp_dtype,
            ),
        }
    probe_labels = {
        "train": train_labels[probe_indices],
        "validation": probe_validation_labels,
        "test": probe_test_labels,
    }
    probes = build_probe_matrix(
        encoded,
        probe_labels,
        num_classes=data_config.num_entities,
        args=args,
        device=device,
    )
    base_probe_metrics = {
        name: probe_metrics(
            probe,
            encoded[probe.encoder_name]["test"],
            probe_test_labels,
            device=device,
            batch_size=args.probe_batch_size,
        )
        for name, probe in probes.items()
    }
    base_reference_features = encode_pooled(
        core.context_encoder,
        reference_images,
        device=device,
        batch_size=args.encode_batch_size,
        amp_dtype=amp_dtype,
    )

    encoder_parameters = tuple(
        parameter for parameter in core.context_encoder.parameters() if parameter.requires_grad
    )
    base_richness = {}
    richness_gradients = {}
    for functional in args.richness_functionals:
        value = richness(
            core,
            reference_images,
            functional=functional,
            grid=grid,
            mask_config=mask_config,
            mask_seed=mask_seed,
            args=args,
            amp_dtype=amp_dtype,
        )
        base_richness[functional] = float(value.detach().item())
        richness_gradients[functional] = tuple(
            gradient.detach()
            for gradient in torch.autograd.grad(value, encoder_parameters)
        )

    base_context = clone_module_state(core.context_encoder)
    base_predictor = clone_module_state(core.predictor)
    base_target = clone_module_state(core.target_encoder)
    base_optimizer = (
        copy.deepcopy(optimizer.state_dict())
        if args.virtual_optimizer == "adamw"
        else None
    )
    ema_start, ema_end = spatial_config.get("ema", [0.996, 1.0])
    schedule_steps = max(
        int(
            float(spatial_config.get("ipe_scale", 1.0))
            * int(spatial_config["epochs"])
            * max(len(train_images) // int(spatial_config["batch_size"]), 1)
        ),
        1,
    )
    schedule_progress = min(float(checkpoint.get("global_step", 0)) / schedule_steps, 1.0)
    ema_momentum = float(ema_start + schedule_progress * (ema_end - ema_start))
    completed = {}
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            record = json.loads(line)
            key = (int(record["candidate"]), int(record.get("mask_repeat", 0)))
            completed[key] = record

    num_trials = args.num_candidates * args.mask_repeats
    for trial in range(num_trials):
        candidate, mask_repeat = divmod(trial, args.mask_repeats)
        key = (candidate, mask_repeat)
        if key in completed:
            continue
        restore_module_state(core.context_encoder, base_context)
        restore_module_state(core.predictor, base_predictor)
        restore_module_state(core.target_encoder, base_target)
        if base_optimizer is not None:
            optimizer.load_state_dict(base_optimizer)
        core.context_encoder.train()
        core.predictor.train()
        core.target_encoder.train()
        batch = train_images[candidate_indices[candidate]].to(device)
        score_generator = torch.Generator().manual_seed(
            derive_seed(args.seed, "candidate-score", candidate, mask_repeat)
        )
        score_context_masks, score_target_masks = sample_masks(
            grid,
            grid,
            mask_config,
            score_generator,
            batch_size=len(batch),
        )
        with autocast(device, amp_dtype):
            candidate_loss = spatial_ijepa_loss(
                core,
                batch,
                score_context_masks,
                score_target_masks,
            )
        loss_gradients = torch.autograd.grad(
            candidate_loss,
            encoder_parameters,
            allow_unused=True,
        )
        alignments = {
            functional: gradient_alignment(
                loss_gradients,
                richness_gradients[functional],
            )
            for functional in args.richness_functionals
        }
        optimizer_alignments = {}
        if args.virtual_optimizer == "adamw":
            update_direction = adamw_update_direction(
                encoder_parameters,
                loss_gradients,
                optimizer,
            )
            optimizer_alignments = {
                functional: update_alignment(
                    update_direction,
                    richness_gradients[functional],
                )
                for functional in args.richness_functionals
            }
        candidate_gradient_norm = float(gradient_norm(loss_gradients).item())

        for virtual_step in range(args.virtual_steps):
            optimizer.zero_grad(set_to_none=True)
            if virtual_step == 0 and args.reuse_score_masks_first_step:
                context_masks, target_masks = score_context_masks, score_target_masks
            else:
                step_generator = torch.Generator().manual_seed(
                    derive_seed(
                        args.seed,
                        "candidate-step",
                        candidate,
                        mask_repeat,
                        virtual_step,
                    )
                )
                context_masks, target_masks = sample_masks(
                    grid,
                    grid,
                    mask_config,
                    step_generator,
                    batch_size=len(batch),
                )
            with autocast(device, amp_dtype):
                loss = spatial_ijepa_loss(core, batch, context_masks, target_masks)
            loss.backward()
            optimizer.step()
            if args.virtual_ema_update:
                ema_update(core.target_encoder, core.context_encoder, ema_momentum)

        core.context_encoder.eval()
        core.predictor.eval()
        core.target_encoder.eval()
        post_richness = {}
        with torch.no_grad():
            for functional in args.richness_functionals:
                post_richness[functional] = float(
                    richness(
                        core,
                        reference_images,
                        functional=functional,
                        grid=grid,
                        mask_config=mask_config,
                        mask_seed=mask_seed,
                        args=args,
                        amp_dtype=amp_dtype,
                    ).item()
                )
        post_features = {
            encoder_name: encode_pooled(
                encoders[encoder_name],
                probe_test_images,
                device=device,
                batch_size=args.encode_batch_size,
                amp_dtype=amp_dtype,
            )
            for encoder_name in args.probe_encoders
        }
        post_reference_features = encode_pooled(
            core.context_encoder,
            reference_images,
            device=device,
            batch_size=args.encode_batch_size,
            amp_dtype=amp_dtype,
        )
        reference_difference = post_reference_features - base_reference_features
        representation_drift = float(
            reference_difference.float().square().mean().sqrt().item()
        )
        normalized_base_reference = F.normalize(base_reference_features.float(), dim=1)
        normalized_post_reference = F.normalize(post_reference_features.float(), dim=1)
        representation_cosine_drift = float(
            (1.0 - (normalized_base_reference * normalized_post_reference).sum(dim=1))
            .mean()
            .item()
        )
        with torch.no_grad(), autocast(device, amp_dtype):
            post_candidate_loss = spatial_ijepa_loss(
                core,
                batch,
                score_context_masks,
                score_target_masks,
            )
        post_probe_metrics = {
            name: probe_metrics(
                probe,
                post_features[probe.encoder_name],
                probe_test_labels,
                device=device,
                batch_size=args.probe_batch_size,
            )
            for name, probe in probes.items()
        }
        record = {
            "candidate": candidate,
            "mask_repeat": mask_repeat,
            "step_multiplier": args.step_multiplier,
            "jepa_loss": float(candidate_loss.detach().item()),
            "post_jepa_loss": float(post_candidate_loss.detach().item()),
            "jepa_loss_improvement": float(
                candidate_loss.detach().item() - post_candidate_loss.detach().item()
            ),
            "gradient_norm": candidate_gradient_norm,
            "representation_drift": representation_drift,
            "representation_cosine_drift": representation_cosine_drift,
            "random_score": float(
                torch.rand(
                    (),
                    generator=torch.Generator().manual_seed(
                        derive_seed(
                            args.seed,
                            "random-score",
                            candidate,
                            mask_repeat,
                        )
                    ),
                ).item()
            ),
        }
        for name in probes:
            base_loss, base_accuracy = base_probe_metrics[name]
            post_loss, post_accuracy = post_probe_metrics[name]
            record[f"probe_{name}_utility_loss"] = base_loss - post_loss
            record[f"probe_{name}_delta_accuracy"] = post_accuracy - base_accuracy
        compatibility_probe = (
            "target_linear" if "target_linear" in probes else next(iter(probes))
        )
        record["negative_delta_probe_loss"] = record[
            f"probe_{compatibility_probe}_utility_loss"
        ]
        record["delta_probe_accuracy"] = record[
            f"probe_{compatibility_probe}_delta_accuracy"
        ]
        for functional in args.richness_functionals:
            slug = FUNCTIONAL_SLUG[functional]
            dot, cosine = alignments[functional]
            record[f"ras_{slug}_dot"] = dot
            record[f"ras_{slug}_cosine"] = cosine
            if functional in optimizer_alignments:
                adamw_dot, adamw_cosine = optimizer_alignments[functional]
                record[f"ras_{slug}_adamw_dot"] = adamw_dot
                record[f"ras_{slug}_adamw_cosine"] = adamw_cosine
            record[f"delta_richness_{slug}"] = (
                post_richness[functional] - base_richness[functional]
            )
            if functional in optimizer_alignments and args.virtual_steps == 1:
                predicted_delta = optimizer_alignments[functional][0]
                record[f"predicted_delta_richness_{slug}"] = predicted_delta
                record[f"taylor_remainder_{slug}"] = (
                    record[f"delta_richness_{slug}"] - predicted_delta
                )
        with records_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        completed[key] = record
        print(
            f"candidate={candidate + 1}/{args.num_candidates} "
            f"mask_repeat={mask_repeat + 1}/{args.mask_repeats} "
            f"{FUNCTIONAL_SLUG[args.richness_functionals[0]]}_cos="
            f"{alignments[args.richness_functionals[0]][1]:.4f} "
            f"dR={record['delta_richness_' + FUNCTIONAL_SLUG[args.richness_functionals[0]]]:.4g} "
            f"probe_utility={record['negative_delta_probe_loss']:.4g}",
            flush=True,
        )

    records = [completed[key] for key in sorted(completed)]
    summary = {
        "checkpoint": str(checkpoint_path),
        "config": vars(args) | {"run_dir": str(run_dir), "output_dir": str(output_dir)},
        "base_probes": {
            name: {"loss": metrics[0], "accuracy": metrics[1]}
            for name, metrics in base_probe_metrics.items()
        },
        "base_richness": base_richness,
        "correlations": summarize(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_summary(
        records,
        output_dir / "correlations.png",
        args.richness_functionals,
    )
    print(json.dumps(summary["correlations"], indent=2), flush=True)


if __name__ == "__main__":
    main()
