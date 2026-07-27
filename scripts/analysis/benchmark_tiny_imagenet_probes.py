#!/usr/bin/env python3
"""Benchmark frozen probes and supervised fine-tuning for a Tiny ImageNet checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from jepa.analysis.subspace import (  # noqa: E402
    classifier_accuracy,
    fit_entity_classifier,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    build_tiny_imagenet_static_dataset_splits,
)
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    build_spatial_ijepa_core_from_metadata,
    load_spatial_checkpoint,
    normalize_ijepa_images,
)

EncoderName = Literal["context", "target"]
SplitName = Literal["train", "validation", "test"]
DEFAULT_MODES = ("ridge", "linear", "linear-l2", "mlp", "finetune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="epoch_0700.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=DEFAULT_MODES,
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=("context", "target"),
        default=("context", "target"),
        help="Frozen encoders to benchmark; fine-tuning uses --finetune-encoder.",
    )
    parser.add_argument(
        "--finetune-encoder",
        choices=("context", "target"),
        default="target",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--linear-epochs", type=int, default=100)
    parser.add_argument("--linear-lr", type=float, default=1.0e-2)
    parser.add_argument("--linear-weight-decay", type=float, default=0.0)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-hidden-dim", type=int, default=1024)
    parser.add_argument("--mlp-lr", type=float, default=1.0e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-batch-size", type=int, default=256)
    parser.add_argument("--finetune-encoder-lr", type=float, default=1.0e-4)
    parser.add_argument("--finetune-head-lr", type=float, default=1.0e-3)
    parser.add_argument("--finetune-weight-decay", type=float, default=0.05)
    parser.add_argument("--finetune-horizontal-flip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "encode_batch_size": args.encode_batch_size,
        "probe_batch_size": args.probe_batch_size,
        "ridge": args.ridge,
        "linear_epochs": args.linear_epochs,
        "linear_lr": args.linear_lr,
        "mlp_epochs": args.mlp_epochs,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_lr": args.mlp_lr,
        "finetune_epochs": args.finetune_epochs,
        "finetune_batch_size": args.finetune_batch_size,
        "finetune_encoder_lr": args.finetune_encoder_lr,
        "finetune_head_lr": args.finetune_head_lr,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.finetune_horizontal_flip <= 1:
        raise ValueError("--finetune-horizontal-flip must be in [0, 1]")
    if args.linear_weight_decay < 0 or args.mlp_weight_decay < 0:
        raise ValueError("probe weight decay must be non-negative")
    if args.finetune_weight_decay < 0:
        raise ValueError("--finetune-weight-decay must be non-negative")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_amp_dtype(value: str) -> torch.dtype | None:
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


def topk_accuracy(scores: torch.Tensor, labels: torch.Tensor, *, k: int) -> float:
    predicted = scores.topk(min(k, scores.shape[-1]), dim=-1).indices
    return predicted.eq(labels.unsqueeze(-1)).any(dim=-1).float().mean().item()


def classification_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    return {
        "loss": float(F.cross_entropy(scores.float(), labels).item()),
        "top1": topk_accuracy(scores, labels, k=1),
        "top5": topk_accuracy(scores, labels, k=5),
    }


def checkpoint_path(run_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = run_dir / "network" / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_core(path: Path, device: torch.device):
    checkpoint = load_spatial_checkpoint(path)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint has no model metadata: {path}")
    core = build_spatial_ijepa_core_from_metadata(metadata)
    core.context_encoder.load_state_dict(checkpoint["context_encoder"])
    core.target_encoder.load_state_dict(checkpoint["target_encoder"])
    core.predictor.load_state_dict(checkpoint["predictor"])
    core.context_encoder.to(device).eval()
    core.target_encoder.to(device).eval()
    core.predictor.to(device).eval()
    return core, checkpoint


@torch.inference_mode()
def encode_images(
    encoder: nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    encoder.eval()
    features = []
    for batch in images.split(batch_size):
        with autocast(device, amp_dtype):
            tokens = encoder(batch.to(device, non_blocking=True))
            pooled = tokens.mean(dim=1)
        features.append(pooled.float().cpu())
    return torch.cat(features)


def feature_cache_path(
    output_dir: Path,
    *,
    encoder_name: EncoderName,
    split_name: SplitName,
) -> Path:
    return output_dir / "features" / f"{encoder_name}_{split_name}.pt"


def encode_or_load(
    encoder: nn.Module,
    images: torch.Tensor,
    *,
    encoder_name: EncoderName,
    split_name: SplitName,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    cache: bool,
) -> torch.Tensor:
    path = feature_cache_path(
        output_dir,
        encoder_name=encoder_name,
        split_name=split_name,
    )
    if cache and path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    features = encode_images(
        encoder,
        images,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
    )
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, path)
    return features


def maybe_l2_normalize(features: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    return F.normalize(features, dim=-1) if enabled else features


def evaluate_ridge(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    num_classes: int,
    ridge: float,
) -> dict[str, float]:
    classifier = fit_entity_classifier(
        train_features,
        train_labels,
        num_classes,
        ridge=ridge,
    )
    scores = classifier.probe.predict(test_features).float()
    metrics = classification_metrics(scores, test_labels)
    metrics["top1"] = classifier_accuracy(classifier, test_features, test_labels)
    return metrics


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@torch.inference_mode()
def evaluate_head(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    scores = []
    for batch in features.split(batch_size):
        scores.append(head(batch.to(device)).float().cpu())
    return classification_metrics(torch.cat(scores), labels)


def train_frozen_head(
    head: nn.Module,
    *,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, torch.Tensor]]:
    head.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = math.inf
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train_features), generator=generator)
        total_loss = 0.0
        for indices in order.split(batch_size):
            features = train_features[indices].to(device, non_blocking=True)
            labels = train_labels[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(indices)
        scheduler.step()
        validation = evaluate_head(
            head,
            validation_features,
            validation_labels,
            batch_size=batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / len(train_features),
                "validation_loss": validation["loss"],
                "validation_top1": validation["top1"],
                "validation_top5": validation["top5"],
            }
        )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
    assert best_state is not None
    head.load_state_dict(best_state)
    test = evaluate_head(
        head,
        test_features,
        test_labels,
        batch_size=batch_size,
        device=device,
    )
    test["best_validation_loss"] = best_validation_loss
    test["best_epoch"] = float(
        min(history, key=lambda row: row["validation_loss"])["epoch"]
    )
    return test, history, best_state


@torch.inference_mode()
def evaluate_finetuned(
    encoder: nn.Module,
    head: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    encoder.eval()
    head.eval()
    scores = []
    for batch in images.split(batch_size):
        with autocast(device, amp_dtype):
            features = encoder(batch.to(device, non_blocking=True)).mean(dim=1)
            logits = head(features)
        scores.append(logits.float().cpu())
    return classification_metrics(torch.cat(scores), labels)


def train_full_finetune(
    encoder: nn.Module,
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    embed_dim: int,
    epochs: int,
    batch_size: int,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
    horizontal_flip_probability: float,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, Any]]:
    encoder = copy.deepcopy(encoder).to(device)
    encoder.requires_grad_(True)
    head = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": encoder_lr},
            {"params": head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_dtype == torch.float16 and device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, Any] | None = None
    best_validation_loss = math.inf
    history = []
    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        order = torch.randperm(len(train_images), generator=generator)
        total_loss = 0.0
        for indices in order.split(batch_size):
            images = train_images[indices].to(device, non_blocking=True)
            labels = train_labels[indices].to(device, non_blocking=True)
            if horizontal_flip_probability > 0:
                flip = torch.rand(
                    len(images),
                    generator=generator,
                ) < horizontal_flip_probability
                if flip.any():
                    images = images.clone()
                    images[flip.to(images.device)] = images[flip.to(images.device)].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, amp_dtype):
                features = encoder(images).mean(dim=1)
                loss = F.cross_entropy(head(features), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * len(indices)
        scheduler.step()
        validation = evaluate_finetuned(
            encoder,
            head,
            validation_images,
            validation_labels,
            batch_size=batch_size,
            device=device,
            amp_dtype=amp_dtype,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / len(train_images),
                "validation_loss": validation["loss"],
                "validation_top1": validation["top1"],
                "validation_top5": validation["top5"],
            }
        )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_state = {
                "encoder": {
                    name: value.detach().cpu().clone()
                    for name, value in encoder.state_dict().items()
                },
                "head": {
                    name: value.detach().cpu().clone()
                    for name, value in head.state_dict().items()
                },
            }
        print(
            f"finetune epoch={epoch:3d}/{epochs} "
            f"train_loss={history[-1]['train_loss']:.4f} "
            f"val_top1={validation['top1']:.4f}",
            flush=True,
        )
    assert best_state is not None
    encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])
    test = evaluate_finetuned(
        encoder,
        head,
        test_images,
        test_labels,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
    )
    test["best_validation_loss"] = best_validation_loss
    test["best_epoch"] = float(
        min(history, key=lambda row: row["validation_loss"])["epoch"]
    )
    return test, history, best_state


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = json.loads((run_dir / "config.json").read_text())
    data_config = TinyImageNetDataConfig(**run_config["tiny_imagenet"]["data"])
    datasets = build_tiny_imagenet_static_dataset_splits(data_config)
    splits = {
        "train": (
            normalize_ijepa_images(datasets.train.images),
            datasets.train.entities.to(torch.long),
        ),
        "validation": (
            normalize_ijepa_images(datasets.validation.images),
            datasets.validation.entities.to(torch.long),
        ),
        "test": (
            normalize_ijepa_images(datasets.test.images),
            datasets.test.entities.to(torch.long),
        ),
    }
    num_classes = data_config.num_entities
    path = checkpoint_path(run_dir, args.checkpoint)
    core, checkpoint = load_core(path, device)
    results: dict[str, Any] = {
        "config": vars(args)
        | {
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "checkpoint": str(path),
            "data": asdict(data_config),
        },
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "probes": {},
    }
    write_json(output_dir / "results.json", results)
    (output_dir / "network").mkdir(parents=True, exist_ok=True)

    encoded: dict[EncoderName, dict[SplitName, torch.Tensor]] = {}
    frozen_modes = set(args.modes) & {"ridge", "linear", "linear-l2", "mlp"}
    if frozen_modes:
        for encoder_name in args.encoders:
            encoder = (
                core.context_encoder if encoder_name == "context" else core.target_encoder
            )
            encoded[encoder_name] = {}
            for split_name, (images, _) in splits.items():
                started = time.time()
                encoded[encoder_name][split_name] = encode_or_load(
                    encoder,
                    images,
                    encoder_name=encoder_name,
                    split_name=split_name,
                    output_dir=output_dir,
                    batch_size=args.encode_batch_size,
                    device=device,
                    amp_dtype=amp_dtype,
                    cache=args.cache_features,
                )
                print(
                    f"encoded {encoder_name}/{split_name} "
                    f"shape={tuple(encoded[encoder_name][split_name].shape)} "
                    f"seconds={time.time() - started:.1f}",
                    flush=True,
                )

    for encoder_name in args.encoders:
        if encoder_name not in encoded:
            continue
        features = encoded[encoder_name]
        train_labels = splits["train"][1]
        validation_labels = splits["validation"][1]
        test_labels = splits["test"][1]
        if "ridge" in args.modes:
            metrics = evaluate_ridge(
                features["train"],
                train_labels,
                features["test"],
                test_labels,
                num_classes=num_classes,
                ridge=args.ridge,
            )
            results["probes"][f"{encoder_name}/ridge"] = metrics
            write_json(output_dir / "results.json", results)
        for mode in ("linear", "linear-l2", "mlp"):
            if mode not in args.modes:
                continue
            normalize = mode == "linear-l2"
            train_features = maybe_l2_normalize(features["train"], enabled=normalize)
            validation_features = maybe_l2_normalize(
                features["validation"],
                enabled=normalize,
            )
            test_features = maybe_l2_normalize(features["test"], enabled=normalize)
            if mode == "mlp":
                head: nn.Module = MLPProbe(
                    train_features.shape[1],
                    args.mlp_hidden_dim,
                    num_classes,
                )
                epochs = args.mlp_epochs
                learning_rate = args.mlp_lr
                weight_decay = args.mlp_weight_decay
            else:
                head = nn.Linear(train_features.shape[1], num_classes)
                epochs = args.linear_epochs
                learning_rate = args.linear_lr
                weight_decay = args.linear_weight_decay
            metrics, history, state = train_frozen_head(
                head,
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                validation_labels=validation_labels,
                test_features=test_features,
                test_labels=test_labels,
                epochs=epochs,
                batch_size=args.probe_batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                seed=derive_seed(args.seed, encoder_name, mode),
            )
            key = f"{encoder_name}/{mode}"
            results["probes"][key] = metrics
            write_json(output_dir / "history" / f"{encoder_name}_{mode}.json", history)
            torch.save(state, output_dir / "network" / f"{encoder_name}_{mode}.pt")
            write_json(output_dir / "results.json", results)
            print(f"{key}: {metrics}", flush=True)

    if "finetune" in args.modes:
        finetune_encoder = (
            core.context_encoder
            if args.finetune_encoder == "context"
            else core.target_encoder
        )
        metrics, history, state = train_full_finetune(
            finetune_encoder,
            train_images=splits["train"][0],
            train_labels=splits["train"][1],
            validation_images=splits["validation"][0],
            validation_labels=splits["validation"][1],
            test_images=splits["test"][0],
            test_labels=splits["test"][1],
            num_classes=num_classes,
            embed_dim=core.embed_dim,
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch_size,
            encoder_lr=args.finetune_encoder_lr,
            head_lr=args.finetune_head_lr,
            weight_decay=args.finetune_weight_decay,
            horizontal_flip_probability=args.finetune_horizontal_flip,
            device=device,
            amp_dtype=amp_dtype,
            seed=derive_seed(args.seed, args.finetune_encoder, "finetune"),
        )
        key = f"{args.finetune_encoder}/finetune"
        results["probes"][key] = metrics
        write_json(output_dir / "history" / f"{args.finetune_encoder}_finetune.json", history)
        torch.save(state, output_dir / "network" / f"{args.finetune_encoder}_finetune.pt")
        write_json(output_dir / "results.json", results)
        print(f"{key}: {metrics}", flush=True)

    print(json.dumps(results["probes"], indent=2), flush=True)


if __name__ == "__main__":
    main()
