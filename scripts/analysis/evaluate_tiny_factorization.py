#!/usr/bin/env python3
"""Evaluate label-free factorization metrics on Tiny ImageNet checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jepa.analysis.quality_metrics import (  # noqa: E402
    MetricValue,
    q1_cross_covariance,
    q2_projector_interaction,
    q3_projector_overlap,
    q4_partition_incompleteness,
    q5_q6_subspace_stability,
    q9_subspace_velocity,
    q10_weighted_shape_entropy,
    q11_cross_subspace_gaussian_mi,
    q12_entropy_change,
    q13_surprise_locality,
    q14_gradient_locality,
    q17_transformation_residual,
    q18_perturbation_concentration,
)
from jepa.analysis.spatial_decomposition import (  # noqa: E402
    EigenBoundaryTieError,
    SubspacePartition,
    fit_label_free_partition,
)
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
    spatial_ijepa_prediction_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", default="epoch_*.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--encoder", choices=("target", "context"), default="target")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "none", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fit-size", type=int, default=2048)
    parser.add_argument("--metric-size", type=int, default=2048)
    parser.add_argument("--gradient-size", type=int, default=128)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-gradient-locality",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def checkpoint_epoch(path: Path) -> tuple[int, str]:
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else -1, path.name)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(value)


def resolve_amp(value: str, spatial_config: dict[str, Any]) -> torch.dtype | None:
    if value == "auto":
        value = str(spatial_config.get("amp_dtype") or "")
        if not value:
            value = "bfloat16" if spatial_config.get("bfloat16", False) else "none"
    return {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def metric_payload(metric: MetricValue) -> dict[str, Any]:
    return {"value": metric.value, "null_reason": metric.reason}


def finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"metric is non-finite: {value}")
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    temporary.replace(path)


def load_core(path: Path, device: torch.device):
    checkpoint = load_spatial_checkpoint(path)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint has no model metadata: {path}")
    core = build_spatial_ijepa_core_from_metadata(metadata)
    core.context_encoder.load_state_dict(checkpoint["context_encoder"])
    core.predictor.load_state_dict(checkpoint["predictor"])
    core.target_encoder.load_state_dict(checkpoint["target_encoder"])
    core.context_encoder.to(device).eval()
    core.predictor.to(device).eval()
    core.target_encoder.to(device).eval()
    return core, checkpoint


@torch.inference_mode()
def encode_masked_views(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    grid: int,
    mask_config: MaskConfig,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generators = (
        torch.Generator().manual_seed(derive_seed(seed, "mask-view-a")),
        torch.Generator().manual_seed(derive_seed(seed, "mask-view-b")),
    )
    encoded: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    for batch in images.split(batch_size):
        batch = batch.to(device, non_blocking=True)
        for slot, generator in enumerate(generators):
            context_masks, _ = sample_masks(
                grid,
                grid,
                mask_config,
                generator,
                batch_size=batch.shape[0],
            )
            context_masks = [mask.to(device, non_blocking=True) for mask in context_masks]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                features = encoder(batch, context_masks).mean(dim=1)
            encoded[slot].append(features.float().cpu())
    return torch.cat(encoded[0]), torch.cat(encoded[1])


def latent_gradients(
    core,
    images: torch.Tensor,
    *,
    grid: int,
    mask_config: MaskConfig,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(derive_seed(seed, "gradient-masks"))
    gradients = []
    for batch in images.split(batch_size):
        batch = batch.to(device, non_blocking=True)
        context_masks, target_masks = sample_masks(
            grid,
            grid,
            mask_config,
            generator,
            batch_size=batch.shape[0],
        )
        with torch.enable_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            prediction, target, context, _, _ = spatial_ijepa_prediction_targets(
                core,
                batch,
                context_masks,
                target_masks,
            )
            loss = F.smooth_l1_loss(prediction, target)
            gradient = torch.autograd.grad(loss, context, retain_graph=False)[0]
        gradients.append(gradient.mean(dim=1).float().cpu())
    return torch.cat(gradients)


def cache_path(
    cache_dir: Path,
    checkpoint: Path,
    *,
    encoder: str,
    fit_size: int,
    metric_size: int,
    seed: int,
) -> Path:
    return cache_dir / (
        f"{checkpoint.stem}_{encoder}_fit{fit_size}_metric{metric_size}_seed{seed}.pt"
    )


def load_or_encode(
    path: Path,
    *,
    encoder: torch.nn.Module,
    fit_images: torch.Tensor,
    metric_images: torch.Tensor,
    grid: int,
    mask_config: MaskConfig,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    seed: int,
    cache: bool,
) -> dict[str, torch.Tensor]:
    if cache and path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"invalid feature cache: {path}")
    fit_a, fit_b = encode_masked_views(
        encoder,
        fit_images,
        grid=grid,
        mask_config=mask_config,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        seed=derive_seed(seed, "fit"),
    )
    metric_a, metric_b = encode_masked_views(
        encoder,
        metric_images,
        grid=grid,
        mask_config=mask_config,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        seed=derive_seed(seed, "metric"),
    )
    payload = {
        "fit_a": fit_a,
        "fit_b": fit_b,
        "metric_a": metric_a,
        "metric_b": metric_b,
    }
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
    return payload


def evaluate_static_metrics(
    features: dict[str, torch.Tensor],
    partition: SubspacePartition,
) -> dict[str, Any]:
    pooled = (features["metric_a"] + features["metric_b"]) / 2
    entropy, entropy_bands = q10_weighted_shape_entropy(pooled, partition)
    mutual_information, clamped = q11_cross_subspace_gaussian_mi(pooled, partition)
    forward = q17_transformation_residual(
        features["fit_a"],
        features["fit_b"],
        features["metric_a"],
        features["metric_b"],
    )
    backward = q17_transformation_residual(
        features["fit_b"],
        features["fit_a"],
        features["metric_b"],
        features["metric_a"],
    )
    if forward.value is None:
        transformation = forward
    elif backward.value is None:
        transformation = backward
    else:
        transformation = MetricValue((forward.value + backward.value) / 2)
    concentration, eligible, total = q18_perturbation_concentration(
        features["metric_a"],
        features["metric_b"],
        partition,
    )
    return {
        "q1_cross_covariance": {
            "value": finite(q1_cross_covariance(pooled, partition)),
            "null_reason": None,
        },
        "q2_projector_interaction": {
            "value": finite(q2_projector_interaction(pooled, partition)),
            "null_reason": None,
        },
        "q3_projector_overlap": {
            "value": finite(q3_projector_overlap(partition)),
            "null_reason": None,
        },
        "q4_partition_incompleteness": {
            "value": finite(q4_partition_incompleteness(partition)),
            "null_reason": None,
        },
        "q10_weighted_shape_entropy": metric_payload(entropy)
        | {"bands": list(entropy_bands)},
        "q11_cross_subspace_gaussian_mi": metric_payload(mutual_information)
        | {"clamped": clamped},
        "q17_mask_transformation_residual": metric_payload(transformation)
        | {
            "forward": forward.value,
            "backward": backward.value,
        },
        "q18_mask_perturbation_concentration": metric_payload(concentration)
        | {"eligible": eligible, "total": total},
    }


def main() -> None:
    args = parse_args()
    positive = {
        "batch_size": args.batch_size,
        "fit_size": args.fit_size,
        "metric_size": args.metric_size,
        "gradient_size": args.gradient_size,
        "bands": args.bands,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"all sizes must be positive: {positive}")
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "factorization"
    )
    records_path = output_dir / "records.json"
    config = json.loads((run_dir / "config.json").read_text())
    spatial_config = config["spatial"]
    data_config = TinyImageNetDataConfig(**config["tiny_imagenet"]["data"])
    datasets = build_tiny_imagenet_static_dataset_splits(data_config)
    train_images = normalize_ijepa_images(datasets.train.images, inplace=True)
    total_required = args.fit_size + args.metric_size + args.gradient_size
    if total_required > len(train_images):
        raise ValueError(
            f"evaluation needs {total_required} distinct images, only "
            f"{len(train_images)} are available"
        )
    order = torch.randperm(
        len(train_images),
        generator=torch.Generator().manual_seed(args.seed),
    )
    fit_indices = order[: args.fit_size]
    metric_indices = order[args.fit_size : args.fit_size + args.metric_size]
    gradient_indices = order[
        args.fit_size + args.metric_size : total_required
    ]
    fit_images = train_images[fit_indices]
    metric_images = train_images[metric_indices]
    gradient_images = train_images[gradient_indices]

    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp_dtype, spatial_config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    image_size = int(spatial_config.get("image_size", 64))
    patch_size = int(spatial_config["patch_size"])
    grid = image_size // patch_size
    min_keep = int(spatial_config.get("mask_min_keep") or (10 if grid >= 16 else 4))
    mask_config = MaskConfig(min_keep=min_keep)
    checkpoints = sorted(
        (run_dir / "network").glob(args.checkpoint_pattern),
        key=checkpoint_epoch,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoints matching {args.checkpoint_pattern!r} in {run_dir / 'network'}"
        )

    existing: dict[str, dict[str, Any]] = {}
    if records_path.exists():
        loaded = json.loads(records_path.read_text())
        existing = {str(record["checkpoint"]): record for record in loaded}
    records: list[dict[str, Any]] = []
    previous_partition: SubspacePartition | None = None
    previous_step: int | None = None
    previous_entropy: float | None = None
    print(
        f"run={run_dir.name} checkpoints={len(checkpoints)} device={device} "
        f"encoder={args.encoder}",
        flush=True,
    )
    for path in checkpoints:
        started = time.time()
        core, checkpoint = load_core(path, device)
        encoder = (
            core.target_encoder if args.encoder == "target" else core.context_encoder
        )
        features = load_or_encode(
            cache_path(
                output_dir / "cache",
                path,
                encoder=args.encoder,
                fit_size=args.fit_size,
                metric_size=args.metric_size,
                seed=args.seed,
            ),
            encoder=encoder,
            fit_images=fit_images,
            metric_images=metric_images,
            grid=grid,
            mask_config=mask_config,
            batch_size=args.batch_size,
            device=device,
            amp_dtype=amp_dtype,
            seed=args.seed,
            cache=args.cache_features,
        )
        epoch = int(checkpoint["epoch"])
        step = int(checkpoint.get("global_step") or epoch)
        try:
            partition = fit_label_free_partition(
                features["fit_a"],
                features["fit_b"],
                checkpoint_id=path.name,
                bands=args.bands,
            )
            metrics = evaluate_static_metrics(features, partition)
            if args.include_gradient_locality:
                gradients = latent_gradients(
                    core,
                    gradient_images,
                    grid=grid,
                    mask_config=mask_config,
                    batch_size=args.batch_size,
                    device=device,
                    amp_dtype=amp_dtype,
                    seed=args.seed,
                )
                locality, shares, maximum = q14_gradient_locality(
                    gradients,
                    partition,
                )
                metrics["q14_gradient_locality"] = metric_payload(locality) | {
                    "shares": list(shares),
                    "maximum_share": maximum,
                }
            if previous_partition is None:
                for name in (
                    "q5_subspace_similarity",
                    "q6_subspace_distance",
                    "q9_subspace_velocity",
                    "q12_absolute_entropy_change",
                    "q13_realized_surprise_locality",
                ):
                    metrics[name] = {
                        "value": None,
                        "null_reason": "no_previous_checkpoint",
                    }
            else:
                q5, q6, distances = q5_q6_subspace_stability(
                    partition,
                    previous_partition,
                )
                metrics["q5_subspace_similarity"] = {
                    "value": finite(q5),
                    "null_reason": None,
                }
                metrics["q6_subspace_distance"] = {
                    "value": finite(q6),
                    "null_reason": None,
                    "bands": list(distances),
                }
                metrics["q9_subspace_velocity"] = {
                    "value": finite(q9_subspace_velocity(q6, step, previous_step or 0)),
                    "null_reason": None,
                }
                current_entropy = metrics["q10_weighted_shape_entropy"]["value"]
                if current_entropy is None or previous_entropy is None:
                    metrics["q12_absolute_entropy_change"] = {
                        "value": None,
                        "null_reason": "degenerate_covariance",
                    }
                else:
                    absolute, signed = q12_entropy_change(
                        current_entropy,
                        previous_entropy,
                    )
                    metrics["q12_absolute_entropy_change"] = {
                        "value": finite(absolute),
                        "null_reason": None,
                        "signed": finite(signed),
                    }
                metrics["q13_realized_surprise_locality"] = metric_payload(
                    q13_surprise_locality(distances)
                )
            record = {
                "checkpoint": path.name,
                "epoch": epoch,
                "global_step": step,
                "encoder": args.encoder,
                "metrics": metrics,
                "seconds": time.time() - started,
            }
            previous_partition = partition
            previous_step = step
            previous_entropy = metrics["q10_weighted_shape_entropy"]["value"]
        except EigenBoundaryTieError as error:
            record = {
                "checkpoint": path.name,
                "epoch": epoch,
                "global_step": step,
                "encoder": args.encoder,
                "partition_error": str(error),
                "metrics": {},
                "seconds": time.time() - started,
            }
            previous_partition = None
            previous_step = None
            previous_entropy = None
        existing[path.name] = record
        records = sorted(existing.values(), key=lambda item: int(item["epoch"]))
        atomic_write_json(records_path, records)
        atomic_write_jsonl(output_dir / "records.jsonl", records)
        print(
            f"checkpoint={path.name} epoch={epoch} seconds={record['seconds']:.1f}",
            flush=True,
        )
        del core
        if device.type == "cuda":
            torch.cuda.empty_cache()

    atomic_write_json(
        output_dir / "config.json",
        {
            "run_dir": str(run_dir),
            "checkpoint_pattern": args.checkpoint_pattern,
            "encoder": args.encoder,
            "device": str(device),
            "amp_dtype": str(amp_dtype),
            "batch_size": args.batch_size,
            "fit_size": args.fit_size,
            "metric_size": args.metric_size,
            "gradient_size": args.gradient_size,
            "bands": args.bands,
            "seed": args.seed,
            "include_gradient_locality": args.include_gradient_locality,
            "cache_features": args.cache_features,
            "data": asdict(data_config),
        },
    )
    print(f"wrote {records_path}", flush=True)


if __name__ == "__main__":
    main()
