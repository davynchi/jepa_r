#!/usr/bin/env python3
"""Replay final training batches and evaluate the nearly-free online LDA proxy."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jepa.analysis.factorization_v2 import lda_spectrum_metrics  # noqa: E402
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    TinyImageNetStaticImageDataset,
)
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    build_spatial_ijepa_core_from_metadata,
    load_spatial_checkpoint,
    normalize_ijepa_images,
)
from jepa.training.images.mask_loader import (  # noqa: E402
    _sample_resized_crop_theta,
    apply_prepared_crop,
)

from evaluate_spatial_ijepa_ss_factorization import checkpoint_epoch  # noqa: E402
from evaluate_upstream_ijepa_lda import atomic_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--online-size", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    return parser.parse_args()


def previous_weight_update(epoch: int, *, cadence: int, warmup: int) -> int | None:
    candidate = ((epoch - 1) // cadence) * cadence
    return candidate if candidate >= max(warmup, cadence) else None


def epoch_order(
    *,
    epoch: int,
    num_samples: int,
    num_draws: int,
    seed: int,
    method: str,
    warmup: int,
    probabilities: torch.Tensor | None,
) -> torch.Tensor:
    if method == "uniform" or epoch <= warmup:
        return torch.randperm(
            num_samples,
            generator=torch.Generator().manual_seed(
                derive_seed(seed, "order", epoch)
            ),
        )[:num_draws]
    if probabilities is None:
        probabilities = torch.full(
            (num_samples,), 1 / num_samples, dtype=torch.float64
        )
    return torch.multinomial(
        probabilities.float(),
        num_draws,
        replacement=True,
        generator=torch.Generator().manual_seed(
            derive_seed(seed, "weighted-order", epoch)
        ),
    )


@torch.inference_mode()
def replay_embeddings(
    encoder: torch.nn.Module,
    train_images: torch.Tensor,
    order: torch.Tensor,
    *,
    epoch: int,
    seed: int,
    batch_size: int,
    online_size: int,
    crop_scale: tuple[float, float],
    horizontal_flip_probability: float,
    image_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_batches = math.ceil(len(order) / batch_size)
    first_batch = max(num_batches - math.ceil(online_size / batch_size), 0)
    features = []
    retained_indices = []
    mask_epoch_seed = derive_seed(seed, "masks", epoch)
    encoder.eval()
    for batch_index in range(first_batch, num_batches):
        start = batch_index * batch_size
        indices = order[start : start + batch_size]
        batch = train_images[indices].to(device, non_blocking=True)
        mask_seed = derive_seed(mask_epoch_seed, "mask-batch", batch_index)
        transform_generator = torch.Generator().manual_seed(
            derive_seed(mask_seed, "transforms")
        )
        theta, flip = _sample_resized_crop_theta(
            batch_size=batch.shape[0],
            source_size=tuple(batch.shape[-2:]),
            scale=crop_scale,
            horizontal_flip_probability=horizontal_flip_probability,
            generator=transform_generator,
        )
        batch = apply_prepared_crop(
            batch,
            theta=theta,
            horizontal_flip=flip,
            output_size=image_size,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            tokens = encoder(batch)
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))
            pooled = tokens.mean(dim=1)
        features.append(pooled.float().cpu())
        retained_indices.append(indices)
    encoded = torch.cat(features)[-online_size:]
    indices = torch.cat(retained_indices)[-online_size:]
    return encoded, indices


def class_entropy(labels: torch.Tensor) -> float:
    counts = torch.bincount(labels)
    probabilities = counts[counts > 0].to(torch.float64)
    probabilities /= probabilities.sum()
    return float(-(probabilities * probabilities.log()).sum())


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text())
    spatial = config["spatial"]
    tiny = config["tiny_imagenet"]["data"]
    weighting = spatial["weighting"]
    method = str(weighting["method"])
    cadence = int(weighting["update_every_epochs"])
    warmup = int(weighting["warmup_epochs"])
    seed = int(spatial["seed"])
    batch_size = int(spatial["batch_size"])
    image_size = int(spatial["image_size"])
    crop_scale = tuple(float(value) for value in spatial["crop_scale"])
    flip_probability = float(spatial["horizontal_flip_probability"])
    if args.online_size <= 1:
        raise ValueError("online-size must exceed one")

    data_config = TinyImageNetDataConfig(
        root=str(Path(tiny["root"]).expanduser().resolve()),
        num_train_samples=int(tiny["num_train_samples"]),
        num_val_samples=1,
        num_test_samples=1,
        train_sample_seed=int(tiny["train_sample_seed"]),
    )
    train = TinyImageNetStaticImageDataset(data_config, "train")
    train_images = normalize_ijepa_images(train.images, inplace=True)
    train_labels = train.entities
    iterations_per_epoch = max(len(train) // batch_size, 1)
    num_draws = (
        len(train)
        if len(train) < batch_size
        else iterations_per_epoch * batch_size
    )
    if args.online_size > len(train):
        raise ValueError("online-size exceeds the training split")

    checkpoint_by_epoch = {
        checkpoint_epoch(path): path
        for path in (run_dir / "network").glob("epoch_*.pt")
    }
    missing = set(args.epochs) - set(checkpoint_by_epoch)
    if missing:
        raise FileNotFoundError(f"missing checkpoint epochs: {sorted(missing)}")
    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    records = []
    for epoch in sorted(args.epochs):
        started = time.time()
        path = checkpoint_by_epoch[epoch]
        checkpoint = load_spatial_checkpoint(path)
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"checkpoint has no model metadata: {path}")
        probabilities = None
        probability_source_epoch = None
        if method != "uniform":
            probability_source_epoch = previous_weight_update(
                epoch, cadence=cadence, warmup=warmup
            )
            if probability_source_epoch is not None:
                probability_checkpoint = load_spatial_checkpoint(
                    checkpoint_by_epoch[probability_source_epoch]
                )
                probabilities = probability_checkpoint["extra_state"][
                    "weighting_probabilities"
                ].to(torch.float64)
        order = epoch_order(
            epoch=epoch,
            num_samples=len(train),
            num_draws=num_draws,
            seed=seed,
            method=method,
            warmup=warmup,
            probabilities=probabilities,
        )
        core = build_spatial_ijepa_core_from_metadata(metadata)
        core.target_encoder.load_state_dict(checkpoint["target_encoder"])
        core.target_encoder.to(device)
        features, indices = replay_embeddings(
            core.target_encoder,
            train_images,
            order,
            epoch=epoch,
            seed=seed,
            batch_size=batch_size,
            online_size=args.online_size,
            crop_scale=crop_scale,
            horizontal_flip_probability=flip_probability,
            image_size=image_size,
            device=device,
            amp_dtype=amp_dtype,
        )
        labels = train_labels[indices]
        metrics = lda_spectrum_metrics(features, labels)
        metrics.update(
            {
                "sample_unique_fraction": float(indices.unique().numel() / len(indices)),
                "class_entropy": class_entropy(labels),
                "class_entropy_normalized": class_entropy(labels)
                / math.log(max(int(labels.unique().numel()), 2)),
            }
        )
        record = {
            "checkpoint": path.name,
            "epoch": epoch,
            "display_epoch": epoch,
            "probability_source_epoch": probability_source_epoch,
            "metrics": {
                name: {"mean": value, "std": 0.0}
                for name, value in metrics.items()
            },
            "seconds": time.time() - started,
        }
        records.append(record)
        atomic_json(args.output_dir / "records.json", records)
        print(
            f"epoch={epoch} lda_trace={metrics['lda_discriminative_trace']:.4f} "
            f"classes={int(metrics['num_classes_present'])} "
            f"unique={metrics['sample_unique_fraction']:.3f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        del core, features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_json(
        args.output_dir / "config.json",
        {
            "run_dir": str(run_dir),
            "epochs": sorted(args.epochs),
            "online_size": args.online_size,
            "method": method,
            "weight_update_cadence": cadence,
            "warmup": warmup,
            "batch_size": batch_size,
            "num_draws_per_epoch": num_draws,
            "crop_scale": list(crop_scale),
            "horizontal_flip_probability": flip_probability,
            "amp_dtype": args.amp_dtype,
            "replay_note": (
                "Exact indices/crops are reconstructed; embeddings use the end-of-epoch "
                "EMA target instead of its values during the final training batches."
            ),
        },
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
