#!/usr/bin/env python3
"""Evaluate label-free crop factorization on official I-JEPA checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jepa.analysis.factorization_v2 import (  # noqa: E402
    factorization_metrics,
    fit_view_factorization,
    principal_subspace_similarity,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    TinyImageNetStaticImageDataset,
)
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    build_spatial_ijepa_core,
    normalize_ijepa_images,
)
from jepa.training.images.mask_loader import (  # noqa: E402
    _sample_resized_crop_theta,
    apply_prepared_crop,
)

from evaluate_upstream_ijepa_lda import (  # noqa: E402
    atomic_json,
    checkpoint_number,
    strip_distributed_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", default="jepa-ep*.pth.tar")
    parser.add_argument("--tiny-imagenet-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=("target", "context"), default="target")
    parser.add_argument("--model-name", default="vit_small")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--predictor-embed-dim", type=int, default=192)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--crop-scale", type=float, nargs=2, default=(0.3, 1.0))
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.0)
    parser.add_argument("--subspace-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    return parser.parse_args()


@torch.inference_mode()
def encode_crop_views(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    num_views: int,
    crop_scale: tuple[float, float],
    horizontal_flip_probability: float,
    image_size: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    views: list[list[torch.Tensor]] = [[] for _ in range(num_views)]
    encoder.eval()
    for batch_index, batch in enumerate(images.split(batch_size)):
        batch = batch.to(device, non_blocking=True)
        for view_index in range(num_views):
            generator = torch.Generator().manual_seed(
                derive_seed(seed, "crop", view_index, batch_index)
            )
            theta, flip = _sample_resized_crop_theta(
                batch_size=batch.shape[0],
                source_size=tuple(batch.shape[-2:]),
                scale=crop_scale,
                horizontal_flip_probability=horizontal_flip_probability,
                generator=generator,
            )
            augmented = apply_prepared_crop(
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
                tokens = encoder(augmented)
                tokens = F.layer_norm(tokens, (tokens.shape[-1],))
                pooled = tokens.mean(dim=1)
            views[view_index].append(pooled.float().cpu())
    return torch.stack([torch.cat(parts) for parts in views])


def main() -> None:
    args = parse_args()
    if args.test_size <= 1 or args.num_views < 2 or args.batch_size <= 0:
        raise ValueError("test-size > 1, num-views >= 2, and positive batch-size required")
    if (
        len(args.crop_scale) != 2
        or not 0 < args.crop_scale[0] <= args.crop_scale[1] <= 1
    ):
        raise ValueError("crop-scale must satisfy 0 < min <= max <= 1")
    if not 0 <= args.horizontal_flip_probability <= 1:
        raise ValueError("horizontal-flip-probability must be in [0, 1]")

    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    data_config = TinyImageNetDataConfig(
        root=str(args.tiny_imagenet_root.expanduser().resolve()),
        num_train_samples=1,
        num_val_samples=1,
        num_test_samples=args.test_size,
    )
    test = TinyImageNetStaticImageDataset(data_config, "test")
    images = normalize_ijepa_images(test.images, inplace=True)
    checkpoints = sorted(
        args.checkpoint_dir.glob(args.checkpoint_pattern),
        key=checkpoint_number,
    )
    if not checkpoints:
        raise FileNotFoundError("no matching official I-JEPA checkpoints")

    records = []
    previous = None
    for path in checkpoints:
        started = time.time()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        core = build_spatial_ijepa_core(
            args.model_name,
            image_size=args.image_size,
            patch_size=args.patch_size,
            predictor_embed_dim=args.predictor_embed_dim,
            predictor_depth=args.predictor_depth,
        )
        core.context_encoder.load_state_dict(
            strip_distributed_prefix(checkpoint["encoder"])
        )
        core.target_encoder.load_state_dict(
            strip_distributed_prefix(checkpoint["target_encoder"])
        )
        encoder = (
            core.target_encoder if args.encoder == "target" else core.context_encoder
        )
        encoder.to(device)
        views = encode_crop_views(
            encoder,
            images,
            num_views=args.num_views,
            crop_scale=tuple(args.crop_scale),
            horizontal_flip_probability=args.horizontal_flip_probability,
            image_size=args.image_size,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            amp_dtype=amp_dtype,
        )
        factorization = fit_view_factorization(views)
        metrics = {
            f"ss_crop_{name}": value
            for name, value in factorization_metrics(factorization).items()
        }
        if previous is not None:
            metrics["ss_crop_invariant_subspace_similarity_previous"] = (
                principal_subspace_similarity(
                    factorization,
                    previous,
                    rank=args.subspace_rank,
                )
            )
        previous = factorization
        record = {
            "checkpoint": path.name,
            "epoch": int(checkpoint["epoch"]),
            "display_epoch": int(checkpoint["epoch"]) + 1,
            "encoder": args.encoder,
            "metrics": {
                name: {"mean": value, "std": 0.0}
                for name, value in metrics.items()
            },
            "seconds": time.time() - started,
        }
        records.append(record)
        atomic_json(args.output_dir / "records.json", records)
        print(
            f"checkpoint={path.name} epoch={record['display_epoch']} "
            f"invariance={metrics['ss_crop_invariance_mean']:.4f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        del core, encoder, views
        if device.type == "cuda":
            torch.cuda.empty_cache()

    atomic_json(
        args.output_dir / "config.json",
        {
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "checkpoint_pattern": args.checkpoint_pattern,
            "tiny_imagenet_root": str(args.tiny_imagenet_root.resolve()),
            "encoder": args.encoder,
            "model_name": args.model_name,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "test_size": args.test_size,
            "num_views": args.num_views,
            "crop_scale": list(args.crop_scale),
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "subspace_rank": args.subspace_rank,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "amp_dtype": args.amp_dtype,
        },
    )
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
