#!/usr/bin/env python3
"""Evaluate robust mask-view factorization diagnostics on Tiny ImageNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from jepa.analysis.factorization_v2 import (  # noqa: E402
    factorization_metrics,
    fit_view_factorization,
    principal_subspace_similarity,
    supervised_alignment_metrics,
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
)

PROTOCOL_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", default="epoch_*.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--encoder", choices=("target", "context"), default="target")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fit-size", type=int, default=4096)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--mask-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--subspace-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def epoch_key(path: Path) -> tuple[int, str]:
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else -1, path.name)


def protocol_payload(args: argparse.Namespace, mask_config: MaskConfig) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "encoder": args.encoder,
        "amp_dtype": args.amp_dtype,
        "fit_size": args.fit_size,
        "num_views": args.num_views,
        "mask_seeds": list(args.mask_seeds),
        "subspace_rank": args.subspace_rank,
        "seed": args.seed,
        "mask_config": vars(mask_config),
    }


def protocol_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()[:12]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
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
def encode_views(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    grid: int,
    mask_config: MaskConfig,
    batch_size: int,
    num_views: int,
    seed: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    generators = [
        torch.Generator().manual_seed(derive_seed(seed, f"view-{index}"))
        for index in range(num_views)
    ]
    views: list[list[torch.Tensor]] = [[] for _ in range(num_views)]
    for batch in images.split(batch_size):
        batch = batch.to(device, non_blocking=True)
        for index, generator in enumerate(generators):
            context_masks, _ = sample_masks(
                grid, grid, mask_config, generator, batch_size=batch.shape[0]
            )
            masks = [mask.to(device, non_blocking=True) for mask in context_masks]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                encoded = encoder(batch, masks).mean(dim=1)
            if encoded.shape[0] != batch.shape[0]:
                raise RuntimeError(
                    "factorization v2 requires exactly one context mask per image"
                )
            views[index].append(encoded.float().cpu())
    return torch.stack([torch.cat(parts) for parts in views])


def aggregate_seed_metrics(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in values[0]:
        tensor = torch.tensor([item[key] for item in values], dtype=torch.float64)
        result[key] = {
            "mean": float(tensor.mean()),
            "std": float(tensor.std(unbiased=False)),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.fit_size <= 1 or args.num_views < 2 or not args.mask_seeds:
        raise ValueError("fit-size > 1, num-views >= 2, and mask-seeds are required")
    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text())
    spatial = config["spatial"]
    image_size = int(spatial.get("image_size", 64))
    patch_size = int(spatial["patch_size"])
    grid = image_size // patch_size
    mask_config = MaskConfig(
        min_keep=int(spatial.get("mask_min_keep") or (10 if grid >= 16 else 4))
    )
    protocol = protocol_payload(args, mask_config)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "factorization_v2" / protocol_id(protocol)
    )
    atomic_json(output_dir / "protocol.json", protocol)

    data_config = TinyImageNetDataConfig(**config["tiny_imagenet"]["data"])
    train = build_tiny_imagenet_static_dataset_splits(data_config).train
    if args.fit_size > len(train):
        raise ValueError(f"fit-size {args.fit_size} exceeds train size {len(train)}")
    order = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(args.seed)
    )[: args.fit_size]
    images = normalize_ijepa_images(train.images[order], inplace=False)
    labels = train.entities[order]

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    checkpoints = sorted(
        (run_dir / "network").glob(args.checkpoint_pattern), key=epoch_key
    )
    if not checkpoints:
        raise FileNotFoundError("no matching checkpoints")

    records: list[dict[str, Any]] = []
    previous_by_seed: dict[int, Any] = {}
    for path in checkpoints:
        started = time.time()
        core, checkpoint = load_core(path, device)
        encoder = core.target_encoder if args.encoder == "target" else core.context_encoder
        seed_metrics = []
        current_by_seed = {}
        for mask_seed in args.mask_seeds:
            cache = output_dir / "cache" / f"{path.stem}_maskseed{mask_seed}.pt"
            if args.cache_features and cache.exists():
                views = torch.load(cache, map_location="cpu", weights_only=True)
            else:
                views = encode_views(
                    encoder,
                    images,
                    grid=grid,
                    mask_config=mask_config,
                    batch_size=args.batch_size,
                    num_views=args.num_views,
                    seed=mask_seed,
                    device=device,
                    amp_dtype=amp_dtype,
                )
                if args.cache_features:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(views, cache)
            factorization = fit_view_factorization(views)
            metrics = factorization_metrics(factorization)
            metrics.update(
                supervised_alignment_metrics(
                    views.mean(dim=0), labels, factorization,
                    maximum_rank=args.subspace_rank,
                )
            )
            previous = previous_by_seed.get(mask_seed)
            metrics["invariant_subspace_similarity_previous"] = (
                principal_subspace_similarity(
                    factorization, previous, rank=args.subspace_rank
                )
                if previous is not None else float("nan")
            )
            seed_metrics.append(metrics)
            current_by_seed[mask_seed] = factorization
        previous_by_seed = current_by_seed
        # Exclude the first-checkpoint temporal NaN from JSON aggregates.
        clean_metrics = [
            {key: value for key, value in item.items() if torch.isfinite(torch.tensor(value))}
            for item in seed_metrics
        ]
        common_keys = set.intersection(*(set(item) for item in clean_metrics))
        aggregated = aggregate_seed_metrics(
            [{key: item[key] for key in common_keys} for item in clean_metrics]
        )
        record = {
            "checkpoint": path.name,
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint.get("global_step") or checkpoint["epoch"]),
            "metrics": aggregated,
            "mask_seed_metrics": [
                {key: value for key, value in item.items() if torch.isfinite(torch.tensor(value))}
                for item in seed_metrics
            ],
            "seconds": time.time() - started,
        }
        records.append(record)
        atomic_json(output_dir / "records.json", records)
        print(
            f"epoch={record['epoch']} seconds={record['seconds']:.1f} "
            f"invariance={aggregated['invariance_mean']['mean']:.4f}",
            flush=True,
        )
        del core
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(output_dir, flush=True)


if __name__ == "__main__":
    main()
