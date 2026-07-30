#!/usr/bin/env python3
"""Evaluate exact full-image LDA metrics on official I-JEPA checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jepa.analysis.factorization_v2 import lda_spectrum_metrics  # noqa: E402
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    TinyImageNetStaticImageDataset,
)
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    build_spatial_ijepa_core,
    normalize_ijepa_images,
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
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    return parser.parse_args()


def strip_distributed_prefix(state: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key.removeprefix("module."), value) for key, value in state.items()
    )


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def checkpoint_number(path: Path) -> int:
    marker = path.name.removeprefix("jepa-ep").split(".", maxsplit=1)[0]
    return int(marker)


@torch.inference_mode()
def encode(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    result = []
    encoder.eval()
    for batch in images.split(batch_size):
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            tokens = encoder(batch.to(device, non_blocking=True))
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))
            pooled = tokens.mean(dim=1)
        result.append(pooled.float().cpu())
    return torch.cat(result)


def main() -> None:
    args = parse_args()
    if args.test_size <= 1 or args.batch_size <= 0:
        raise ValueError("test-size must exceed one and batch-size must be positive")
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
    labels = test.entities
    checkpoints = sorted(
        args.checkpoint_dir.glob(args.checkpoint_pattern),
        key=checkpoint_number,
    )
    if not checkpoints:
        raise FileNotFoundError("no matching official I-JEPA checkpoints")

    records = []
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
        features = encode(
            encoder,
            images,
            batch_size=args.batch_size,
            device=device,
            amp_dtype=amp_dtype,
        )
        metrics = lda_spectrum_metrics(features, labels)
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
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        del core, encoder, features
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
            "batch_size": args.batch_size,
            "amp_dtype": args.amp_dtype,
        },
    )
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
