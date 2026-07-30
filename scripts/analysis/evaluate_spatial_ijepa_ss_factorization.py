#!/usr/bin/env python3
"""Evaluate label-free crop factorization on project spatial-I-JEPA runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from jepa.analysis.factorization_v2 import (  # noqa: E402
    factorization_metrics,
    fit_view_factorization,
    principal_subspace_similarity,
)
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    TinyImageNetStaticImageDataset,
)
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    build_spatial_ijepa_core_from_metadata,
    load_spatial_checkpoint,
    normalize_ijepa_images,
)

from evaluate_upstream_ijepa_lda import atomic_json  # noqa: E402
from evaluate_upstream_ijepa_ss_factorization import encode_crop_views  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+")
    parser.add_argument("--encoder", choices=("target", "context"), default="target")
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


def checkpoint_epoch(path: Path) -> int:
    return int(path.stem.removeprefix("epoch_"))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text())
    spatial = config["spatial"]
    tiny = config["tiny_imagenet"]["data"]
    image_size = int(spatial["image_size"])
    patch_size = int(spatial["patch_size"])
    data_config = TinyImageNetDataConfig(
        root=str(Path(tiny["root"]).expanduser().resolve()),
        num_train_samples=1,
        num_val_samples=1,
        num_test_samples=args.test_size,
    )
    test = TinyImageNetStaticImageDataset(data_config, "test")
    images = normalize_ijepa_images(test.images, inplace=True)
    wanted = set(args.epochs or ())
    checkpoints = sorted(
        (run_dir / "network").glob("epoch_*.pt"),
        key=checkpoint_epoch,
    )
    if wanted:
        checkpoints = [path for path in checkpoints if checkpoint_epoch(path) in wanted]
        missing = wanted - {checkpoint_epoch(path) for path in checkpoints}
        if missing:
            raise FileNotFoundError(f"missing requested checkpoint epochs: {sorted(missing)}")
    if not checkpoints:
        raise FileNotFoundError("no matching project checkpoints")

    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    records = []
    previous = None
    for path in checkpoints:
        started = time.time()
        checkpoint = load_spatial_checkpoint(path)
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"checkpoint has no model metadata: {path}")
        core = build_spatial_ijepa_core_from_metadata(metadata)
        core.context_encoder.load_state_dict(checkpoint["context_encoder"])
        core.target_encoder.load_state_dict(checkpoint["target_encoder"])
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
            image_size=image_size,
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
                    factorization, previous, rank=args.subspace_rank
                )
            )
        previous = factorization
        record = {
            "checkpoint": path.name,
            "epoch": int(checkpoint["epoch"]),
            "display_epoch": int(checkpoint["epoch"]),
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
            "run_dir": str(run_dir),
            "epochs": sorted(wanted) if wanted else None,
            "encoder": args.encoder,
            "test_size": args.test_size,
            "num_views": args.num_views,
            "crop_scale": list(args.crop_scale),
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "subspace_rank": args.subspace_rank,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "amp_dtype": args.amp_dtype,
            "image_size": image_size,
            "patch_size": patch_size,
        },
    )
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
