#!/usr/bin/env python3
"""Replay final official I-JEPA train batches and measure online LDA."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor  # noqa: E402

from jepa.analysis.factorization_v2 import lda_spectrum_metrics  # noqa: E402
from jepa.analysis.upstream_online_lda import (  # noqa: E402
    checkpoint_epochs,
    checkpoint_number,
    final_epoch_indices,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.training.images.ijepa_spatial import build_spatial_ijepa_core  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tiny-imagenet-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-json", type=Path)
    parser.add_argument("--checkpoint-pattern", default="jepa-ep*.pth.tar")
    parser.add_argument("--online-size", type=int, default=4096)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--sampler-rank", type=int, default=0)
    parser.add_argument("--crop-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--crop-scale", type=float, nargs=2, default=(0.3, 1.0))
    parser.add_argument("--model-name", default="vit_small")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--predictor-embed-dim", type=int, default=192)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def strip_distributed_prefix(
    state: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key.removeprefix("module."), value) for key, value in state.items()
    )


def class_entropy(labels: torch.Tensor) -> float:
    counts = torch.bincount(labels)
    probabilities = counts[counts > 0].to(torch.float64)
    probabilities /= probabilities.sum()
    return float(-(probabilities * probabilities.log()).sum())


@torch.inference_mode()
def encode_selected(
    encoder: torch.nn.Module,
    dataset: ImageFolder,
    indices: torch.Tensor,
    *,
    transform: Compose,
    transform_seed: int,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_batches: list[torch.Tensor] = []
    labels: list[int] = []
    image_batch: list[torch.Tensor] = []
    encoder.eval()

    def flush() -> None:
        if not image_batch:
            return
        images = torch.stack(image_batch).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            tokens = encoder(images)
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))
            feature_batches.append(tokens.mean(dim=1).float().cpu())
        image_batch.clear()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(transform_seed)
        for index in indices.tolist():
            path, label = dataset.samples[index]
            image_batch.append(transform(dataset.loader(path)))
            labels.append(label)
            if len(image_batch) == batch_size:
                flush()
        flush()
    return torch.cat(feature_batches), torch.tensor(labels, dtype=torch.long)


def aggregate(repeats: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean([row[name] for row in repeats])),
            "std": float(np.std([row[name] for row in repeats])),
        }
        for name in repeats[0]
    }


def correlate_with_probes(
    records: list[dict[str, object]], probe_path: Path
) -> list[dict[str, object]]:
    probes = json.loads(probe_path.read_text())
    ratios = {
        int(record["epoch"]): float(
            record["metrics"]["between_total_trace_ratio"]["mean"]  # type: ignore[index]
        )
        for record in records
    }
    probe_names = sorted(
        {
            name
            for row in probes
            for name in row
            if name != "epoch" and name.endswith("top1")
        }
    )
    results = []
    for name in probe_names:
        aligned = [
            (ratios[int(row["epoch"])], float(row[name]))
            for row in probes
            if name in row and int(row["epoch"]) in ratios
        ]
        if len(aligned) < 3:
            continue
        metric, accuracy = (np.asarray(values) for values in zip(*aligned, strict=True))
        results.append(
            {
                "probe": name,
                "epochs": [
                    int(row["epoch"])
                    for row in probes
                    if name in row and int(row["epoch"]) in ratios
                ],
                "n": len(aligned),
                "pearson": float(pearsonr(metric, accuracy).statistic),
                "spearman": float(spearmanr(metric, accuracy).statistic),
                "first_difference_pearson": float(
                    pearsonr(np.diff(metric), np.diff(accuracy)).statistic
                ),
                "first_difference_spearman": float(
                    spearmanr(np.diff(metric), np.diff(accuracy)).statistic
                ),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if args.online_size <= 1:
        raise ValueError("online-size must exceed one")
    if args.train_batch_size <= 0 or args.encode_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not 0 < args.crop_scale[0] <= args.crop_scale[1] <= 1:
        raise ValueError("crop-scale must satisfy 0 < min <= max <= 1")
    if len(set(args.crop_seeds)) != len(args.crop_seeds):
        raise ValueError("crop-seeds must be unique")

    dataset = ImageFolder(args.tiny_imagenet_root.expanduser().resolve() / "train")
    transform = Compose(
        [
            RandomResizedCrop(args.image_size, scale=tuple(args.crop_scale)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    checkpoints = sorted(
        args.checkpoint_dir.expanduser().resolve().glob(args.checkpoint_pattern),
        key=checkpoint_number,
    )
    if not checkpoints:
        raise FileNotFoundError("no matching official I-JEPA checkpoints")

    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    records: list[dict[str, object]] = []
    for path in checkpoints:
        started = time.time()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        displayed_epoch, sampler_epoch = checkpoint_epochs(path, checkpoint)
        world_size = int(checkpoint.get("world_size", 1))
        checkpoint_batch_size = int(
            checkpoint.get("batch_size", args.train_batch_size)
        )
        if checkpoint_batch_size != args.train_batch_size:
            raise ValueError(
                f"{path.name}: checkpoint batch size {checkpoint_batch_size} "
                f"!= requested {args.train_batch_size}"
            )
        indices = final_epoch_indices(
            dataset,
            sampler_epoch=sampler_epoch,
            batch_size=args.train_batch_size,
            online_size=args.online_size,
            world_size=world_size,
            rank=args.sampler_rank,
            seed=args.sampler_seed,
        )

        core = build_spatial_ijepa_core(
            args.model_name,
            image_size=args.image_size,
            patch_size=args.patch_size,
            predictor_embed_dim=args.predictor_embed_dim,
            predictor_depth=args.predictor_depth,
        )
        core.target_encoder.load_state_dict(
            strip_distributed_prefix(checkpoint["target_encoder"])
        )
        core.target_encoder.to(device)

        repeat_metrics = []
        labels = None
        for crop_seed in args.crop_seeds:
            features, current_labels = encode_selected(
                core.target_encoder,
                dataset,
                indices,
                transform=transform,
                transform_seed=derive_seed(
                    args.sampler_seed,
                    "official-online-crops",
                    sampler_epoch,
                    crop_seed,
                ),
                batch_size=args.encode_batch_size,
                device=device,
                amp_dtype=amp_dtype,
            )
            if labels is not None and not torch.equal(labels, current_labels):
                raise RuntimeError("labels changed across crop repeats")
            labels = current_labels
            repeat_metrics.append(lda_spectrum_metrics(features, current_labels))

        assert labels is not None
        metrics = aggregate(repeat_metrics)
        metrics["sample_unique_fraction"] = {
            "mean": float(indices.unique().numel() / len(indices)),
            "std": 0.0,
        }
        entropy = class_entropy(labels)
        metrics["class_entropy_normalized"] = {
            "mean": entropy / math.log(max(int(labels.unique().numel()), 2)),
            "std": 0.0,
        }
        record = {
            "checkpoint": path.name,
            "epoch": displayed_epoch,
            "sampler_epoch_zero_based": sampler_epoch,
            "encoder": "target",
            "metrics": metrics,
            "seconds": time.time() - started,
        }
        records.append(record)
        atomic_json(args.output_dir / "records.json", records)
        ratio = metrics["between_total_trace_ratio"]
        print(
            f"epoch={displayed_epoch} between/total="
            f"{ratio['mean']:.6f}+/-{ratio['std']:.6f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        del core
        if device.type == "cuda":
            torch.cuda.empty_cache()

    config = {
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "tiny_imagenet_root": str(args.tiny_imagenet_root.expanduser().resolve()),
        "online_size": args.online_size,
        "train_batch_size": args.train_batch_size,
        "encode_batch_size": args.encode_batch_size,
        "sampler_seed": args.sampler_seed,
        "sampler_rank": args.sampler_rank,
        "crop_seeds": args.crop_seeds,
        "crop_scale": args.crop_scale,
        "encoder": "target",
        "replay_limit": (
            "Sampler order and drop_last are exact. Historical worker crop RNG and "
            "within-epoch EMA states were not saved; crop repeats and the final "
            "checkpoint target encoder provide an approximate replay."
        ),
    }
    atomic_json(args.output_dir / "config.json", config)
    if args.probe_json is not None:
        correlations = correlate_with_probes(
            records, args.probe_json.expanduser().resolve()
        )
        atomic_json(args.output_dir / "correlations.json", correlations)
        print(json.dumps(correlations, indent=2), flush=True)


if __name__ == "__main__":
    main()
