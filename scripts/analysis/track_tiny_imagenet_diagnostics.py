#!/usr/bin/env python3
"""Watch a Tiny ImageNet spatial I-JEPA run and compute held-out diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from jepa.analysis.subspace import (  # noqa: E402
    classifier_accuracy,
    compute_latent_spectrum,
    fit_entity_classifier,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    build_tiny_imagenet_static_dataset_splits,
)
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    MaskConfig,
    build_spatial_ijepa_core_from_metadata,
    encode_samples_pooled,
    load_spatial_checkpoint,
    normalize_ijepa_images,
    sample_masks,
    spatial_ijepa_loss,
)
from jepa.training.images.spatial_logging import SpatialRunLogger  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "ijepa_spatial"
DEFAULT_BATCH_SIZE = 1024
POLL_SECONDS = 15
PROBE_RIDGE = 1.0e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help=f"Spatial run directory under {DEFAULT_OUTPUT_ROOT} or an absolute path",
    )
    parser.add_argument("--checkpoint-pattern", default="epoch_*.pt")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "none", "float16", "bfloat16"),
        default="auto",
        help="Autocast dtype; auto reuses the training run configuration",
    )
    parser.add_argument(
        "--resident-device-data",
        action="store_true",
        help="Keep normalized train/test images on the evaluation device",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for diagnostics; use cpu to avoid competing with training GPU",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def _resolve_run_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = DEFAULT_OUTPUT_ROOT / path
    return path.resolve()


def _held_out_loss(
    core,
    test_samples,
    grid,
    mask_config,
    device,
    *,
    batch_size: int,
    amp_dtype: torch.dtype | None,
) -> float:
    mask_generator = torch.Generator().manual_seed(derive_seed(0, "tiny-test-masks"))
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for batch in test_samples.split(batch_size):
            batch = batch.to(device, non_blocking=True)
            context_masks, target_masks = sample_masks(
                grid,
                grid,
                mask_config,
                mask_generator,
                batch_size=batch.shape[0],
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                loss = spatial_ijepa_loss(core, batch, context_masks, target_masks)
            total_loss += float(loss.item()) * batch.shape[0]
            total_examples += batch.shape[0]
    return total_loss / total_examples


def _encode_pooled(
    core,
    frames: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    encoded = []
    with torch.inference_mode():
        for batch in frames.split(batch_size):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                latents = encode_samples_pooled(core, batch)
            encoded.append(latents.float().cpu())
    return torch.cat(encoded, dim=0)


def _resolve_amp_dtype(args: argparse.Namespace, spatial_config: dict) -> torch.dtype | None:
    value = args.amp_dtype
    if value == "auto":
        value = str(spatial_config.get("amp_dtype", ""))
        if not value:
            value = "bfloat16" if spatial_config.get("bfloat16", False) else "none"
    return {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def _topk_accuracy(scores: torch.Tensor, labels: torch.Tensor, *, k: int) -> float:
    k = min(k, scores.shape[-1])
    predicted = scores.topk(k, dim=-1).indices
    return predicted.eq(labels.unsqueeze(-1)).any(dim=-1).to(torch.float64).mean().item()


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    checkpoint_dir = run_dir / "network"
    records_path = run_dir / "metrics" / "diagnostics.json"
    logger = SpatialRunLogger(run_dir, enable_tensorboard=not args.no_tensorboard)

    run_config_path = run_dir / "config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"missing run config: {run_config_path}")
    run_config = json.loads(run_config_path.read_text())
    spatial_config = run_config["spatial"]
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    patch_size = int(spatial_config["patch_size"])
    image_size = int(spatial_config.get("image_size", 64))
    data_config = TinyImageNetDataConfig(**run_config["tiny_imagenet"]["data"])
    datasets = build_tiny_imagenet_static_dataset_splits(data_config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_dtype = _resolve_amp_dtype(args, spatial_config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_frames = normalize_ijepa_images(datasets.train.images)
    test_frames = normalize_ijepa_images(datasets.test.images)
    if args.resident_device_data:
        train_frames = train_frames.to(device)
        test_frames = test_frames.to(device)
    train_labels = datasets.train.entities
    test_labels = datasets.test.entities
    num_classes = data_config.num_entities

    grid = image_size // patch_size
    mask_config = MaskConfig()

    records_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if records_path.exists():
        loaded_records = json.loads(records_path.read_text())
        if not isinstance(loaded_records, list):
            raise ValueError(f"diagnostics records must be a list: {records_path}")
        records = loaded_records
    seen: set[str] = {str(record["checkpoint"]) for record in records if isinstance(record, dict)}
    print(f"watching {checkpoint_dir}", flush=True)
    try:
        while True:
            if checkpoint_dir.exists():
                for path in sorted(checkpoint_dir.glob(args.checkpoint_pattern)):
                    if path.name in seen:
                        continue
                    seen.add(path.name)
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

                    test_loss = _held_out_loss(
                        core,
                        test_frames,
                        grid,
                        mask_config,
                        device,
                        batch_size=args.batch_size,
                        amp_dtype=amp_dtype,
                    )
                    train_z = _encode_pooled(
                        core,
                        train_frames,
                        batch_size=args.batch_size,
                        device=device,
                        amp_dtype=amp_dtype,
                    )
                    test_z = _encode_pooled(
                        core,
                        test_frames,
                        batch_size=args.batch_size,
                        device=device,
                        amp_dtype=amp_dtype,
                    )
                    spectrum = compute_latent_spectrum(test_z)

                    classifier = fit_entity_classifier(
                        train_z, train_labels, num_classes, ridge=PROBE_RIDGE
                    )
                    scores = classifier.probe.predict(test_z)
                    class_acc = classifier_accuracy(classifier, test_z, test_labels)
                    class_top5_acc = _topk_accuracy(scores, test_labels, k=5)
                    step = int(checkpoint.get("global_step") or 0)

                    record = {
                        "checkpoint": path.name,
                        "epoch": checkpoint["epoch"],
                        "global_step": step,
                        "num_classes": num_classes,
                        "test_loss": test_loss,
                        "effective_rank": spectrum.effective_rank,
                        "trace_covariance": spectrum.trace_covariance,
                        "top_eigenvalues": spectrum.eigenvalues[:6].tolist(),
                        "class_accuracy": class_acc,
                        "class_top5_accuracy": class_top5_acc,
                    }
                    records.append(record)
                    records_path.write_text(json.dumps(records, indent=2))
                    logger.log(
                        step=step,
                        epoch=int(checkpoint["epoch"]),
                        event="heldout_diagnostics",
                        scalars={
                            "diag/test_loss": test_loss,
                            "diag/effective_rank": spectrum.effective_rank,
                            "diag/trace_covariance": spectrum.trace_covariance,
                            "diag/class_accuracy": class_acc,
                            "diag/class_top5_accuracy": class_top5_acc,
                        },
                        histograms={"hist/diag_eigenvalues": torch.as_tensor(spectrum.eigenvalues)},
                    )
                    print(
                        f"checkpoint={path.name} epoch={record['epoch']:4d} "
                        f"test_loss={test_loss:9.5f} eff_rank={spectrum.effective_rank:6.3f} "
                        f"class_acc={class_acc:.3f} class_top5_acc={class_top5_acc:.3f}",
                        flush=True,
                    )
            if args.once:
                break
            time.sleep(args.poll_seconds)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
