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
from jepa.models.patches import patchify  # noqa: E402
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    MaskConfig,
    build_spatial_ijepa_core,
    encode_frames_pooled,
    load_spatial_checkpoint,
    sample_masks,
    spatial_ijepa_loss,
)
from jepa.training.images.spatial_logging import SpatialRunLogger  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "ijepa_spatial"
PATCH_SIZE = 8
PATCH_LATENT_DIM = 16
BATCH_SIZE = 128
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


def _held_out_loss(core, test_patches, grid, mask_config, device) -> float:
    mask_generator = torch.Generator().manual_seed(derive_seed(0, "tiny-test-masks"))
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in test_patches.split(BATCH_SIZE):
            batch = batch.to(device)
            context_masks, target_masks = sample_masks(grid, grid, mask_config, mask_generator)
            loss = torch.zeros((), device=device)
            for context_mask in context_masks:
                for target_mask in target_masks:
                    loss = loss + spatial_ijepa_loss(
                        core, batch, context_mask.to(device), target_mask.to(device)
                    )
            loss = loss / (len(context_masks) * len(target_masks))
            total_loss += float(loss.item()) * batch.shape[0]
            total_examples += batch.shape[0]
    return total_loss / total_examples


def _topk_accuracy(scores: torch.Tensor, labels: torch.Tensor, *, k: int) -> float:
    k = min(k, scores.shape[-1])
    predicted = scores.topk(k, dim=-1).indices
    return predicted.eq(labels.unsqueeze(-1)).any(dim=-1).to(torch.float64).mean().item()


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    records_path = run_dir / "metrics" / "diagnostics.json"
    logger = SpatialRunLogger(run_dir, enable_tensorboard=not args.no_tensorboard)

    run_config_path = run_dir / "config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"missing run config: {run_config_path}")
    run_config = json.loads(run_config_path.read_text())
    data_config = TinyImageNetDataConfig(**run_config["tiny_imagenet"]["data"])
    datasets = build_tiny_imagenet_static_dataset_splits(data_config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_frames = datasets.train.images
    test_frames = datasets.test.images
    train_labels = datasets.train.entities
    test_labels = datasets.test.entities
    num_classes = data_config.num_entities

    test_patches = patchify(test_frames, PATCH_SIZE)
    grid = 64 // PATCH_SIZE
    num_patches = test_patches.shape[1]
    patch_dim = test_patches.shape[2]
    mask_config = MaskConfig()

    records_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seen: set[str] = set()
    print(f"watching {checkpoint_dir}", flush=True)
    try:
        while True:
            if checkpoint_dir.exists():
                for path in sorted(checkpoint_dir.glob(args.checkpoint_pattern)):
                    if path.name in seen:
                        continue
                    seen.add(path.name)
                    checkpoint = load_spatial_checkpoint(path)
                    core = build_spatial_ijepa_core(
                        "cnn",
                        patch_dim=patch_dim,
                        patch_latent_dim=PATCH_LATENT_DIM,
                        num_patches=num_patches,
                    )
                    core.context_encoder.load_state_dict(checkpoint["context_encoder"])
                    core.predictor.load_state_dict(checkpoint["predictor"])
                    core.target_encoder.load_state_dict(checkpoint["target_encoder"])
                    core.context_encoder.to(device).eval()
                    core.predictor.to(device).eval()
                    core.target_encoder.to(device).eval()

                    test_loss = _held_out_loss(core, test_patches, grid, mask_config, device)
                    train_z = encode_frames_pooled(core, train_frames, patch_size=PATCH_SIZE).cpu()
                    test_z = encode_frames_pooled(core, test_frames, patch_size=PATCH_SIZE).cpu()
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
