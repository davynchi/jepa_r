#!/usr/bin/env python3
"""Measure joint latent factorization of augmentations in official I-JEPA."""

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

from jepa.analysis.joint_transform_factorization import (  # noqa: E402
    apply_whitening,
    fit_joint_block_diagonalization,
    fit_linear_operator,
    fit_whitening_projection,
    remove_isotropic_component,
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


DEFAULT_TRANSFORMS = (
    "crop_small",
    "crop_medium",
    "crop_large",
    "color",
    "blur",
    "grayscale",
    "flip",
)
SUPPORTED_TRANSFORMS = DEFAULT_TRANSFORMS + ("color_fixed",)
CROP_SCALES = {
    "crop_small": (0.30, 0.50),
    "crop_medium": (0.50, 0.70),
    "crop_large": (0.70, 1.00),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", default="jepa-ep*.pth.tar")
    parser.add_argument("--epochs", type=int, nargs="*", default=None)
    parser.add_argument("--tiny-imagenet-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=("target", "context"), default="target")
    parser.add_argument("--model-name", default="vit_small")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--predictor-embed-dim", type=int, default=192)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--operator-train-size", type=int, default=6000)
    parser.add_argument("--operator-validation-size", type=int, default=2000)
    parser.add_argument("--pca-dimension", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument("--transforms", nargs="+", default=DEFAULT_TRANSFORMS)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--jbd-restarts", type=int, default=4)
    parser.add_argument("--jbd-steps", type=int, default=600)
    parser.add_argument("--jbd-learning-rate", type=float, default=0.05)
    parser.add_argument("--center-operators", action="store_true")
    parser.add_argument("--seed", type=int, default=4701)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--no-untrained", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def checkpoint_number(path: Path) -> int:
    marker = path.name.removeprefix("jepa-ep").split(".", maxsplit=1)[0]
    return int(marker)


def strip_distributed_prefix(
    state: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key.removeprefix("module."), value) for key, value in state.items()
    )


def _gaussian_blur(images: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    radius = 3
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=images.device,
        dtype=images.dtype,
    )
    kernel = torch.exp(-coordinates.square() / (2 * sigma**2))
    kernel /= kernel.sum()
    horizontal = kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    blurred = F.conv2d(
        F.pad(images, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=3,
    )
    return F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=3,
    )


def _color_transform(
    images: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = images.shape[0]
    brightness = 0.6 + 0.8 * torch.rand(batch_size, generator=generator)
    contrast = 0.6 + 0.8 * torch.rand(batch_size, generator=generator)
    saturation = 0.6 + 0.8 * torch.rand(batch_size, generator=generator)
    brightness = brightness.to(images.device)[:, None, None, None]
    contrast = contrast.to(images.device)[:, None, None, None]
    saturation = saturation.to(images.device)[:, None, None, None]
    result = images * brightness
    spatial_mean = result.mean(dim=(1, 2, 3), keepdim=True)
    result = (result - spatial_mean) * contrast + spatial_mean
    grayscale = (
        0.2989 * result[:, 0:1]
        + 0.5870 * result[:, 1:2]
        + 0.1140 * result[:, 2:3]
    )
    return ((result - grayscale) * saturation + grayscale).clamp(0, 1)


def _fixed_color_transform(images: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
    factor = 1.0 + strength
    result = images * factor
    spatial_mean = result.mean(dim=(1, 2, 3), keepdim=True)
    result = (result - spatial_mean) * factor + spatial_mean
    grayscale = (
        0.2989 * result[:, 0:1]
        + 0.5870 * result[:, 1:2]
        + 0.1140 * result[:, 2:3]
    )
    return ((result - grayscale) * factor + grayscale).clamp(0, 1)


def apply_transform(
    images: torch.Tensor,
    name: str,
    *,
    generator: torch.Generator,
    image_size: int,
) -> torch.Tensor:
    if name in CROP_SCALES:
        theta, flip = _sample_resized_crop_theta(
            batch_size=images.shape[0],
            source_size=tuple(images.shape[-2:]),
            scale=CROP_SCALES[name],
            horizontal_flip_probability=0.0,
            generator=generator,
        )
        return apply_prepared_crop(
            images,
            theta=theta,
            horizontal_flip=flip,
            output_size=image_size,
        )
    if name == "color":
        return _color_transform(images, generator=generator)
    if name == "color_fixed":
        return _fixed_color_transform(images)
    if name == "blur":
        return _gaussian_blur(images)
    if name == "grayscale":
        gray = (
            0.2989 * images[:, 0:1]
            + 0.5870 * images[:, 1:2]
            + 0.1140 * images[:, 2:3]
        )
        return gray.expand(-1, 3, -1, -1)
    if name == "flip":
        return images.flip(-1)
    raise ValueError(f"unknown transform: {name!r}")


@torch.inference_mode()
def encode_transform_pairs(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    transforms: tuple[str, ...],
    image_size: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sources: list[torch.Tensor] = []
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in transforms}
    encoder.eval()

    def encode(batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            tokens = encoder(normalize_ijepa_images(batch))
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))
            return tokens.mean(dim=1).float().cpu()

    for batch_index, cpu_batch in enumerate(images.split(batch_size)):
        batch = cpu_batch.to(device, non_blocking=True)
        sources.append(encode(batch))
        for name in transforms:
            generator = torch.Generator().manual_seed(
                derive_seed(seed, "joint-transform", name, batch_index)
            )
            targets[name].append(
                encode(
                    apply_transform(
                        batch,
                        name,
                        generator=generator,
                        image_size=image_size,
                    )
                )
            )
    return (
        torch.cat(sources),
        {name: torch.cat(parts) for name, parts in targets.items()},
    )


def split_indices(
    size: int,
    *,
    train_size: int,
    validation_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_size + validation_size >= size:
        raise ValueError("operator train and validation splits must leave a test split")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    return (
        order[:train_size],
        order[train_size : train_size + validation_size],
        order[train_size + validation_size :],
    )


def _operator_family(
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    ridge: float,
    shuffle: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float]]]:
    train_operators = []
    validation_operators = []
    diagnostics = {}
    for transform_index, (name, target) in enumerate(targets.items()):
        train_target = target[train_indices]
        validation_target = target[validation_indices]
        test_target = target[test_indices]
        if shuffle:
            generator = torch.Generator().manual_seed(
                derive_seed(seed, "shuffle", name, transform_index)
            )
            train_target = train_target[
                torch.randperm(train_target.shape[0], generator=generator)
            ]
            validation_target = validation_target[
                torch.randperm(validation_target.shape[0], generator=generator)
            ]
            test_target = test_target[
                torch.randperm(test_target.shape[0], generator=generator)
            ]
        train_fit = fit_linear_operator(
            source[train_indices],
            train_target,
            source[test_indices],
            test_target,
            ridge=ridge,
        )
        validation_fit = fit_linear_operator(
            source[validation_indices],
            validation_target,
            source[test_indices],
            test_target,
            ridge=ridge,
        )
        train_operators.append(train_fit.operator)
        validation_operators.append(validation_fit.operator)
        diagnostics[name] = {
            "train_operator_test_r2": train_fit.r_squared,
            "train_operator_test_nmse": train_fit.normalized_mse,
            "train_operator_energy": train_fit.energy_per_dimension,
            "validation_operator_test_r2": validation_fit.r_squared,
            "validation_operator_test_nmse": validation_fit.normalized_mse,
            "validation_operator_energy": validation_fit.energy_per_dimension,
            "train_operator_isotropic_energy_fraction": (
                float(train_fit.operator.trace().square() / train_fit.operator.shape[0])
                / max(float(train_fit.operator.square().sum()), 1e-12)
            ),
            "validation_operator_isotropic_energy_fraction": (
                float(
                    validation_fit.operator.trace().square()
                    / validation_fit.operator.shape[0]
                )
                / max(float(validation_fit.operator.square().sum()), 1e-12)
            ),
        }
    return (
        torch.stack(train_operators),
        torch.stack(validation_operators),
        diagnostics,
    )


def evaluate_embeddings(
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    pca_dimension: int,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    ridge: float,
    num_blocks: tuple[int, ...],
    jbd_restarts: int,
    jbd_steps: int,
    jbd_learning_rate: float,
    optimization_device: torch.device,
    seed: int,
    center_operators: bool,
) -> dict[str, object]:
    whitening = fit_whitening_projection(
        source[train_indices],
        dimension=pca_dimension,
    )
    source_white = apply_whitening(source, whitening)
    targets_white = {
        name: apply_whitening(target, whitening) for name, target in targets.items()
    }
    families = {}
    for shuffle in (False, True):
        key = "shuffled" if shuffle else "paired"
        train_operators, validation_operators, diagnostics = _operator_family(
            source_white,
            targets_white,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            ridge=ridge,
            shuffle=shuffle,
            seed=seed,
        )
        train_operators = train_operators.to(
            device=optimization_device,
            dtype=torch.float32,
        )
        validation_operators = validation_operators.to(
            device=optimization_device,
            dtype=torch.float32,
        )
        if center_operators:
            train_operators = remove_isotropic_component(train_operators)
            validation_operators = remove_isotropic_component(validation_operators)
        block_results = {}
        for block_count in num_blocks:
            fit = fit_joint_block_diagonalization(
                train_operators,
                validation_operators,
                num_blocks=block_count,
                restarts=jbd_restarts,
                steps=jbd_steps,
                learning_rate=jbd_learning_rate,
                seed=derive_seed(seed, key, "blocks", block_count),
            )
            block_results[str(block_count)] = {
                "train_factorization": fit.train_factorization,
                "validation_factorization": fit.validation_factorization,
                "random_validation_factorization": (
                    fit.random_validation_factorization
                ),
                "restart_validation_factorizations": list(
                    fit.restart_validation_factorizations
                ),
                "gain_over_random": (
                    fit.validation_factorization
                    - fit.random_validation_factorization
                )
                / max(1.0 - fit.random_validation_factorization, 1e-12),
            }
        families[key] = {
            "operator_diagnostics": diagnostics,
            "blocks": block_results,
        }
    return {
        "pca_retained_variance_fraction": whitening.retained_variance_fraction,
        "families": families,
    }


def build_encoder(
    *,
    checkpoint: dict[str, object] | None,
    encoder_name: str,
    model_name: str,
    image_size: int,
    patch_size: int,
    predictor_embed_dim: int,
    predictor_depth: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    core = build_spatial_ijepa_core(
        model_name,
        image_size=image_size,
        patch_size=patch_size,
        predictor_embed_dim=predictor_embed_dim,
        predictor_depth=predictor_depth,
    )
    if checkpoint is not None:
        core.context_encoder.load_state_dict(
            strip_distributed_prefix(checkpoint["encoder"])  # type: ignore[arg-type]
        )
        core.target_encoder.load_state_dict(
            strip_distributed_prefix(checkpoint["target_encoder"])  # type: ignore[arg-type]
        )
    encoder = (
        core.target_encoder if encoder_name == "target" else core.context_encoder
    )
    encoder.to(device)
    return core, encoder


def plot_records(records: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [int(row["display_epoch"]) for row in records]
    block_counts = sorted(
        int(value)
        for value in records[0]["result"]["families"]["paired"]["blocks"]  # type: ignore[index]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for block_count in block_counts:
        paired = [
            row["result"]["families"]["paired"]["blocks"][str(block_count)][  # type: ignore[index]
                "validation_factorization"
            ]
            for row in records
        ]
        shuffled = [
            row["result"]["families"]["shuffled"]["blocks"][str(block_count)][  # type: ignore[index]
                "validation_factorization"
            ]
            for row in records
        ]
        axes[0].plot(epochs, paired, marker="o", label=f"K={block_count}")
        random_basis = [
            row["result"]["families"]["paired"]["blocks"][str(block_count)][  # type: ignore[index]
                "random_validation_factorization"
            ]
            for row in records
        ]
        axes[0].plot(epochs, random_basis, linestyle="--", alpha=0.65)
        axes[0].plot(epochs, shuffled, linestyle=":", alpha=0.65)
    axes[0].set_title("Joint transform factorization")
    axes[0].set_xlabel("Epoch (0 = untrained)")
    axes[0].set_ylabel("$F_K$ on held-out operators")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[0].text(
        0.02,
        0.02,
        "solid: paired   dashed: random basis   dotted: shuffled pairs",
        transform=axes[0].transAxes,
        fontsize=8,
    )

    transforms = list(
        records[0]["result"]["families"]["paired"]["operator_diagnostics"]  # type: ignore[index]
    )
    for name in transforms:
        values = [
            row["result"]["families"]["paired"]["operator_diagnostics"][name][  # type: ignore[index]
                "train_operator_test_r2"
            ]
            for row in records
        ]
        axes[1].plot(epochs, values, marker="o", label=name)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Linear operator held-out fit")
    axes[1].set_xlabel("Epoch (0 = untrained)")
    axes[1].set_ylabel("$R^2$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    transforms = tuple(args.transforms)
    unknown = sorted(set(transforms) - set(SUPPORTED_TRANSFORMS))
    if unknown:
        raise ValueError(f"unknown transforms: {unknown}")
    if len(set(transforms)) != len(transforms):
        raise ValueError("transforms must be unique")
    if args.pca_dimension <= 0 or any(
        count <= 1 or args.pca_dimension % count for count in args.num_blocks
    ):
        raise ValueError("each num-blocks value must divide pca-dimension")
    if args.test_size <= 2:
        raise ValueError("test-size must exceed two")

    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    dataset = TinyImageNetStaticImageDataset(
        TinyImageNetDataConfig(
            root=str(args.tiny_imagenet_root.expanduser().resolve()),
            num_train_samples=1,
            num_val_samples=1,
            num_test_samples=args.test_size,
        ),
        "test",
    )
    train_indices, validation_indices, test_indices = split_indices(
        len(dataset),
        train_size=args.operator_train_size,
        validation_size=args.operator_validation_size,
        seed=derive_seed(args.seed, "operator-splits"),
    )
    checkpoints = sorted(
        args.checkpoint_dir.glob(args.checkpoint_pattern),
        key=checkpoint_number,
    )
    if args.epochs:
        wanted = set(args.epochs)
        checkpoints = [
            path for path in checkpoints if checkpoint_number(path) in wanted
        ]
    if not checkpoints:
        raise FileNotFoundError("no matching official I-JEPA checkpoints")

    work: list[tuple[str, int, dict[str, object] | None]] = []
    if not args.no_untrained:
        work.append(("untrained", 0, None))
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        work.append(
            (
                path.name,
                int(checkpoint["epoch"]) + 1,
                checkpoint,
            )
        )

    records: list[dict[str, object]] = []
    for checkpoint_name, display_epoch, checkpoint in work:
        started = time.time()
        torch.manual_seed(derive_seed(args.seed, "model", checkpoint_name))
        core, encoder = build_encoder(
            checkpoint=checkpoint,
            encoder_name=args.encoder,
            model_name=args.model_name,
            image_size=args.image_size,
            patch_size=args.patch_size,
            predictor_embed_dim=args.predictor_embed_dim,
            predictor_depth=args.predictor_depth,
            device=device,
        )
        source, targets = encode_transform_pairs(
            encoder,
            dataset.images,
            transforms=transforms,
            image_size=args.image_size,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            amp_dtype=amp_dtype,
        )
        result = evaluate_embeddings(
            source,
            targets,
            pca_dimension=args.pca_dimension,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            ridge=args.ridge,
            num_blocks=tuple(args.num_blocks),
            jbd_restarts=args.jbd_restarts,
            jbd_steps=args.jbd_steps,
            jbd_learning_rate=args.jbd_learning_rate,
            optimization_device=device,
            seed=derive_seed(args.seed, checkpoint_name),
            center_operators=args.center_operators,
        )
        record = {
            "checkpoint": checkpoint_name,
            "display_epoch": display_epoch,
            "encoder": args.encoder,
            "seconds": time.time() - started,
            "result": result,
        }
        records.append(record)
        atomic_json(args.output_dir / "records.json", records)
        paired = result["families"]["paired"]["blocks"]  # type: ignore[index]
        scores = " ".join(
            f"F_{count}={paired[str(count)]['validation_factorization']:.4f}"  # type: ignore[index]
            for count in args.num_blocks
        )
        print(
            f"checkpoint={checkpoint_name} epoch={display_epoch} {scores} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        del core, encoder, source, targets
        if device.type == "cuda":
            torch.cuda.empty_cache()

    config = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint_pattern": args.checkpoint_pattern,
        "epochs": args.epochs,
        "tiny_imagenet_root": str(args.tiny_imagenet_root.resolve()),
        "encoder": args.encoder,
        "model_name": args.model_name,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "predictor_embed_dim": args.predictor_embed_dim,
        "predictor_depth": args.predictor_depth,
        "test_size": args.test_size,
        "operator_train_size": args.operator_train_size,
        "operator_validation_size": args.operator_validation_size,
        "operator_test_size": int(test_indices.numel()),
        "pca_dimension": args.pca_dimension,
        "num_blocks": args.num_blocks,
        "transforms": transforms,
        "crop_scales": CROP_SCALES,
        "ridge": args.ridge,
        "jbd_restarts": args.jbd_restarts,
        "jbd_steps": args.jbd_steps,
        "jbd_learning_rate": args.jbd_learning_rate,
        "center_operators": args.center_operators,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "amp_dtype": args.amp_dtype,
        "untrained": not args.no_untrained,
    }
    atomic_json(args.output_dir / "config.json", config)
    plot_records(records, args.output_dir / "joint_transform_factorization.png")
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
