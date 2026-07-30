#!/usr/bin/env python3
"""Evaluate corrected temporal factorization metrics on official I-JEPA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
from evaluate_upstream_ijepa_joint_transform_factorization import (  # noqa: E402
    CROP_SCALES,
    DEFAULT_TRANSFORMS,
    build_encoder,
    checkpoint_number,
    encode_transform_pairs,
    split_indices,
)

from jepa.analysis.joint_transform_factorization import (  # noqa: E402
    apply_whitening,
    fit_joint_block_diagonalization,
    fit_linear_operator,
    fit_whitening_projection,
    remove_isotropic_component,
)
from jepa.analysis.temporal_factorization import (  # noqa: E402
    apply_affine_alignment,
    block_gaussian_statistics,
    cross_block_gaussian_mi,
    fit_orthogonal_procrustes,
    linear_cka,
    match_subspaces,
    perturbation_concentration,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    TinyImageNetStaticImageDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", default="jepa-ep*.pth.tar")
    parser.add_argument("--epochs", type=int, nargs="+", default=(50, 100, 150, 200, 250, 300))
    parser.add_argument("--reference-epoch", type=int, default=50)
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
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=("crop_small", "crop_medium", "crop_large", "color", "blur", "flip"),
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--jbd-restarts", type=int, default=4)
    parser.add_argument("--jbd-steps", type=int, default=600)
    parser.add_argument("--jbd-learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=4701)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--cache-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def embedding_cache_path(output_dir: Path, checkpoint: Path, encoder: str) -> Path:
    return output_dir / "cache" / f"{checkpoint.stem}_{encoder}_transform_embeddings.pt"


def load_or_encode(
    checkpoint_path: Path,
    *,
    images: torch.Tensor,
    transforms: tuple[str, ...],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cache_path = embedding_cache_path(args.output_dir, checkpoint_path, args.encoder)
    if args.cache_embeddings and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cached.get("transforms") == transforms
            and int(cached.get("seed", -1)) == args.seed
            and int(cached.get("num_images", -1)) == images.shape[0]
        ):
            return cached["source"], cached["targets"]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
        images,
        transforms=transforms,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        amp_dtype=amp_dtype,
    )
    if args.cache_embeddings:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        torch.save(
            {
                "source": source,
                "targets": targets,
                "transforms": transforms,
                "seed": args.seed,
                "num_images": images.shape[0],
            },
            temporary,
        )
        temporary.replace(cache_path)
    del core, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return source, targets


def align_to_reference(
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    reference_source: torch.Tensor,
    *,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    procrustes = fit_orthogonal_procrustes(
        source[train_indices],
        reference_source[train_indices],
        source[test_indices],
        reference_source[test_indices],
    )
    source_mean = source[train_indices].to(torch.float64).mean(dim=0)
    reference_mean = reference_source[train_indices].to(torch.float64).mean(dim=0)
    aligned_source = apply_affine_alignment(
        source,
        source_mean=source_mean,
        target_mean=reference_mean,
        rotation=procrustes.rotation,
    )
    aligned_targets = {
        name: apply_affine_alignment(
            target,
            source_mean=source_mean,
            target_mean=reference_mean,
            rotation=procrustes.rotation,
        )
        for name, target in targets.items()
    }
    return (
        aligned_source,
        aligned_targets,
        {
            "train_normalized_residual": procrustes.train_normalized_residual,
            "test_normalized_residual": procrustes.validation_normalized_residual,
        },
    )


def representation_drift(
    current: torch.Tensor,
    previous: torch.Tensor,
    reference: torch.Tensor,
    *,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
) -> dict[str, Any]:
    def compare(target: torch.Tensor) -> dict[str, float]:
        fit = fit_orthogonal_procrustes(
            current[train_indices],
            target[train_indices],
            current[test_indices],
            target[test_indices],
        )
        return {
            "linear_cka_test": linear_cka(
                current[test_indices],
                target[test_indices],
            ),
            "procrustes_train_residual": fit.train_normalized_residual,
            "procrustes_test_residual": fit.validation_normalized_residual,
        }

    return {
        "q21_vs_previous": compare(previous),
        "q21_vs_reference": compare(reference),
    }


def common_subspace_energy_fraction(
    features: torch.Tensor,
    projection: torch.Tensor,
) -> float:
    matrix = features.detach().cpu().to(torch.float64)
    matrix -= matrix.mean(dim=0)
    pca_basis, _ = torch.linalg.qr(projection, mode="reduced")
    return float(
        (matrix @ pca_basis).square().sum()
        / matrix.square().sum().clamp_min(1e-12)
    )


def fit_operator_families(
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float]]]:
    train_operators = []
    validation_operators = []
    diagnostics = {}
    for name, target in targets.items():
        train_fit = fit_linear_operator(
            source[train_indices],
            target[train_indices],
            source[test_indices],
            target[test_indices],
            ridge=ridge,
        )
        validation_fit = fit_linear_operator(
            source[validation_indices],
            target[validation_indices],
            source[test_indices],
            target[test_indices],
            ridge=ridge,
        )
        train_operators.append(train_fit.operator)
        validation_operators.append(validation_fit.operator)
        diagnostics[name] = {
            "q17_test_r2": train_fit.r_squared,
            "q17_test_nmse": train_fit.normalized_mse,
            "q17_intercept_norm": float(torch.linalg.vector_norm(train_fit.intercept)),
            "validation_fit_test_r2": validation_fit.r_squared,
            "operator_energy_per_dimension": train_fit.energy_per_dimension,
        }
    return (
        remove_isotropic_component(torch.stack(train_operators)),
        remove_isotropic_component(torch.stack(validation_operators)),
        diagnostics,
    )


def temporal_payload(
    current_basis: torch.Tensor,
    previous_basis: torch.Tensor | None,
    *,
    num_blocks: int,
) -> dict[str, Any]:
    if previous_basis is None:
        return {
            "q5_similarity": None,
            "q6_grassmann_distance": None,
            "q22_set_distance": None,
            "matching": None,
            "principal_angles": None,
            "null_reason": "no_previous_checkpoint",
        }
    metrics = match_subspaces(
        current_basis,
        previous_basis,
        num_blocks=num_blocks,
    )
    return {
        "q5_similarity": metrics.similarity,
        "q6_grassmann_distance": metrics.grassmann_distance,
        # Q22 is the aggregate distance between optimally matched unordered sets.
        "q22_set_distance": metrics.grassmann_distance,
        "matching": list(metrics.assignment),
        "per_block_similarity": list(metrics.per_block_similarity),
        "per_block_distance": list(metrics.per_block_distance),
        "principal_angles": {
            "degrees_by_block": [list(row) for row in metrics.principal_angles_degrees],
            "mean_degrees": metrics.principal_angle_mean_degrees,
            "max_degrees": metrics.principal_angle_max_degrees,
            "rms_sine": metrics.principal_angle_rms_sine,
        },
        "null_reason": None,
    }


def q12_payload(
    current: dict[str, object],
    previous: dict[str, object] | None,
    matching: list[int] | None,
) -> dict[str, Any]:
    if previous is None or matching is None:
        return {"value": None, "null_reason": "no_previous_checkpoint"}
    current_mean = current["weighted_mean"]
    previous_mean = previous["weighted_mean"]
    current_blocks = current["blocks"]
    previous_blocks = previous["blocks"]
    entropy_signed = float(current_mean["gaussian_entropy_per_dimension"]) - float(
        previous_mean["gaussian_entropy_per_dimension"]
    )
    rank_signed = float(current_mean["normalized_effective_rank"]) - float(
        previous_mean["normalized_effective_rank"]
    )
    return {
        "value": abs(entropy_signed),
        "signed_entropy_change": entropy_signed,
        "signed_normalized_effective_rank_change": rank_signed,
        "matched_block_entropy_changes": [
            float(current_blocks[index]["gaussian_entropy_per_dimension"])
            - float(previous_blocks[matching[index]]["gaussian_entropy_per_dimension"])
            for index in range(len(matching))
        ],
        "null_reason": None,
    }


def plot_records(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in records]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for block_count in sorted(int(key) for key in records[0]["blocks"]):
        key = str(block_count)
        axes[0, 0].plot(
            epochs,
            [record["blocks"][key]["temporal"]["q5_similarity"] for record in records],
            marker="o",
            label=f"K={key}",
        )
        axes[0, 1].plot(
            epochs,
            [
                record["blocks"][key]["q10"]["weighted_mean"][
                    "normalized_effective_rank"
                ]
                for record in records
            ],
            marker="o",
            label=f"K={key}",
        )
        axes[0, 2].plot(
            epochs,
            [record["blocks"][key]["q11"]["normalized_mean"] for record in records],
            marker="o",
            label=f"K={key}",
        )
    axes[0, 0].set_title("Q5 matched subspace similarity")
    axes[0, 1].set_title("Q10 normalized effective rank")
    axes[0, 2].set_title("Q11 cross-block Gaussian MI")

    transforms = list(records[0]["q17"])
    for name in transforms:
        axes[1, 0].plot(
            epochs,
            [record["q17"][name]["q17_test_r2"] for record in records],
            marker="o",
            label=name,
        )
        axes[1, 1].plot(
            epochs,
            [
                record["blocks"]["4"]["q18"][name]["mean_concentration"]
                for record in records
            ],
            marker="o",
            label=name,
        )
    axes[1, 0].set_title("Q17 affine operator R2")
    axes[1, 1].set_title("Q18 perturbation concentration (K=4)")
    axes[1, 2].plot(
        epochs,
        [record["q21"]["q21_vs_reference"]["linear_cka_test"] for record in records],
        marker="o",
        label="CKA vs epoch 50",
    )
    axes[1, 2].plot(
        epochs,
        [
            record["q21"]["q21_vs_reference"]["procrustes_test_residual"]
            for record in records
        ],
        marker="s",
        label="Procrustes residual",
    )
    axes[1, 2].set_title("Q21 representation drift")

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    transforms = tuple(args.transforms)
    unknown = sorted(set(transforms) - set(DEFAULT_TRANSFORMS))
    if unknown:
        raise ValueError(f"unknown transforms: {unknown}")
    if "grayscale" in transforms:
        raise ValueError("grayscale is excluded because its held-out linear R2 is negative")
    if args.reference_epoch not in args.epochs:
        raise ValueError("reference-epoch must be included in epochs")
    if any(count <= 1 or args.pca_dimension % count for count in args.num_blocks):
        raise ValueError("each num-blocks value must divide pca-dimension")

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
    wanted = set(args.epochs)
    checkpoints = {
        checkpoint_number(path): path
        for path in args.checkpoint_dir.glob(args.checkpoint_pattern)
        if checkpoint_number(path) in wanted
    }
    if set(checkpoints) != wanted:
        raise FileNotFoundError(f"missing checkpoints: {sorted(wanted - set(checkpoints))}")

    reference_path = checkpoints[args.reference_epoch]
    reference_source, _ = load_or_encode(
        reference_path,
        images=dataset.images,
        transforms=transforms,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    reference_whitening = fit_whitening_projection(
        reference_source[train_indices],
        dimension=args.pca_dimension,
    )

    records = []
    previous_source: torch.Tensor | None = None
    previous_bases: dict[str, torch.Tensor] = {}
    previous_q10: dict[str, dict[str, object]] = {}
    state_dir = args.output_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    for epoch in sorted(args.epochs):
        started = time.time()
        path = checkpoints[epoch]
        source, targets = load_or_encode(
            path,
            images=dataset.images,
            transforms=transforms,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
        )
        q21 = representation_drift(
            source,
            previous_source if previous_source is not None else source,
            reference_source,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        if previous_source is None:
            q21["q21_vs_previous"]["null_reason"] = "no_previous_checkpoint"
        aligned_source, aligned_targets, alignment = align_to_reference(
            source,
            targets,
            reference_source,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        source_white = apply_whitening(aligned_source, reference_whitening)
        targets_white = {
            name: apply_whitening(target, reference_whitening)
            for name, target in aligned_targets.items()
        }
        train_operators, validation_operators, q17 = fit_operator_families(
            source_white,
            targets_white,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            ridge=args.ridge,
        )
        train_operators = train_operators.to(device=device, dtype=torch.float32)
        validation_operators = validation_operators.to(
            device=device,
            dtype=torch.float32,
        )
        block_records = {}
        basis_state = {}
        for block_count in args.num_blocks:
            key = str(block_count)
            fit = fit_joint_block_diagonalization(
                train_operators,
                validation_operators,
                num_blocks=block_count,
                restarts=args.jbd_restarts,
                steps=args.jbd_steps,
                learning_rate=args.jbd_learning_rate,
                # Shared restarts avoid injecting epoch-specific optimizer noise
                # into the temporal subspace comparison.
                seed=derive_seed(args.seed, "temporal", block_count),
            )
            basis = fit.basis.double().cpu()
            temporal = temporal_payload(
                basis,
                previous_bases.get(key),
                num_blocks=block_count,
            )
            q10 = block_gaussian_statistics(
                source_white[test_indices],
                basis,
                num_blocks=block_count,
            )
            matching = temporal.get("matching")
            q12 = q12_payload(
                q10,
                previous_q10.get(key),
                matching if isinstance(matching, list) else None,
            )
            q18 = {
                name: perturbation_concentration(
                    source_white[test_indices],
                    target[test_indices],
                    basis,
                    num_blocks=block_count,
                )
                for name, target in targets_white.items()
            }
            block_records[key] = {
                "factorization": {
                    "train": fit.train_factorization,
                    "validation": fit.validation_factorization,
                    "random_validation": fit.random_validation_factorization,
                },
                "temporal": temporal,
                "q10": q10,
                "q11": cross_block_gaussian_mi(
                    source_white[test_indices],
                    basis,
                    num_blocks=block_count,
                ),
                "q12": q12,
                "q18": q18,
            }
            basis_state[key] = basis
            previous_bases[key] = basis
            previous_q10[key] = q10

        record = {
            "checkpoint": path.name,
            "epoch": epoch,
            "encoder": args.encoder,
            "alignment_to_reference": alignment,
            "common_pca_energy_fraction": common_subspace_energy_fraction(
                aligned_source[train_indices],
                reference_whitening.projection,
            ),
            "q17": q17,
            "q21": q21,
            "blocks": block_records,
            "seconds": time.time() - started,
        }
        records.append(record)
        torch.save(
            {
                "epoch": epoch,
                "bases": basis_state,
                "alignment_rotation": fit_orthogonal_procrustes(
                    source[train_indices],
                    reference_source[train_indices],
                    source[test_indices],
                    reference_source[test_indices],
                ).rotation,
            },
            state_dir / f"epoch_{epoch:04d}.pt",
        )
        atomic_json(args.output_dir / "records.json", records)
        print(
            f"checkpoint={path.name} epoch={epoch} "
            f"q17_mean={sum(row['q17_test_r2'] for row in q17.values()) / len(q17):.4f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        previous_source = source
        del source, targets, aligned_source, aligned_targets
        if device.type == "cuda":
            torch.cuda.empty_cache()

    atomic_json(
        args.output_dir / "config.json",
        {
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "checkpoint_pattern": args.checkpoint_pattern,
            "epochs": args.epochs,
            "reference_epoch": args.reference_epoch,
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
            "reference_pca_retained_variance_fraction": (
                reference_whitening.retained_variance_fraction
            ),
            "num_blocks": args.num_blocks,
            "transforms": transforms,
            "crop_scales": CROP_SCALES,
            "ridge": args.ridge,
            "jbd_restarts": args.jbd_restarts,
            "jbd_steps": args.jbd_steps,
            "jbd_learning_rate": args.jbd_learning_rate,
            "center_operators": True,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "amp_dtype": args.amp_dtype,
            "metric_version": "temporal-factorization-v1",
            "q23": "omitted_as_duplicate_of_q22_matched_set_distance",
        },
    )
    plot_records(records, args.output_dir / "temporal_factorization.png")
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
