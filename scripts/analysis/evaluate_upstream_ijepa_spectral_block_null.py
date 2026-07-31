#!/usr/bin/env python3
"""Measure excess joint block structure above an operator spectral null."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
from evaluate_upstream_ijepa_joint_transform_factorization import (  # noqa: E402
    build_encoder,
    checkpoint_number,
    encode_transform_pairs,
    split_indices,
)

from jepa.analysis.joint_transform_factorization import (  # noqa: E402
    apply_whitening,
    fit_linear_operator,
    remove_isotropic_component,
)
from jepa.analysis.spectral_block_null import (  # noqa: E402
    fit_variance_whitening_projection,
    spectral_null_test,
)
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    TinyImageNetStaticImageDataset,
)

DEFAULT_TRANSFORMS = ("flip", "color_fixed", "blur")


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
    parser.add_argument("--operator-fit-size", type=int, default=6000)
    parser.add_argument("--operator-test-size", type=int, default=4000)
    parser.add_argument("--minimum-variance-fraction", type=float, default=0.95)
    parser.add_argument("--num-blocks", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument("--transforms", nargs="+", default=DEFAULT_TRANSFORMS)
    parser.add_argument("--minimum-test-r2", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--null-samples", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10)
    parser.add_argument("--bootstrap-null-samples", type=int, default=20)
    parser.add_argument("--null-batch-size", type=int, default=10)
    parser.add_argument("--jbd-restarts", type=int, default=2)
    parser.add_argument("--jbd-steps", type=int, default=300)
    parser.add_argument("--jbd-learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--no-untrained", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, probability))


def fit_operator_family(
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    fit_indices: torch.Tensor,
    test_indices: torch.Tensor,
    ridge: float,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float]]]:
    operators = {}
    diagnostics = {}
    for name, target in targets.items():
        fit = fit_linear_operator(
            source[fit_indices],
            target[fit_indices],
            source[test_indices],
            target[test_indices],
            ridge=ridge,
        )
        operators[name] = fit.operator
        diagnostics[name] = {
            "test_r2": fit.r_squared,
            "test_nmse": fit.normalized_mse,
            "energy_per_dimension": fit.energy_per_dimension,
        }
    return operators, diagnostics


def stack_centered_operators(
    operators: dict[str, torch.Tensor],
    names: tuple[str, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    stacked = torch.stack([operators[name] for name in names]).to(
        device=device,
        dtype=torch.float32,
    )
    return remove_isotropic_component(stacked)


def evaluate_embeddings(
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    fit_indices: torch.Tensor,
    test_indices: torch.Tensor,
    minimum_variance_fraction: float,
    num_blocks: tuple[int, ...],
    accepted_transforms: tuple[str, ...],
    ridge: float,
    null_samples: int,
    bootstrap_samples: int,
    bootstrap_null_samples: int,
    jbd_restarts: int,
    jbd_steps: int,
    jbd_learning_rate: float,
    null_batch_size: int,
    optimization_device: torch.device,
    seed: int,
) -> dict[str, object]:
    dimension_multiple = math.lcm(*num_blocks)
    whitening = fit_variance_whitening_projection(
        source[fit_indices],
        minimum_variance_fraction=minimum_variance_fraction,
        dimension_multiple=dimension_multiple,
    )
    source_white = apply_whitening(source, whitening)
    targets_white = {name: apply_whitening(target, whitening) for name, target in targets.items()}
    primary_operators, diagnostics = fit_operator_family(
        source_white,
        targets_white,
        fit_indices=fit_indices,
        test_indices=test_indices,
        ridge=ridge,
    )
    accepted = accepted_transforms
    rejected = tuple(name for name in targets if name not in accepted)
    primary = stack_centered_operators(
        primary_operators,
        accepted,
        device=optimization_device,
    )

    blocks: dict[str, dict[str, object]] = {}
    for block_count in num_blocks:
        primary_null = spectral_null_test(
            primary,
            num_blocks=block_count,
            null_samples=null_samples,
            restarts=jbd_restarts,
            steps=jbd_steps,
            learning_rate=jbd_learning_rate,
            seed=derive_seed(seed, "primary", block_count),
            null_batch_size=null_batch_size,
        )
        bootstrap_deltas = []
        bootstrap_z_scores = []
        for bootstrap_index in range(bootstrap_samples):
            generator = torch.Generator().manual_seed(
                derive_seed(seed, "bootstrap-data", bootstrap_index)
            )
            positions = torch.randint(
                fit_indices.numel(),
                (fit_indices.numel(),),
                generator=generator,
            )
            bootstrap_indices = fit_indices[positions]
            bootstrap_operators, _ = fit_operator_family(
                source_white,
                {name: targets_white[name] for name in accepted},
                fit_indices=bootstrap_indices,
                test_indices=test_indices,
                ridge=ridge,
            )
            bootstrap_stack = stack_centered_operators(
                bootstrap_operators,
                accepted,
                device=optimization_device,
            )
            bootstrap_null = spectral_null_test(
                bootstrap_stack,
                num_blocks=block_count,
                null_samples=bootstrap_null_samples,
                restarts=jbd_restarts,
                steps=jbd_steps,
                learning_rate=jbd_learning_rate,
                seed=derive_seed(seed, "bootstrap-null", block_count, bootstrap_index),
                null_batch_size=null_batch_size,
            )
            bootstrap_deltas.append(bootstrap_null.delta)
            bootstrap_z_scores.append(bootstrap_null.z_score)
        blocks[str(block_count)] = {
            "real_factorization": primary_null.real_factorization,
            "null_mean": primary_null.null_mean,
            "null_std": primary_null.null_std,
            "null_q025": percentile(list(primary_null.null_factorizations), 0.025),
            "null_q975": percentile(list(primary_null.null_factorizations), 0.975),
            "delta": primary_null.delta,
            "z_score": primary_null.z_score,
            "null_factorizations": list(primary_null.null_factorizations),
            "bootstrap_deltas": bootstrap_deltas,
            "bootstrap_z_scores": bootstrap_z_scores,
            "delta_ci_low": percentile(bootstrap_deltas, 0.025),
            "delta_ci_high": percentile(bootstrap_deltas, 0.975),
        }

    return {
        "whitening_dimension": whitening.projection.shape[1],
        "retained_variance_fraction": whitening.retained_variance_fraction,
        "accepted_transforms": list(accepted),
        "rejected_transforms": list(rejected),
        "operator_diagnostics": diagnostics,
        "blocks": blocks,
    }


def preflight_embeddings(
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    fit_indices: torch.Tensor,
    test_indices: torch.Tensor,
    minimum_variance_fraction: float,
    dimension_multiple: int,
    ridge: float,
) -> dict[str, object]:
    whitening = fit_variance_whitening_projection(
        source[fit_indices],
        minimum_variance_fraction=minimum_variance_fraction,
        dimension_multiple=dimension_multiple,
    )
    source_white = apply_whitening(source, whitening)
    targets_white = {name: apply_whitening(target, whitening) for name, target in targets.items()}
    _, diagnostics = fit_operator_family(
        source_white,
        targets_white,
        fit_indices=fit_indices,
        test_indices=test_indices,
        ridge=ridge,
    )
    return {
        "whitening_dimension": whitening.projection.shape[1],
        "retained_variance_fraction": whitening.retained_variance_fraction,
        "operator_diagnostics": diagnostics,
    }


def plot_records(
    records: list[dict[str, object]],
    output: Path,
    *,
    minimum_test_r2: float,
) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(records, key=lambda row: int(row["epoch"]))
    epochs = [int(row["epoch"]) for row in ordered]
    block_counts = sorted(int(key) for key in ordered[0]["result"]["blocks"])  # type: ignore[index]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for block_count in block_counts:
        rows = [row["result"]["blocks"][str(block_count)] for row in ordered]  # type: ignore[index]
        values = [float(row["delta"]) for row in rows]
        lower = [float(row["delta_ci_low"]) for row in rows]
        upper = [float(row["delta_ci_high"]) for row in rows]
        axes[0].plot(epochs, values, marker="o", label=f"K={block_count}")
        axes[0].fill_between(epochs, lower, upper, alpha=0.15)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Excess joint block structure")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"$\Delta F_K$ (bootstrap 95% CI)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    transforms = list(ordered[0]["result"]["operator_diagnostics"])  # type: ignore[index]
    for name in transforms:
        values = [
            float(row["result"]["operator_diagnostics"][name]["test_r2"])  # type: ignore[index]
            for row in ordered
        ]
        axes[1].plot(epochs, values, marker="o", label=name)
    axes[1].axhline(minimum_test_r2, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Held-out operator fit")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"test-$R^2$")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    transforms = tuple(args.transforms)
    num_blocks = tuple(args.num_blocks)
    if len(set(transforms)) != len(transforms):
        raise ValueError("transforms must be unique")
    if len(set(num_blocks)) != len(num_blocks) or any(value <= 1 for value in num_blocks):
        raise ValueError("num-blocks must contain unique integers greater than one")
    if args.operator_fit_size + args.operator_test_size != args.test_size:
        raise ValueError("operator fit and test sizes must sum to test-size")
    if args.null_samples < 100:
        raise ValueError("null-samples must be at least 100")
    if args.bootstrap_samples < 2 or args.bootstrap_null_samples < 2:
        raise ValueError("bootstrap settings require at least two samples")

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
    fit_indices, _, test_indices = split_indices(
        len(dataset),
        train_size=args.operator_fit_size,
        validation_size=0,
        seed=derive_seed(args.seed, "operator-split"),
    )

    checkpoints = sorted(
        args.checkpoint_dir.glob(args.checkpoint_pattern),
        key=checkpoint_number,
    )
    if args.epochs:
        wanted = set(args.epochs)
        checkpoints = [path for path in checkpoints if checkpoint_number(path) in wanted]
    if not checkpoints:
        raise FileNotFoundError("no matching official I-JEPA checkpoints")
    work: list[tuple[str, int, Path | None]] = []
    if not args.no_untrained:
        work.append(("untrained", 0, None))
    work.extend((path.name, checkpoint_number(path), path) for path in checkpoints)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths: dict[int, Path] = {}
    for checkpoint_name, epoch, checkpoint_path in work:
        cache_path = cache_dir / f"epoch_{epoch:04d}.pt"
        cache_paths[epoch] = cache_path
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if tuple(cached["targets"]) != transforms:
                raise ValueError(
                    f"cached transforms do not match requested transforms: {cache_path}"
                )
            print(f"epoch={epoch} embeddings cached", flush=True)
            continue
        else:
            checkpoint = (
                None
                if checkpoint_path is None
                else torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            )
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
            torch.save({"source": source, "targets": targets}, cache_path)
            del core, encoder, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"epoch={epoch} embeddings encoded", flush=True)

    preflight: dict[str, dict[str, object]] = {}
    dimension_multiple = math.lcm(*num_blocks)
    for _, epoch, _ in work:
        cached = torch.load(cache_paths[epoch], map_location="cpu", weights_only=False)
        preflight[str(epoch)] = preflight_embeddings(
            cached["source"],
            cached["targets"],
            fit_indices=fit_indices,
            test_indices=test_indices,
            minimum_variance_fraction=args.minimum_variance_fraction,
            dimension_multiple=dimension_multiple,
            ridge=args.ridge,
        )
    accepted_transforms = tuple(
        name
        for name in transforms
        if all(
            float(preflight[str(epoch)]["operator_diagnostics"][name]["test_r2"])  # type: ignore[index]
            >= args.minimum_test_r2
            for _, epoch, _ in work
        )
    )
    if len(accepted_transforms) < 2:
        raise RuntimeError(
            "fewer than two transformations pass the R2 threshold at every checkpoint: "
            f"accepted={accepted_transforms}"
        )
    atomic_json(
        args.output_dir / "operator_preflight.json",
        {
            "minimum_test_r2": args.minimum_test_r2,
            "accepted_transforms": list(accepted_transforms),
            "rejected_transforms": [name for name in transforms if name not in accepted_transforms],
            "epochs": preflight,
        },
    )
    print(f"globally accepted transforms={accepted_transforms}", flush=True)

    if args.preflight_only:
        config = vars(args).copy()
        for key in ("checkpoint_dir", "tiny_imagenet_root", "output_dir"):
            config[key] = str(config[key])
        config["transforms"] = list(transforms)
        config["num_blocks"] = list(num_blocks)
        atomic_json(args.output_dir / "config.json", config)
        print(args.output_dir, flush=True)
        return

    records_path = args.output_dir / "records.json"
    records: list[dict[str, object]] = []
    if records_path.exists():
        records = json.loads(records_path.read_text())
    completed = {int(row["epoch"]) for row in records}
    for checkpoint_name, epoch, _ in work:
        if epoch in completed:
            print(f"epoch={epoch} already complete", flush=True)
            continue
        started = time.time()
        cached = torch.load(cache_paths[epoch], map_location="cpu", weights_only=False)
        source = cached["source"]
        targets = cached["targets"]

        result = evaluate_embeddings(
            source,
            targets,
            fit_indices=fit_indices,
            test_indices=test_indices,
            minimum_variance_fraction=args.minimum_variance_fraction,
            num_blocks=num_blocks,
            accepted_transforms=accepted_transforms,
            ridge=args.ridge,
            null_samples=args.null_samples,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_null_samples=args.bootstrap_null_samples,
            jbd_restarts=args.jbd_restarts,
            jbd_steps=args.jbd_steps,
            jbd_learning_rate=args.jbd_learning_rate,
            null_batch_size=args.null_batch_size,
            optimization_device=device,
            seed=derive_seed(args.seed, "spectral-null", checkpoint_name),
        )
        record = {
            "checkpoint": checkpoint_name,
            "epoch": epoch,
            "seconds": time.time() - started,
            "result": result,
        }
        records.append(record)
        records.sort(key=lambda row: int(row["epoch"]))
        atomic_json(records_path, records)
        summary = " ".join(
            f"dF_{block_count}={result['blocks'][str(block_count)]['delta']:.4f}"  # type: ignore[index]
            for block_count in num_blocks
        )
        print(
            f"epoch={epoch} dim={result['whitening_dimension']} "
            f"retained={result['retained_variance_fraction']:.4f} {summary} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )

    config = vars(args).copy()
    for key in ("checkpoint_dir", "tiny_imagenet_root", "output_dir"):
        config[key] = str(config[key])
    config["transforms"] = list(transforms)
    config["num_blocks"] = list(num_blocks)
    atomic_json(args.output_dir / "config.json", config)
    plot_records(
        records,
        args.output_dir / "spectral_block_null.png",
        minimum_test_r2=args.minimum_test_r2,
    )
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
