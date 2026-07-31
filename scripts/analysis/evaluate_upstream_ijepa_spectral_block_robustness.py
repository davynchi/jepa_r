#!/usr/bin/env python3
"""Audit spectral-null block structure at fixed whitening dimensions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
from evaluate_upstream_ijepa_joint_transform_factorization import split_indices  # noqa: E402
from evaluate_upstream_ijepa_spectral_block_null import (  # noqa: E402
    atomic_json,
    fit_operator_family,
    percentile,
    stack_centered_operators,
)

from jepa.analysis.joint_transform_factorization import (  # noqa: E402
    apply_whitening,
    fit_whitening_projection,
)
from jepa.analysis.spectral_block_null import (  # noqa: E402
    haar_orthogonal,
    optimal_projector_agreement,
    optimized_factorization,
    spectral_null_test,
)
from jepa.configs.base import derive_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=(50, 100, 150, 200, 250, 300))
    parser.add_argument("--dimensions", type=int, nargs="+", default=(32, 56, 96))
    parser.add_argument("--num-blocks", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument("--transforms", nargs="+", default=("flip", "color_fixed", "blur"))
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--operator-fit-size", type=int, default=6000)
    parser.add_argument("--operator-test-size", type=int, default=4000)
    parser.add_argument("--minimum-test-r2", type=float, default=0.5)
    parser.add_argument("--whitening-relative-floor", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--null-samples", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-null-samples", type=int, default=20)
    parser.add_argument("--null-batch-size", type=int, default=100)
    parser.add_argument("--projector-null-samples", type=int, default=100)
    parser.add_argument("--jbd-restarts", type=int, default=2)
    parser.add_argument("--jbd-steps", type=int, default=300)
    parser.add_argument("--jbd-learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def load_embeddings(
    cache_dir: Path,
    epoch: int,
    transforms: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    path = cache_dir / f"epoch_{epoch:04d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(payload["targets"]) != transforms:
        raise ValueError(f"cached transforms do not match requested transforms: {path}")
    return payload["source"], payload["targets"]


def fixed_whiten(
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    fit_indices: torch.Tensor,
    dimension: int,
    relative_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    whitening = fit_whitening_projection(
        source[fit_indices],
        dimension=dimension,
        relative_floor=relative_floor,
    )
    return (
        apply_whitening(source, whitening),
        {name: apply_whitening(target, whitening) for name, target in targets.items()},
        whitening.retained_variance_fraction,
    )


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "median": float(tensor.median()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=True)),
        "percentile_low": percentile(values, 0.025),
        "percentile_high": percentile(values, 0.975),
    }


def projector_agreement_with_null(
    first_basis: torch.Tensor,
    second_basis: torch.Tensor,
    *,
    num_blocks: int,
    null_samples: int,
    seed: int,
) -> dict[str, object]:
    observed = optimal_projector_agreement(
        first_basis,
        second_basis,
        num_blocks=num_blocks,
    )
    generator = torch.Generator().manual_seed(seed)
    null_values = []
    for _ in range(null_samples):
        rotation = haar_orthogonal(
            second_basis.shape[0],
            generator=generator,
            device=second_basis.device,
            dtype=second_basis.dtype,
        )
        null_values.append(
            optimal_projector_agreement(
                first_basis,
                rotation @ second_basis,
                num_blocks=num_blocks,
            ).mean_overlap
        )
    null = summarize(null_values)
    null_std = null["std"]
    excess = (observed.mean_overlap - null["mean"]) / max(1.0 - null["mean"], 1e-12)
    z_score = (observed.mean_overlap - null["mean"]) / null_std if null_std > 0 else float("nan")
    return {
        "mean": observed.mean_overlap,
        "minimum": observed.minimum_overlap,
        "matched": list(observed.matched_overlaps),
        "assignment": list(observed.assignment),
        "null_mean": null["mean"],
        "null_std": null_std,
        "excess_over_null": excess,
        "z_score": z_score,
        "null_values": null_values,
    }


def evaluate_fixed_dimension(
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    fit_indices: torch.Tensor,
    test_indices: torch.Tensor,
    dimension: int,
    transforms: tuple[str, ...],
    relative_floor: float,
    ridge: float,
    num_blocks: tuple[int, ...],
    null_samples: int,
    bootstrap_samples: int,
    bootstrap_null_samples: int,
    null_batch_size: int,
    projector_null_samples: int,
    jbd_restarts: int,
    jbd_steps: int,
    jbd_learning_rate: float,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    source_white, targets_white, retained = fixed_whiten(
        source,
        targets,
        fit_indices=fit_indices,
        dimension=dimension,
        relative_floor=relative_floor,
    )
    primary_operators, diagnostics = fit_operator_family(
        source_white,
        targets_white,
        fit_indices=fit_indices,
        test_indices=test_indices,
        ridge=ridge,
    )
    primary = stack_centered_operators(primary_operators, transforms, device=device)

    split_point = fit_indices.numel() // 2
    first_indices = fit_indices[:split_point]
    second_indices = fit_indices[split_point:]
    first_operators, first_diagnostics = fit_operator_family(
        source_white,
        {name: targets_white[name] for name in transforms},
        fit_indices=first_indices,
        test_indices=test_indices,
        ridge=ridge,
    )
    second_operators, second_diagnostics = fit_operator_family(
        source_white,
        {name: targets_white[name] for name in transforms},
        fit_indices=second_indices,
        test_indices=test_indices,
        ridge=ridge,
    )
    first_stack = stack_centered_operators(first_operators, transforms, device=device)
    second_stack = stack_centered_operators(second_operators, transforms, device=device)

    bootstrap_stacks = []
    for bootstrap_index in range(bootstrap_samples):
        generator = torch.Generator().manual_seed(
            derive_seed(seed, "bootstrap-data", bootstrap_index)
        )
        positions = torch.randint(
            fit_indices.numel(),
            (fit_indices.numel(),),
            generator=generator,
        )
        sampled_indices = fit_indices[positions]
        operators, _ = fit_operator_family(
            source_white,
            {name: targets_white[name] for name in transforms},
            fit_indices=sampled_indices,
            test_indices=test_indices,
            ridge=ridge,
        )
        bootstrap_stacks.append(stack_centered_operators(operators, transforms, device=device))

    block_results: dict[str, dict[str, object]] = {}
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
        repeat_score, repeat_fit = optimized_factorization(
            primary,
            num_blocks=block_count,
            restarts=jbd_restarts,
            steps=jbd_steps,
            learning_rate=jbd_learning_rate,
            seed=derive_seed(seed, "primary-repeat", block_count),
        )
        optimizer_agreement = projector_agreement_with_null(
            primary_null.real_basis,
            repeat_fit.basis,
            num_blocks=block_count,
            null_samples=projector_null_samples,
            seed=derive_seed(seed, "optimizer-agreement-null", block_count),
        )
        _, first_fit = optimized_factorization(
            first_stack,
            num_blocks=block_count,
            restarts=jbd_restarts,
            steps=jbd_steps,
            learning_rate=jbd_learning_rate,
            seed=derive_seed(seed, "split-first", block_count),
        )
        _, second_fit = optimized_factorization(
            second_stack,
            num_blocks=block_count,
            restarts=jbd_restarts,
            steps=jbd_steps,
            learning_rate=jbd_learning_rate,
            seed=derive_seed(seed, "split-second", block_count),
        )
        split_agreement = projector_agreement_with_null(
            first_fit.basis,
            second_fit.basis,
            num_blocks=block_count,
            null_samples=projector_null_samples,
            seed=derive_seed(seed, "split-agreement-null", block_count),
        )

        bootstrap_deltas = []
        bootstrap_z_scores = []
        for bootstrap_index, operators in enumerate(bootstrap_stacks):
            result = spectral_null_test(
                operators,
                num_blocks=block_count,
                null_samples=bootstrap_null_samples,
                restarts=jbd_restarts,
                steps=jbd_steps,
                learning_rate=jbd_learning_rate,
                seed=derive_seed(seed, "bootstrap-null", block_count, bootstrap_index),
                null_batch_size=null_batch_size,
            )
            bootstrap_deltas.append(result.delta)
            bootstrap_z_scores.append(result.z_score)

        block_results[str(block_count)] = {
            "primary": {
                "real_factorization": primary_null.real_factorization,
                "null_mean": primary_null.null_mean,
                "null_std": primary_null.null_std,
                "delta": primary_null.delta,
                "z_score": primary_null.z_score,
                "null_factorizations": list(primary_null.null_factorizations),
            },
            "bootstrap_delta": summarize(bootstrap_deltas),
            "bootstrap_z": summarize(bootstrap_z_scores),
            "bootstrap_deltas": bootstrap_deltas,
            "bootstrap_z_scores": bootstrap_z_scores,
            "optimizer_repeat_score": repeat_score,
            "optimizer_projector_agreement": optimizer_agreement,
            "split_projector_agreement": split_agreement,
        }

    return {
        "dimension": dimension,
        "retained_variance_fraction": retained,
        "operator_diagnostics": diagnostics,
        "split_operator_diagnostics": {
            "first": first_diagnostics,
            "second": second_diagnostics,
        },
        "blocks": block_results,
    }


def plot_records(records: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    dimensions = sorted({int(row["dimension"]) for row in records})
    figure, axes = plt.subplots(2, len(dimensions), figsize=(5 * len(dimensions), 8), squeeze=False)
    for column, dimension in enumerate(dimensions):
        rows = sorted(
            (row for row in records if int(row["dimension"]) == dimension),
            key=lambda row: int(row["epoch"]),
        )
        epochs = [int(row["epoch"]) for row in rows]
        block_counts = sorted(int(key) for key in rows[0]["result"]["blocks"])  # type: ignore[index]
        for block_count in block_counts:
            blocks = [row["result"]["blocks"][str(block_count)] for row in rows]  # type: ignore[index]
            medians = [float(block["bootstrap_delta"]["median"]) for block in blocks]
            lower = [float(block["bootstrap_delta"]["percentile_low"]) for block in blocks]
            upper = [float(block["bootstrap_delta"]["percentile_high"]) for block in blocks]
            axes[0, column].plot(epochs, medians, marker="o", label=f"K={block_count}")
            axes[0, column].fill_between(epochs, lower, upper, alpha=0.15)
        axes[0, column].axhline(0, color="black", linewidth=0.8)
        axes[0, column].set_title(f"Fixed d={dimension}")
        axes[0, column].set_xlabel("Epoch")
        axes[0, column].set_ylabel(r"bootstrap median $\Delta F_K$")
        axes[0, column].grid(alpha=0.25)
        axes[0, column].legend()

        transforms = list(rows[0]["result"]["operator_diagnostics"])  # type: ignore[index]
        for name in transforms:
            values = [
                float(row["result"]["operator_diagnostics"][name]["test_r2"])  # type: ignore[index]
                for row in rows
            ]
            axes[1, column].plot(epochs, values, marker="o", label=name)
        axes[1, column].set_xlabel("Epoch")
        axes[1, column].set_ylabel(r"test-$R^2$")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    epochs = tuple(args.epochs)
    dimensions = tuple(args.dimensions)
    num_blocks = tuple(args.num_blocks)
    transforms = tuple(args.transforms)
    if len(set(epochs)) != len(epochs) or len(set(dimensions)) != len(dimensions):
        raise ValueError("epochs and dimensions must be unique")
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("dimensions must be positive")
    if any(dimension % block_count for dimension in dimensions for block_count in num_blocks):
        raise ValueError("every dimension must be divisible by every block count")
    if args.operator_fit_size + args.operator_test_size != args.test_size:
        raise ValueError("operator fit and test sizes must sum to test-size")
    if args.null_samples < 100 or args.bootstrap_samples < 100 or args.projector_null_samples < 100:
        raise ValueError(
            "robustness audit requires at least 100 null, bootstrap, and projector-null samples"
        )
    if args.whitening_relative_floor <= 0:
        raise ValueError("whitening-relative-floor must be positive")

    fit_indices, _, test_indices = split_indices(
        args.test_size,
        train_size=args.operator_fit_size,
        validation_size=0,
        seed=derive_seed(args.seed, "operator-split"),
    )
    preflight: dict[str, dict[str, object]] = {}
    for dimension in dimensions:
        for epoch in epochs:
            source, targets = load_embeddings(args.embedding_cache_dir, epoch, transforms)
            source_white, targets_white, retained = fixed_whiten(
                source,
                targets,
                fit_indices=fit_indices,
                dimension=dimension,
                relative_floor=args.whitening_relative_floor,
            )
            _, diagnostics = fit_operator_family(
                source_white,
                targets_white,
                fit_indices=fit_indices,
                test_indices=test_indices,
                ridge=args.ridge,
            )
            preflight[f"d{dimension}_e{epoch}"] = {
                "dimension": dimension,
                "epoch": epoch,
                "retained_variance_fraction": retained,
                "operator_diagnostics": diagnostics,
            }
    failed = [
        (key, name, values["operator_diagnostics"][name]["test_r2"])  # type: ignore[index]
        for key, values in preflight.items()
        for name in transforms
        if float(values["operator_diagnostics"][name]["test_r2"]) < args.minimum_test_r2  # type: ignore[index]
    ]
    atomic_json(args.output_dir / "operator_preflight.json", {"failed": failed, "runs": preflight})
    if failed:
        raise RuntimeError(f"operators failed the common R2 threshold: {failed}")
    if args.preflight_only:
        print(args.output_dir, flush=True)
        return

    records_path = args.output_dir / "records.json"
    records: list[dict[str, object]] = []
    if records_path.exists():
        records = json.loads(records_path.read_text())
    completed = {(int(row["dimension"]), int(row["epoch"])) for row in records}
    device = torch.device(args.device)
    for dimension in dimensions:
        for epoch in epochs:
            if (dimension, epoch) in completed:
                print(f"d={dimension} epoch={epoch} already complete", flush=True)
                continue
            source, targets = load_embeddings(args.embedding_cache_dir, epoch, transforms)
            started = time.time()
            result = evaluate_fixed_dimension(
                source,
                targets,
                fit_indices=fit_indices,
                test_indices=test_indices,
                dimension=dimension,
                transforms=transforms,
                relative_floor=args.whitening_relative_floor,
                ridge=args.ridge,
                num_blocks=num_blocks,
                null_samples=args.null_samples,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_null_samples=args.bootstrap_null_samples,
                null_batch_size=args.null_batch_size,
                projector_null_samples=args.projector_null_samples,
                jbd_restarts=args.jbd_restarts,
                jbd_steps=args.jbd_steps,
                jbd_learning_rate=args.jbd_learning_rate,
                device=device,
                seed=derive_seed(args.seed, "fixed-d", dimension, epoch),
            )
            record = {
                "dimension": dimension,
                "epoch": epoch,
                "seconds": time.time() - started,
                "result": result,
            }
            records.append(record)
            records.sort(key=lambda row: (int(row["dimension"]), int(row["epoch"])))
            atomic_json(records_path, records)
            summary = " ".join(
                f"median_dF_{block_count}="
                f"{result['blocks'][str(block_count)]['bootstrap_delta']['median']:.4f}"  # type: ignore[index]
                for block_count in num_blocks
            )
            print(
                f"d={dimension} epoch={epoch} {summary} seconds={record['seconds']:.1f}",
                flush=True,
            )

    config = vars(args).copy()
    for key in ("embedding_cache_dir", "output_dir"):
        config[key] = str(config[key])
    config["epochs"] = list(epochs)
    config["dimensions"] = list(dimensions)
    config["num_blocks"] = list(num_blocks)
    config["transforms"] = list(transforms)
    atomic_json(args.output_dir / "config.json", config)
    plot_records(records, args.output_dir / "fixed_dimension_robustness.png")
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
