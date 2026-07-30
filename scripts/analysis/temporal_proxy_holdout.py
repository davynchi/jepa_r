#!/usr/bin/env python3
"""Evaluate representation-quality proxies on unseen future checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from audit_ss_factorization_predictiveness import (  # noqa: E402
    load_factorization_display_epochs,
    load_probe_display_epochs,
)
from correlate_tiny_factorization_v2 import correlation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME,LDA_JSON,SS_JSON,PROBE_ROOT,METRICS_JSONL",
    )
    parser.add_argument("--train-through-epoch", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_train_loss(path: Path) -> dict[int, float]:
    result = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") != "epoch":
            continue
        scalars = record.get("scalars", {})
        value = scalars.get("train/epoch_loss")
        if value is not None and math.isfinite(float(value)):
            result[int(record["epoch"])] = float(value)
    return result


def parse_runs(values: list[str]) -> dict[str, dict[str, Any]]:
    runs = {}
    for value in values:
        fields = value.split(",", maxsplit=4)
        if len(fields) != 5:
            raise ValueError(
                "run must be NAME,LDA_JSON,SS_JSON,PROBE_ROOT,METRICS_JSONL"
            )
        name, lda_path, ss_path, probe_root, metrics_path = fields
        lda = load_factorization_display_epochs(Path(lda_path).resolve())
        ss = load_factorization_display_epochs(Path(ss_path).resolve())
        runs[name] = {
            "features": {
                "lda_trace": lda["lda_discriminative_trace"],
                "lda_between_total": lda["between_total_trace_ratio"],
                "ss_invariance": ss["ss_crop_invariance_mean"],
                "ss_latent_erank": ss["ss_crop_latent_effective_rank"],
                "train_loss": load_train_loss(Path(metrics_path).resolve()),
            },
            "probes": load_probe_display_epochs(Path(probe_root).resolve()),
        }
    return runs


def fit_ridge(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-12] = 1
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    design = np.column_stack((np.ones(len(train)), train))
    test_design = np.column_stack((np.ones(len(test)), test))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ train_targets,
    )
    return test_design @ coefficients, {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "alpha": alpha,
    }


def choose_alpha(features: np.ndarray, targets: np.ndarray) -> float:
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0)
    losses = []
    for alpha in candidates:
        errors = []
        for held_out in range(len(targets)):
            keep = np.arange(len(targets)) != held_out
            prediction, _ = fit_ridge(
                features[keep],
                targets[keep],
                features[held_out : held_out + 1],
                alpha=alpha,
            )
            errors.append(float((prediction[0] - targets[held_out]) ** 2))
        losses.append(float(np.mean(errors)))
    return candidates[int(np.argmin(losses))]


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    residual = target - prediction
    denominator = float(((target - target.mean()) ** 2).sum())
    pearson = correlation(target, prediction, spearman=False)
    spearman = correlation(target, prediction, spearman=True)
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1 - np.square(residual).sum() / denominator)
        if denominator > 0 else None,
        "pearson": pearson if math.isfinite(pearson) else None,
        "spearman": spearman if math.isfinite(spearman) else None,
    }


def feature_matrix(
    sources: dict[str, dict[int, float]],
    names: tuple[str, ...],
    epochs: list[int],
) -> np.ndarray:
    columns = []
    for name in names:
        if name == "epoch":
            columns.append(np.asarray(epochs, dtype=np.float64))
        else:
            columns.append(np.asarray([sources[name][epoch] for epoch in epochs]))
    return np.column_stack(columns)


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    predictors = {
        "epoch": ("epoch",),
        "train_loss": ("train_loss",),
        "lda_trace": ("lda_trace",),
        "lda_between_total": ("lda_between_total",),
        "ss_invariance": ("ss_invariance",),
        "ss_latent_erank": ("ss_latent_erank",),
        "composite": (
            "lda_trace",
            "lda_between_total",
            "ss_invariance",
            "ss_latent_erank",
        ),
    }
    results = []
    curves: dict[tuple[str, str], dict[str, tuple[list[int], list[float]]]] = {}
    for run_name, run in runs.items():
        sources = run["features"]
        for probe_name, probe_points in run["probes"].items():
            if not probe_name.startswith("target/") or not (
                probe_name.endswith("/top1") or probe_name.endswith("/top5")
            ):
                continue
            available = set(probe_points)
            for points in sources.values():
                available &= set(points)
            train_epochs = sorted(
                epoch for epoch in available if epoch <= args.train_through_epoch
            )
            test_epochs = sorted(
                epoch for epoch in available if epoch > args.train_through_epoch
            )
            if len(train_epochs) < 5 or len(test_epochs) < 2:
                continue
            train_target = np.asarray([probe_points[e] for e in train_epochs])
            test_target = np.asarray([probe_points[e] for e in test_epochs])
            curve = {"actual": (test_epochs, test_target.tolist())}
            # A no-change forecast is a hard-to-beat saturation baseline.
            last_prediction = np.full_like(test_target, train_target[-1])
            results.append(
                {
                    "run": run_name,
                    "probe": probe_name,
                    "predictor": "last_value",
                    "train_epochs": train_epochs,
                    "test_epochs": test_epochs,
                    **metrics(test_target, last_prediction),
                }
            )
            curve["last_value"] = (test_epochs, last_prediction.tolist())
            for predictor_name, feature_names in predictors.items():
                train_features = feature_matrix(sources, feature_names, train_epochs)
                test_features = feature_matrix(sources, feature_names, test_epochs)
                alpha = choose_alpha(train_features, train_target)
                prediction, model = fit_ridge(
                    train_features,
                    train_target,
                    test_features,
                    alpha=alpha,
                )
                results.append(
                    {
                        "run": run_name,
                        "probe": probe_name,
                        "predictor": predictor_name,
                        "feature_names": feature_names,
                        "train_epochs": train_epochs,
                        "test_epochs": test_epochs,
                        "model": model,
                        **metrics(test_target, prediction),
                    }
                )
                curve[predictor_name] = (test_epochs, prediction.tolist())
            curves[(run_name, probe_name)] = curve

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False)
    )
    selected_probes = ("target/ridge/top1", "target/linear/top1", "target/mlp/top1")
    figure, axes = plt.subplots(
        len(runs), len(selected_probes),
        figsize=(16, 5 * len(runs)),
        squeeze=False,
    )
    for row, run_name in enumerate(runs):
        for column, probe_name in enumerate(selected_probes):
            axis = axes[row, column]
            curve = curves.get((run_name, probe_name))
            if curve is None:
                axis.set_visible(False)
                continue
            epochs, actual = curve["actual"]
            axis.plot(epochs, np.asarray(actual) * 100, "ko-", label="actual")
            for predictor_name, style in (
                ("last_value", "--"),
                ("epoch", ":"),
                ("train_loss", "-."),
                ("ss_invariance", "-"),
                ("lda_trace", "-"),
                ("composite", "-"),
            ):
                values = curve[predictor_name][1]
                axis.plot(epochs, np.asarray(values) * 100, style, label=predictor_name)
            axis.set_title(f"{run_name}: {probe_name}")
            axis.set_xlabel("Held-out epoch")
            axis.set_ylabel("Accuracy, %")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(args.output_dir / "temporal_holdout.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output_dir)


if __name__ == "__main__":
    main()
