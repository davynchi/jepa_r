#!/usr/bin/env python3
"""Test whether one factorization metric transfers between two training runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-factorization", type=Path, required=True)
    parser.add_argument("--official-probes", type=Path, required=True)
    parser.add_argument("--ras-factorization", type=Path, required=True)
    parser.add_argument("--ras-probes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", default="ss_crop_invariance_mean")
    parser.add_argument("--probe", default="target/linear")
    parser.add_argument("--score", default="top1")
    parser.add_argument("--shared-max-epoch", type=int, default=250)
    return parser.parse_args()


def load_factorization(path: Path, metric: str) -> dict[int, float]:
    records = json.loads(path.read_text())
    return {
        int(record.get("display_epoch", record["epoch"])): float(
            record["metrics"][metric]["mean"]
        )
        for record in records
    }


def load_probes(root: Path, probe: str, score: str) -> dict[int, float]:
    values = {}
    for path in sorted(root.glob("epoch_*/results.json")):
        epoch = int(path.parent.name.removeprefix("epoch_"))
        payload = json.loads(path.read_text())
        values[epoch] = float(payload["probes"][probe][score])
    return values


def aligned(
    metric: dict[int, float],
    accuracy: dict[int, float],
    *,
    minimum_epoch: int | None = None,
    maximum_epoch: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs = sorted(set(metric) & set(accuracy))
    if minimum_epoch is not None:
        epochs = [epoch for epoch in epochs if epoch >= minimum_epoch]
    if maximum_epoch is not None:
        epochs = [epoch for epoch in epochs if epoch <= maximum_epoch]
    return (
        np.asarray(epochs, dtype=np.float64),
        np.asarray([metric[epoch] for epoch in epochs], dtype=np.float64),
        np.asarray([accuracy[epoch] for epoch in epochs], dtype=np.float64),
    )


def finite_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = spearmanr(left, right).statistic if rank else pearsonr(left, right).statistic
    return float(value) if math.isfinite(value) else None


def association(
    epochs: np.ndarray, metric: np.ndarray, accuracy: np.ndarray
) -> dict[str, object]:
    epoch_design = np.column_stack([np.ones(len(epochs)), epochs])
    metric_residual = metric - epoch_design @ np.linalg.lstsq(
        epoch_design, metric, rcond=None
    )[0]
    accuracy_residual = accuracy - epoch_design @ np.linalg.lstsq(
        epoch_design, accuracy, rcond=None
    )[0]
    return {
        "epochs": epochs.astype(int).tolist(),
        "n": len(epochs),
        "pearson": finite_correlation(metric, accuracy, rank=False),
        "spearman": finite_correlation(metric, accuracy, rank=True),
        "difference_pearson": finite_correlation(
            np.diff(metric), np.diff(accuracy), rank=False
        ),
        "difference_spearman": finite_correlation(
            np.diff(metric), np.diff(accuracy), rank=True
        ),
        "epoch_residual_pearson": finite_correlation(
            metric_residual, accuracy_residual, rank=False
        ),
        "epoch_residual_spearman": finite_correlation(
            metric_residual, accuracy_residual, rank=True
        ),
    }


def fit_affine(feature: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(feature)), feature])
    return np.linalg.lstsq(design, target, rcond=None)[0]


def predict_affine(coefficients: np.ndarray, feature: np.ndarray) -> np.ndarray:
    return coefficients[0] + coefficients[1] * feature


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    residual = actual - predicted
    denominator = float(np.square(actual - actual.mean()).sum())
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": (
            1.0 - float(np.square(residual).sum()) / denominator
            if denominator > 0
            else None
        ),
        "pearson": finite_correlation(predicted, actual, rank=False),
        "spearman": finite_correlation(predicted, actual, rank=True),
    }


def transfer(
    *,
    name: str,
    train_epochs: np.ndarray,
    train_metric: np.ndarray,
    train_accuracy: np.ndarray,
    test_epochs: np.ndarray,
    test_metric: np.ndarray,
    test_accuracy: np.ndarray,
) -> dict[str, object]:
    metric_model = fit_affine(train_metric, train_accuracy)
    epoch_model = fit_affine(train_epochs, train_accuracy)
    metric_prediction = predict_affine(metric_model, test_metric)
    epoch_prediction = predict_affine(epoch_model, test_epochs)
    last_value_prediction = np.full_like(test_accuracy, train_accuracy[-1])
    return {
        "name": name,
        "train_epochs": train_epochs.astype(int).tolist(),
        "test_epochs": test_epochs.astype(int).tolist(),
        "metric_model": {
            "intercept": float(metric_model[0]),
            "slope": float(metric_model[1]),
        },
        "epoch_model": {
            "intercept": float(epoch_model[0]),
            "slope": float(epoch_model[1]),
        },
        "factorization": prediction_metrics(test_accuracy, metric_prediction),
        "epoch": prediction_metrics(test_accuracy, epoch_prediction),
        "last_value": prediction_metrics(test_accuracy, last_value_prediction),
        "actual": test_accuracy.tolist(),
        "factorization_prediction": metric_prediction.tolist(),
        "epoch_prediction": epoch_prediction.tolist(),
    }


def ranking_concordance(
    official_metric: dict[int, float],
    official_accuracy: dict[int, float],
    ras_metric: dict[int, float],
    ras_accuracy: dict[int, float],
    *,
    maximum_epoch: int,
) -> dict[str, object]:
    epochs = sorted(
        set(official_metric)
        & set(official_accuracy)
        & set(ras_metric)
        & set(ras_accuracy)
    )
    epochs = [epoch for epoch in epochs if epoch <= maximum_epoch]
    matches = []
    for epoch in epochs:
        metric_difference = ras_metric[epoch] - official_metric[epoch]
        accuracy_difference = ras_accuracy[epoch] - official_accuracy[epoch]
        matches.append(metric_difference * accuracy_difference > 0)
    return {
        "epochs": epochs,
        "correct": int(sum(matches)),
        "total": len(matches),
        "fraction": float(np.mean(matches)) if matches else None,
    }


def plot(
    output: Path,
    official: tuple[np.ndarray, np.ndarray, np.ndarray],
    ras: tuple[np.ndarray, np.ndarray, np.ndarray],
    transfers: list[dict[str, object]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, values, color in (
        ("official I-JEPA", official, "#2878b5"),
        ("RAS", ras, "#d95f35"),
    ):
        epochs, metric, accuracy = values
        axes[0].plot(metric, 100 * accuracy, "o-", label=label, color=color)
        for epoch, x, y in zip(epochs.astype(int), metric, 100 * accuracy, strict=True):
            axes[0].annotate(str(epoch), (x, y), fontsize=7, xytext=(3, 3),
                             textcoords="offset points")
    axes[0].set_xlabel("SS crop invariance mean")
    axes[0].set_ylabel("Target linear Top-1, %")
    axes[0].set_title("Metric and representation quality")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    labels = [row["name"] for row in transfers]
    metric_mae = [100 * row["factorization"]["mae"] for row in transfers]  # type: ignore[index]
    epoch_mae = [100 * row["epoch"]["mae"] for row in transfers]  # type: ignore[index]
    last_mae = [100 * row["last_value"]["mae"] for row in transfers]  # type: ignore[index]
    positions = np.arange(len(labels))
    axes[1].bar(positions - 0.27, metric_mae, width=0.27, label="SS")
    axes[1].bar(positions, epoch_mae, width=0.27, label="Epoch")
    axes[1].bar(positions + 0.27, last_mae, width=0.27, label="Last accuracy")
    axes[1].set_xticks(positions, labels, rotation=15, ha="right")
    axes[1].set_ylabel("Top-1 MAE, percentage points")
    axes[1].set_title("Prediction on held-out trajectory")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    official_metric = load_factorization(args.official_factorization, args.metric)
    official_accuracy = load_probes(args.official_probes, args.probe, args.score)
    ras_metric = load_factorization(args.ras_factorization, args.metric)
    ras_accuracy = load_probes(args.ras_probes, args.probe, args.score)

    official = aligned(
        official_metric,
        official_accuracy,
        maximum_epoch=args.shared_max_epoch,
    )
    ras_shared = aligned(
        ras_metric,
        ras_accuracy,
        maximum_epoch=args.shared_max_epoch,
    )
    ras_late = aligned(
        ras_metric,
        ras_accuracy,
        minimum_epoch=args.shared_max_epoch + 1,
    )
    transfers = [
        transfer(
            name="official_to_ras",
            train_epochs=official[0],
            train_metric=official[1],
            train_accuracy=official[2],
            test_epochs=ras_shared[0],
            test_metric=ras_shared[1],
            test_accuracy=ras_shared[2],
        ),
        transfer(
            name="ras_to_official",
            train_epochs=ras_shared[0],
            train_metric=ras_shared[1],
            train_accuracy=ras_shared[2],
            test_epochs=official[0],
            test_metric=official[1],
            test_accuracy=official[2],
        ),
        transfer(
            name="ras_early_to_late",
            train_epochs=ras_shared[0],
            train_metric=ras_shared[1],
            train_accuracy=ras_shared[2],
            test_epochs=ras_late[0],
            test_metric=ras_late[1],
            test_accuracy=ras_late[2],
        ),
    ]
    result = {
        "config": {
            "metric": args.metric,
            "probe": args.probe,
            "score": args.score,
            "shared_max_epoch": args.shared_max_epoch,
        },
        "associations": {
            "official": association(*official),
            "ras": association(*aligned(ras_metric, ras_accuracy)),
        },
        "transfers": transfers,
        "cross_run_ranking": ranking_concordance(
            official_metric,
            official_accuracy,
            ras_metric,
            ras_accuracy,
            maximum_epoch=args.shared_max_epoch,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    plot(
        args.output_dir / "summary.png",
        official,
        aligned(ras_metric, ras_accuracy),
        transfers,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
