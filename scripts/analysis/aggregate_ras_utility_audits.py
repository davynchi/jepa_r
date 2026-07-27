#!/usr/bin/env python3
"""Aggregate checkpoint-level RAS utility audits and measure temporal stability."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

UTILITY = "negative_delta_probe_loss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def correlation(left: torch.Tensor, right: torch.Tensor, *, ranks: bool) -> float:
    x = rankdata(left) if ranks else left.double()
    y = rankdata(right) if ranks else right.double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if denominator <= torch.finfo(torch.float64).eps:
        return float("nan")
    return float((x @ y / denominator).item())


def group_records(records: list[dict[str, float]]) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, float]]] = {}
    for record in records:
        grouped.setdefault(int(record["candidate"]), []).append(record)
    result = []
    for candidate, repeats in sorted(grouped.items()):
        numeric_keys = [
            key
            for key, value in repeats[0].items()
            if key not in {"candidate", "mask_repeat"} and isinstance(value, int | float)
        ]
        result.append(
            {"candidate": candidate}
            | {
                key: sum(float(record[key]) for record in repeats) / len(repeats)
                for key in numeric_keys
            }
        )
    return result


def oracle_gap(utilities: torch.Tensor) -> float:
    keep = max(int(math.ceil(0.1 * len(utilities))), 1)
    return float(torch.topk(utilities, keep).values.mean() - utilities.mean())


def bootstrap_oracle_gap(
    utilities: torch.Tensor,
    *,
    samples: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    estimates = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        draw = torch.randint(
            len(utilities),
            (len(utilities),),
            generator=generator,
        )
        estimates[index] = oracle_gap(utilities[draw])
    low, high = torch.quantile(
        estimates,
        torch.tensor([0.025, 0.975], dtype=estimates.dtype),
    )
    return float(low), float(high)


def variance_decomposition(records: list[dict[str, float]]) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    for record in records:
        grouped.setdefault(int(record["candidate"]), []).append(float(record[UTILITY]))
    means = torch.tensor(
        [sum(values) / len(values) for values in grouped.values()],
        dtype=torch.float64,
    )
    within = torch.tensor(
        [
            (value - sum(values) / len(values)) ** 2
            for values in grouped.values()
            for value in values
        ],
        dtype=torch.float64,
    )
    between_variance = float(means.var(unbiased=True)) if len(means) > 1 else 0.0
    mask_variance = float(within.mean()) if len(within) else 0.0
    return {
        "between_candidate_variance": between_variance,
        "mask_noise_variance": mask_variance,
        "signal_to_mask_noise": (
            between_variance / mask_variance if mask_variance > 0 else math.inf
        ),
    }


def score_keys(record: dict[str, Any]) -> list[str]:
    return sorted(key for key in record if key.startswith("ras_")) + [
        "jepa_loss",
        "gradient_norm",
        "random_score",
    ]


def checkpoint_summary(
    records: list[dict[str, float]],
    *,
    bootstrap_samples: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    grouped = group_records(records)
    utilities = torch.tensor([row[UTILITY] for row in grouped], dtype=torch.float64)
    keep = max(int(math.ceil(0.1 * len(grouped))), 1)
    oracle = set(torch.topk(utilities, keep).indices.tolist())
    low, high = bootstrap_oracle_gap(
        utilities,
        samples=bootstrap_samples,
        generator=generator,
    )
    scores = {}
    for key in score_keys(grouped[0]):
        values = torch.tensor([row[key] for row in grouped], dtype=torch.float64)
        predicted = set(torch.topk(values, keep).indices.tolist())
        scores[key] = {
            "pearson": correlation(values, utilities, ranks=False),
            "spearman": correlation(values, utilities, ranks=True),
            "top10_precision": len(predicted & oracle) / keep,
        }
    return {
        "num_candidates": len(grouped),
        "mask_repeats": len(records) / len(grouped),
        "mean_utility": float(utilities.mean()),
        "oracle_gap": oracle_gap(utilities),
        "oracle_gap_ci95": [low, high],
        "utility": variance_decomposition(records),
        "scores": scores,
        "candidate_scores": {
            str(int(row["candidate"])): {
                UTILITY: row[UTILITY],
                **{key: row[key] for key in score_keys(row)},
            }
            for row in grouped
        },
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    generator = torch.Generator().manual_seed(args.seed)
    checkpoints = {}
    for directory in args.audit_dirs:
        summary = json.loads((directory / "summary.json").read_text())
        epoch = int(Path(summary["checkpoint"]).stem.removeprefix("epoch_"))
        records = [
            json.loads(line)
            for line in (directory / "records.jsonl").read_text().splitlines()
        ]
        checkpoints[str(epoch)] = checkpoint_summary(
            records,
            bootstrap_samples=args.bootstrap_samples,
            generator=generator,
        )

    temporal = {}
    epochs = sorted(checkpoints, key=int)
    for left, right in zip(epochs, epochs[1:], strict=False):
        left_rows = checkpoints[left]["candidate_scores"]
        right_rows = checkpoints[right]["candidate_scores"]
        common = sorted(set(left_rows) & set(right_rows), key=int)
        temporal[f"{left}->{right}"] = {}
        for key in left_rows[common[0]]:
            left_values = torch.tensor([left_rows[index][key] for index in common])
            right_values = torch.tensor([right_rows[index][key] for index in common])
            temporal[f"{left}->{right}"][key] = correlation(
                left_values,
                right_values,
                ranks=True,
            )

    result = {"checkpoints": checkpoints, "temporal_spearman": temporal}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
