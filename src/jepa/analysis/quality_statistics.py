"""Block-aware inference for quality/accuracy association."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class TerminalObservation:
    run_id: str
    seed_block: int
    curriculum: str
    metric: float
    accuracy: float


def _design(values: Sequence[object]) -> np.ndarray:
    levels = tuple(dict.fromkeys(values))
    columns = [np.ones(len(values), dtype=np.float64)]
    for level in levels[1:]:
        columns.append(np.asarray([value == level for value in values], dtype=np.float64))
    return np.column_stack(columns)


def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def partial_spearman(
    metric: Sequence[float], accuracy: Sequence[float], controls: Sequence[object]
) -> float:
    if not (len(metric) == len(accuracy) == len(controls)) or len(metric) < 3:
        raise ValueError("partial Spearman inputs must align and contain at least three rows")
    metric_ranks = rankdata(np.asarray(metric), method="average")
    accuracy_ranks = rankdata(np.asarray(accuracy), method="average")
    design = _design(controls)
    metric_residual = _residualize(metric_ranks, design)
    accuracy_residual = _residualize(accuracy_ranks, design)
    denominator = np.linalg.norm(metric_residual) * np.linalg.norm(accuracy_residual)
    if denominator <= 1e-12:
        raise ValueError("constant residual metric or accuracy")
    return float(metric_residual @ accuracy_residual / denominator)


def seed_block_partial_spearman(rows: Sequence[TerminalObservation]) -> float:
    return partial_spearman(
        [row.metric for row in rows],
        [row.accuracy for row in rows],
        [row.seed_block for row in rows],
    )


def curriculum_partial_spearman(rows: Sequence[TerminalObservation]) -> float:
    return partial_spearman(
        [row.metric for row in rows],
        [row.accuracy for row in rows],
        [row.curriculum for row in rows],
    )


def complete_seed_blocks(
    rows: Sequence[TerminalObservation], *, expected_curricula: Sequence[str]
) -> tuple[TerminalObservation, ...]:
    expected = set(expected_curricula)
    by_block: dict[int, list[TerminalObservation]] = {}
    for row in rows:
        by_block.setdefault(row.seed_block, []).append(row)
    complete: list[TerminalObservation] = []
    for block_rows in by_block.values():
        if {row.curriculum for row in block_rows} == expected and len(block_rows) == len(expected):
            complete.extend(block_rows)
    return tuple(complete)


def block_bootstrap_interval(
    rows: Sequence[TerminalObservation],
    statistic: Callable[[Sequence[TerminalObservation]], float] = seed_block_partial_spearman,
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    blocks = sorted({row.seed_block for row in rows})
    if len(blocks) < 2:
        raise ValueError("block bootstrap requires at least two blocks")
    grouped = {block: [row for row in rows if row.seed_block == block] for block in blocks}
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        selected = generator.choice(blocks, size=len(blocks), replace=True)
        sample: list[TerminalObservation] = []
        for duplicate, block in enumerate(selected):
            for row in grouped[int(block)]:
                sample.append(
                    TerminalObservation(
                        f"{row.run_id}:bootstrap:{duplicate}",
                        duplicate,
                        row.curriculum,
                        row.metric,
                        row.accuracy,
                    )
                )
        try:
            estimates.append(statistic(sample))
        except ValueError:
            continue
    if not estimates:
        raise ValueError("all bootstrap samples were degenerate")
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def block_permutation_pvalue(
    rows: Sequence[TerminalObservation],
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> float:
    observed = abs(seed_block_partial_spearman(rows))
    blocks = sorted({row.seed_block for row in rows})
    generator = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        permuted: list[TerminalObservation] = []
        for block in blocks:
            block_rows = [row for row in rows if row.seed_block == block]
            accuracies = generator.permutation([row.accuracy for row in block_rows])
            permuted.extend(
                TerminalObservation(
                    row.run_id, row.seed_block, row.curriculum, row.metric, float(accuracy)
                )
                for row, accuracy in zip(block_rows, accuracies, strict=True)
            )
        if abs(seed_block_partial_spearman(permuted)) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def benjamini_hochberg(pvalues: Sequence[float]) -> tuple[float, ...]:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite vector in [0,1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank)
        adjusted[index] = running
    return tuple(float(min(value, 1.0)) for value in adjusted)


def pearson(metric: Sequence[float], accuracy: Sequence[float]) -> float:
    value = float(np.corrcoef(np.asarray(metric), np.asarray(accuracy))[0, 1])
    if not math.isfinite(value):
        raise ValueError("constant metric or accuracy")
    return value


__all__ = [
    "TerminalObservation",
    "benjamini_hochberg",
    "block_bootstrap_interval",
    "block_permutation_pvalue",
    "complete_seed_blocks",
    "curriculum_partial_spearman",
    "partial_spearman",
    "pearson",
    "seed_block_partial_spearman",
]
