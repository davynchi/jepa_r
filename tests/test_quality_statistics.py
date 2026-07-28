from __future__ import annotations

import pytest

from jepa.analysis.quality_statistics import (
    TerminalObservation,
    benjamini_hochberg,
    block_bootstrap_interval,
    block_permutation_pvalue,
    complete_seed_blocks,
    curriculum_partial_spearman,
    seed_block_partial_spearman,
)

CURRICULA = ("uniform", "loss", "ras")


def _rows() -> tuple[TerminalObservation, ...]:
    rows = []
    for block in range(5):
        for curriculum_index, curriculum in enumerate(CURRICULA):
            metric = curriculum_index + block * 0.01
            accuracy = 0.4 + curriculum_index * 0.1 + block * 0.001
            rows.append(
                TerminalObservation(f"{block}-{curriculum}", block, curriculum, metric, accuracy)
            )
    return tuple(rows)


def test_seed_block_primary_preserves_curriculum_signal() -> None:
    assert seed_block_partial_spearman(_rows()) > 0.99
    assert abs(curriculum_partial_spearman(_rows())) > 0.9


def test_complete_blocks_drop_partial_block() -> None:
    rows = _rows()[:-1]
    complete = complete_seed_blocks(rows, expected_curricula=CURRICULA)
    assert len(complete) == 12
    assert {row.seed_block for row in complete} == {0, 1, 2, 3}


def test_block_bootstrap_and_permutation_are_deterministic() -> None:
    first = block_bootstrap_interval(_rows(), iterations=100, seed=4)
    second = block_bootstrap_interval(_rows(), iterations=100, seed=4)
    assert first == second
    assert first[0] > 0.99
    pvalue = block_permutation_pvalue(_rows(), iterations=200, seed=5)
    assert 0 < pvalue < 0.05


def test_bh_matches_known_example_and_rejects_invalid_values() -> None:
    assert benjamini_hochberg((0.01, 0.04, 0.03, 0.2)) == pytest.approx(
        (0.04, 0.0533333333, 0.0533333333, 0.2)
    )
    with pytest.raises(ValueError):
        benjamini_hochberg((1.2,))
