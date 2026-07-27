from __future__ import annotations

import math
import runpy
from pathlib import Path

import torch

MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analysis"
        / "aggregate_ras_utility_audits.py"
    ),
    run_name="ras_utility_aggregation",
)
bootstrap_oracle_gap = MODULE["bootstrap_oracle_gap"]
checkpoint_summary = MODULE["checkpoint_summary"]
group_records = MODULE["group_records"]
oracle_gap = MODULE["oracle_gap"]


def records() -> list[dict[str, float]]:
    rows = []
    for candidate, utility in enumerate((0.0, 1.0, 2.0, 3.0)):
        for repeat, noise in enumerate((-0.1, 0.1)):
            rows.append(
                {
                    "candidate": candidate,
                    "mask_repeat": repeat,
                    "negative_delta_probe_loss": utility + noise,
                    "ras_barlow_adamw_dot": utility,
                    "jepa_loss": -utility,
                    "gradient_norm": 1.0,
                    "random_score": float(candidate % 2),
                }
            )
    return rows


def test_group_records_averages_mask_repeats() -> None:
    grouped = group_records(records())
    assert [row["negative_delta_probe_loss"] for row in grouped] == [0.0, 1.0, 2.0, 3.0]


def test_oracle_gap_compares_top_decile_to_uniform_mean() -> None:
    utilities = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    assert oracle_gap(utilities) == 1.5
    low, high = bootstrap_oracle_gap(
        utilities,
        samples=100,
        generator=torch.Generator().manual_seed(0),
    )
    assert math.isfinite(low)
    assert low <= high


def test_checkpoint_summary_separates_signal_from_mask_noise() -> None:
    summary = checkpoint_summary(
        records(),
        bootstrap_samples=100,
        generator=torch.Generator().manual_seed(0),
    )
    assert math.isclose(
        summary["scores"]["ras_barlow_adamw_dot"]["spearman"],
        1.0,
    )
    assert summary["scores"]["ras_barlow_adamw_dot"]["top10_precision"] == 1.0
    assert summary["utility"]["signal_to_mask_noise"] > 1.0
