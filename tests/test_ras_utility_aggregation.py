from __future__ import annotations

import math
import runpy
from pathlib import Path
from types import SimpleNamespace

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

AUDIT_MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analysis"
        / "audit_ras_local_utility.py"
    ),
    run_name="ras_local_utility",
)
build_probe_matrix = AUDIT_MODULE["build_probe_matrix"]
probe_metrics = AUDIT_MODULE["probe_metrics"]


def records() -> list[dict[str, float]]:
    rows = []
    for candidate, utility in enumerate((0.0, 1.0, 2.0, 3.0)):
        for repeat, noise in enumerate((-0.1, 0.1)):
            rows.append(
                {
                    "candidate": candidate,
                    "mask_repeat": repeat,
                    "probe_target_linear_utility_loss": utility + noise,
                    "probe_target_linear_delta_accuracy": utility / 10 + noise,
                    "ras_barlow_adamw_dot": utility,
                    "jepa_loss": -utility,
                    "gradient_norm": 1.0,
                    "random_score": float(candidate % 2),
                }
            )
    return rows


def test_group_records_averages_mask_repeats() -> None:
    grouped = group_records(records())
    assert [row["probe_target_linear_utility_loss"] for row in grouped] == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]


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
    utility = summary["utilities"]["probe_target_linear_utility_loss"]
    assert math.isclose(
        utility["scores"]["ras_barlow_adamw_dot"]["spearman"],
        1.0,
    )
    assert utility["scores"]["ras_barlow_adamw_dot"]["top10_precision"] == 1.0
    assert utility["variance"]["signal_to_mask_noise"] > 1.0


def test_probe_matrix_builds_all_context_and_target_probes() -> None:
    generator = torch.Generator().manual_seed(0)
    encoded = {}
    for encoder_name in ("context", "target"):
        encoded[encoder_name] = {
            "train": torch.randn(18, 6, generator=generator),
            "validation": torch.randn(9, 6, generator=generator),
            "test": torch.randn(9, 6, generator=generator),
        }
    labels = {
        "train": torch.arange(18) % 3,
        "validation": torch.arange(9) % 3,
        "test": torch.arange(9) % 3,
    }
    args = SimpleNamespace(
        probe_encoders=("context", "target"),
        probe_modes=("ridge", "linear", "linear-l2", "mlp"),
        probe_ridge=1.0e-3,
        probe_batch_size=9,
        probe_epochs=2,
        probe_learning_rate=1.0e-2,
        probe_weight_decay=0.0,
        mlp_epochs=2,
        mlp_hidden_dim=8,
        mlp_learning_rate=1.0e-2,
        mlp_weight_decay=0.0,
        seed=0,
    )
    probes = build_probe_matrix(
        encoded,
        labels,
        num_classes=3,
        args=args,
        device=torch.device("cpu"),
    )
    assert len(probes) == 8
    for probe in probes.values():
        loss, accuracy = probe_metrics(
            probe,
            encoded[probe.encoder_name]["test"],
            labels["test"],
            device=torch.device("cpu"),
            batch_size=9,
        )
        assert math.isfinite(loss)
        assert 0 <= accuracy <= 1
