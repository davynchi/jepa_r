from __future__ import annotations

from jepa.video_selection import (
    CheckpointCandidate,
    SelectionCriterion,
    select_checkpoint,
)


def candidate(
    label: str,
    epoch: int,
    speed: float,
    fde: float,
    rank: float,
) -> CheckpointCandidate:
    return CheckpointCandidate(
        label=label,
        path=f"{label}.pt",
        epoch=epoch,
        step=epoch * 10,
        clips_seen=epoch * 100,
        scoring_clips=0,
        probe_metrics={"speed_rmse": speed, "future_fde": fde},
        pretrain_metrics={"context_token_effective_rank": rank},
    )


def test_checkpoint_selection_uses_ordered_tie_breakers() -> None:
    candidates = [
        candidate("early", 2, speed=0.60, fde=13.0, rank=20.0),
        candidate("late", 5, speed=0.60, fde=12.9, rank=10.0),
        candidate("worse", 3, speed=0.62, fde=12.0, rank=40.0),
    ]
    criteria = [
        SelectionCriterion("speed_rmse", "min", "probe"),
        SelectionCriterion("future_fde", "min", "probe"),
        SelectionCriterion("context_token_effective_rank", "max", "pretrain"),
    ]
    assert select_checkpoint(candidates, criteria).label == "late"


def test_checkpoint_selection_prefers_earlier_exact_tie() -> None:
    candidates = [
        candidate("early", 2, speed=0.60, fde=13.0, rank=20.0),
        candidate("late", 5, speed=0.60, fde=13.0, rank=20.0),
    ]
    criteria = [SelectionCriterion("speed_rmse", "min", "probe")]
    assert select_checkpoint(candidates, criteria).label == "early"
