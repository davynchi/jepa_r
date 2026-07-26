"""Checkpoint selection utilities for Moving-MNIST representation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

SelectionMode = Literal["min", "max"]
SelectionSource = Literal["probe", "pretrain"]


@dataclass(frozen=True)
class SelectionCriterion:
    metric: str
    mode: SelectionMode
    source: SelectionSource = "probe"

    def validate(self) -> None:
        if not self.metric:
            raise ValueError("selection metric cannot be empty")
        if self.mode not in {"min", "max"}:
            raise ValueError(f"unknown selection mode: {self.mode}")
        if self.source not in {"probe", "pretrain"}:
            raise ValueError(f"unknown selection source: {self.source}")


@dataclass(frozen=True)
class CheckpointCandidate:
    label: str
    path: str
    epoch: int
    step: int
    clips_seen: int
    scoring_clips: int
    probe_metrics: Mapping[str, float]
    pretrain_metrics: Mapping[str, float]

    def value(self, criterion: SelectionCriterion) -> float:
        values = (
            self.probe_metrics
            if criterion.source == "probe"
            else self.pretrain_metrics
        )
        if criterion.metric not in values:
            raise KeyError(
                f"candidate {self.label!r} is missing "
                f"{criterion.source} metric {criterion.metric!r}"
            )
        return float(values[criterion.metric])


def _ordering_value(value: float, mode: SelectionMode) -> float:
    return value if mode == "min" else -value


def select_checkpoint(
    candidates: Sequence[CheckpointCandidate],
    criteria: Sequence[SelectionCriterion],
) -> CheckpointCandidate:
    """Select one checkpoint using a deterministic lexicographic rule.

    Each criterion is applied in order. Lower values are preferred for ``min``
    criteria and higher values for ``max`` criteria. Epoch is used as the final
    deterministic tie-breaker, preferring the earlier checkpoint.
    """

    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    if not criteria:
        raise ValueError("at least one selection criterion is required")
    for criterion in criteria:
        criterion.validate()

    def key(candidate: CheckpointCandidate) -> tuple[float, ...]:
        ordered = tuple(
            _ordering_value(candidate.value(criterion), criterion.mode)
            for criterion in criteria
        )
        return (*ordered, float(candidate.epoch), float(candidate.step))

    return min(candidates, key=key)


def criteria_from_config(config: Mapping[str, object]) -> tuple[SelectionCriterion, ...]:
    """Parse the ``checkpoint_selection`` config section.

    The default intentionally uses a linear regression metric that the current
    random attentive probe does not already solve: speed RMSE, followed by FDE
    and token effective rank.
    """

    section = config.get("checkpoint_selection", {})
    if not isinstance(section, Mapping):
        raise ValueError("checkpoint_selection must be a mapping")

    primary = section.get(
        "primary",
        {"metric": "speed_rmse", "mode": "min", "source": "probe"},
    )
    tie_breakers = section.get(
        "tie_breakers",
        [
            {"metric": "future_fde", "mode": "min", "source": "probe"},
            {
                "metric": "context_token_effective_rank",
                "mode": "max",
                "source": "pretrain",
            },
        ],
    )
    raw_items = [primary, *list(tie_breakers)]
    result: list[SelectionCriterion] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            raise ValueError("each checkpoint selection criterion must be a mapping")
        criterion = SelectionCriterion(
            metric=str(item["metric"]),
            mode=str(item.get("mode", "max")),  # type: ignore[arg-type]
            source=str(item.get("source", "probe")),  # type: ignore[arg-type]
        )
        criterion.validate()
        result.append(criterion)
    return tuple(result)
