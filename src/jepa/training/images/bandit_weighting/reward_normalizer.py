"""Discounted online normalization for non-stationary bandit rewards."""

from __future__ import annotations

import math
from typing import Any


class DiscountedRewardNormalizer:
    def __init__(self, *, decay: float = 0.99, epsilon: float = 1.0e-8, clip: float = 5.0) -> None:
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")
        if epsilon <= 0 or clip <= 0:
            raise ValueError("epsilon and clip must be positive")
        self.decay = decay
        self.epsilon = epsilon
        self.clip = clip
        self.mean = 0.0
        self.second_moment = 0.0
        self.num_updates = 0

    @property
    def variance(self) -> float:
        return max(self.second_moment - self.mean**2, self.epsilon)

    def update(self, reward: float) -> float:
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError("reward must be finite")
        if self.num_updates == 0:
            self.mean = value
            self.second_moment = value**2
            self.num_updates = 1
            return 0.0
        self.mean = self.decay * self.mean + (1.0 - self.decay) * value
        self.second_moment = self.decay * self.second_moment + (1.0 - self.decay) * value**2
        self.num_updates += 1
        normalized = (value - self.mean) / math.sqrt(self.variance)
        return max(-self.clip, min(self.clip, normalized))

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "mean": self.mean,
            "second_moment": self.second_moment,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("reward normalizer decay does not match checkpoint")
        if float(state["epsilon"]) != self.epsilon:
            raise ValueError("reward normalizer epsilon does not match checkpoint")
        if float(state["clip"]) != self.clip:
            raise ValueError("reward normalizer clip does not match checkpoint")
        self.mean = float(state["mean"])
        self.second_moment = float(state["second_moment"])
        self.num_updates = int(state["num_updates"])
