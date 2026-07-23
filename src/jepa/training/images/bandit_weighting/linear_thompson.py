"""Discounted linear Thompson sampling for non-stationary data utility."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class LinearThompsonConfig:
    context_dim: int
    prior_precision: float = 1.0
    observation_noise: float = 1.0
    discount: float = 0.99
    exploration_scale: float = 1.0
    temperature: float = 1.0
    uniform_mix: float = 0.1
    fit_intercept: bool = True
    normalize_scores: bool = True

    def __post_init__(self) -> None:
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.prior_precision <= 0:
            raise ValueError("prior_precision must be positive")
        if self.observation_noise <= 0:
            raise ValueError("observation_noise must be positive")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if self.exploration_scale < 0:
            raise ValueError("exploration_scale must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.uniform_mix <= 1:
            raise ValueError("uniform_mix must be in [0, 1]")


class DiscountedLinearThompsonSampler:
    """Bayesian linear utility model with exponential posterior forgetting."""

    def __init__(
        self,
        config: LinearThompsonConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_dim = config.context_dim + int(config.fit_intercept)
        self._prior_eye = torch.eye(self.model_dim, device=self.device, dtype=dtype)
        self.precision = config.prior_precision * self._prior_eye.clone()
        self.information = torch.zeros(self.model_dim, device=self.device, dtype=dtype)
        self.num_updates = 0

    def _features(self, contexts: torch.Tensor) -> torch.Tensor:
        features = contexts.detach().to(device=self.device, dtype=self.dtype)
        if features.ndim != 2 or features.shape[1] != self.config.context_dim:
            raise ValueError(f"contexts must have shape [batch, {self.config.context_dim}]")
        if features.shape[0] == 0:
            raise ValueError("contexts must be non-empty")
        if not torch.isfinite(features).all():
            raise ValueError("contexts must be finite")
        if self.config.fit_intercept:
            ones = torch.ones((features.shape[0], 1), device=self.device, dtype=self.dtype)
            features = torch.cat((features, ones), dim=1)
        return features

    def update_batch(self, contexts: torch.Tensor, batch_reward: float) -> None:
        features = self._features(contexts)
        self.update_aggregate(features.mean(dim=0), batch_reward, augmented=True)

    def update_aggregate(
        self,
        aggregate_context: torch.Tensor,
        reward: float,
        *,
        augmented: bool = False,
    ) -> None:
        context = aggregate_context.detach().to(device=self.device, dtype=self.dtype).flatten()
        if augmented:
            if context.numel() != self.model_dim:
                raise ValueError(f"augmented context must contain {self.model_dim} values")
        else:
            if context.numel() != self.config.context_dim:
                raise ValueError(f"context must contain {self.config.context_dim} values")
            context = self._features(context.unsqueeze(0)).squeeze(0)
        reward_tensor = torch.as_tensor(reward, device=self.device, dtype=self.dtype)
        if reward_tensor.numel() != 1 or not torch.isfinite(reward_tensor):
            raise ValueError("reward must be a finite scalar")

        discount = self.config.discount
        self.precision.mul_(discount)
        self.precision.add_(
            self._prior_eye,
            alpha=(1.0 - discount) * self.config.prior_precision,
        )
        self.information.mul_(discount)
        noise = self.config.observation_noise
        self.precision.add_(torch.outer(context, context), alpha=1.0 / noise)
        self.information.add_(context, alpha=float(reward_tensor.item()) / noise)
        self.num_updates += 1

    def posterior_mean(self) -> torch.Tensor:
        return torch.linalg.solve(self.precision, self.information)

    def posterior_covariance(self) -> torch.Tensor:
        return torch.linalg.inv(self.precision) * self.config.exploration_scale**2

    def sample_parameters(self, *, generator: torch.Generator | None = None) -> torch.Tensor:
        mean = self.posterior_mean()
        if self.config.exploration_scale == 0:
            return mean
        cholesky = torch.linalg.cholesky(self.precision)
        noise = torch.randn(
            self.model_dim,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        posterior_noise = torch.linalg.solve_triangular(
            cholesky.mT, noise.unsqueeze(1), upper=True
        ).squeeze(1)
        return mean + self.config.exploration_scale * posterior_noise

    def sample_scores(
        self,
        contexts: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self._features(contexts) @ self.sample_parameters(generator=generator)

    def mean_scores(self, contexts: torch.Tensor) -> torch.Tensor:
        return self._features(contexts) @ self.posterior_mean()

    def predict_batch_reward(self, contexts: torch.Tensor) -> float:
        return float(self.mean_scores(contexts).mean().item())

    def sampling_probabilities(
        self,
        contexts: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.probabilities_from_scores(self.sample_scores(contexts, generator=generator))

    def probabilities_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        values = scores.detach().to(device=self.device, dtype=self.dtype).flatten()
        if values.numel() == 0:
            raise ValueError("scores must be non-empty")
        if not torch.isfinite(values).all():
            raise ValueError("scores must be finite")
        if self.config.normalize_scores:
            values = values - values.mean()
            std = values.std(unbiased=False)
            if std > 0:
                values = values / std
            else:
                values.zero_()
        probabilities = torch.softmax(values / self.config.temperature, dim=0)
        if self.config.uniform_mix > 0:
            uniform = torch.full_like(probabilities, 1.0 / probabilities.numel())
            probabilities = (
                1.0 - self.config.uniform_mix
            ) * probabilities + self.config.uniform_mix * uniform
        return probabilities / probabilities.sum()

    def draw_policy(
        self,
        contexts: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.sample_scores(contexts, generator=generator)
        return scores, self.probabilities_from_scores(scores)

    def sample_indices(
        self,
        contexts: torch.Tensor,
        *,
        num_samples: int,
        generator: torch.Generator | None = None,
        replacement: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        probabilities = self.sampling_probabilities(contexts, generator=generator)
        if not replacement and num_samples > probabilities.numel():
            raise ValueError("cannot draw more samples than contexts without replacement")
        indices = torch.multinomial(
            probabilities,
            num_samples,
            replacement=replacement,
            generator=generator,
        )
        return indices, probabilities

    def diagnostics(self) -> dict[str, float]:
        covariance = self.posterior_covariance()
        condition = torch.linalg.cond(self.precision)
        return {
            "bandit/num_updates": float(self.num_updates),
            "bandit/posterior_mean_norm": float(
                torch.linalg.vector_norm(self.posterior_mean()).item()
            ),
            "bandit/posterior_trace": float(torch.trace(covariance).item()),
            "bandit/precision_condition": float(condition.item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "precision": self.precision.detach().cpu(),
            "information": self.information.detach().cpu(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("config") != asdict(self.config):
            raise ValueError("Thompson sampler config does not match checkpoint")
        precision = torch.as_tensor(state["precision"])
        information = torch.as_tensor(state["information"])
        if precision.shape != self.precision.shape:
            raise ValueError("checkpoint precision shape does not match sampler")
        if information.shape != self.information.shape:
            raise ValueError("checkpoint information shape does not match sampler")
        self.precision.copy_(precision.to(device=self.device, dtype=self.dtype))
        self.information.copy_(information.to(device=self.device, dtype=self.dtype))
        self.num_updates = int(state["num_updates"])
