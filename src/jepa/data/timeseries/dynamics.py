"""Controlled, eagerly materialized synthetic time-series dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from torch.utils.data import Dataset

from jepa.configs.base import DataConfig, derive_seed, validate_data_config

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class SyntheticSystem:
    transition: torch.Tensor
    observation: torch.Tensor


class TimeSeriesSample(NamedTuple):
    context: torch.Tensor
    target: torch.Tensor
    context_state: torch.Tensor
    target_state: torch.Tensor


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: LatentDynamicsDataset
    validation: LatentDynamicsDataset
    test: LatentDynamicsDataset
    system: SyntheticSystem


def _cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def generate_system(config: DataConfig, *, dtype: torch.dtype = torch.float32) -> SyntheticSystem:
    """Generate a contraction transition and a well-conditioned observation map."""
    validate_data_config(config)
    generator = _cpu_generator(config.system_seed)
    raw_transition = torch.randn(
        config.latent_state_dim,
        config.latent_state_dim,
        generator=generator,
        dtype=torch.float64,
    )
    operator_norm = torch.linalg.matrix_norm(raw_transition, ord=2)
    if not torch.isfinite(operator_norm) or operator_norm <= 0:
        raise RuntimeError("failed to generate a finite non-zero transition matrix")
    transition = config.transition_norm * raw_transition / operator_norm

    raw_observation = torch.randn(
        config.observation_dim,
        config.latent_state_dim,
        generator=generator,
        dtype=torch.float64,
    )
    observation, upper = torch.linalg.qr(raw_observation, mode="reduced")
    diagonal = torch.diagonal(upper)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    observation = observation * signs.unsqueeze(0)

    return SyntheticSystem(transition.to(dtype=dtype), observation.to(dtype=dtype))


def _validate_system(config: DataConfig, system: SyntheticSystem) -> None:
    expected_transition = (config.latent_state_dim, config.latent_state_dim)
    expected_observation = (config.observation_dim, config.latent_state_dim)
    if system.transition.shape != expected_transition:
        raise ValueError(
            f"transition shape must be {expected_transition}, got {tuple(system.transition.shape)}"
        )
    if system.observation.shape != expected_observation:
        raise ValueError(
            "observation shape must be "
            f"{expected_observation}, got {tuple(system.observation.shape)}"
        )
    if system.transition.dtype != system.observation.dtype:
        raise ValueError("transition and observation matrices must use the same dtype")
    if not torch.isfinite(system.transition).all() or not torch.isfinite(system.observation).all():
        raise ValueError("system matrices must be finite")


def _split_size_and_seed(config: DataConfig, split: Split) -> tuple[int, int]:
    if split == "train":
        return config.num_samples, config.train_sample_seed
    if split == "validation":
        return config.validation_samples, config.validation_sample_seed
    if split == "test":
        return config.test_samples, config.test_sample_seed
    raise ValueError(f"unknown split: {split!r}")


def _trajectory_sample(
    config: DataConfig,
    system: SyntheticSystem,
    *,
    split_seed: int,
    index: int,
) -> TimeSeriesSample:
    generator = _cpu_generator(derive_seed(split_seed, "trajectory", index))
    total_length = config.burn_in_steps + config.sequence_length
    dtype = system.transition.dtype
    state = torch.randn(config.latent_state_dim, generator=generator, dtype=dtype)
    states = torch.empty(total_length, config.latent_state_dim, dtype=dtype)
    observations = torch.empty(total_length, config.observation_dim, dtype=dtype)

    for step in range(total_length):
        states[step] = state
        observation_noise = config.observation_noise * torch.randn(
            config.observation_dim, generator=generator, dtype=dtype
        )
        observations[step] = system.observation @ state + observation_noise
        if step + 1 < total_length:
            next_state = system.transition @ state
            if config.system_kind == "nonlinear":
                next_state = torch.tanh(next_state)
            process_noise = config.process_noise * torch.randn(
                config.latent_state_dim, generator=generator, dtype=dtype
            )
            state = next_state + process_noise

    states = states[config.burn_in_steps :]
    observations = observations[config.burn_in_steps :]
    max_start = config.sequence_length - 2 * config.window_size
    start = int(torch.randint(max_start + 1, (), generator=generator).item())
    midpoint = start + config.window_size
    end = midpoint + config.window_size
    return TimeSeriesSample(
        context=observations[start:midpoint],
        target=observations[midpoint:end],
        context_state=states[midpoint - 1],
        target_state=states[end - 1],
    )


class LatentDynamicsDataset(Dataset[TimeSeriesSample]):
    """A deterministic split whose returned samples are materialized once."""

    def __init__(
        self,
        config: DataConfig,
        split: Split,
        *,
        system: SyntheticSystem | None = None,
    ) -> None:
        validate_data_config(config)
        self.config = config
        self.split = split
        self.system = system if system is not None else generate_system(config)
        _validate_system(config, self.system)
        size, split_seed = _split_size_and_seed(config, split)
        samples = [
            _trajectory_sample(
                config,
                self.system,
                split_seed=split_seed,
                index=index,
            )
            for index in range(size)
        ]
        self.contexts = torch.stack([sample.context for sample in samples])
        self.targets = torch.stack([sample.target for sample in samples])
        self.context_states = torch.stack([sample.context_state for sample in samples])
        self.target_states = torch.stack([sample.target_state for sample in samples])

    def __len__(self) -> int:
        return self.contexts.shape[0]

    def __getitem__(self, index: int) -> TimeSeriesSample:
        return TimeSeriesSample(
            self.contexts[index],
            self.targets[index],
            self.context_states[index],
            self.target_states[index],
        )


def build_dataset_splits(config: DataConfig) -> DatasetSplits:
    system = generate_system(config)
    return DatasetSplits(
        train=LatentDynamicsDataset(config, "train", system=system),
        validation=LatentDynamicsDataset(config, "validation", system=system),
        test=LatentDynamicsDataset(config, "test", system=system),
        system=system,
    )
