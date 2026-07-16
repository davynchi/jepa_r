"""Synthetic dynamics and source-neutral fixed-window dataset bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, NamedTuple

import torch
from torch.utils.data import Dataset

from jepa.config import (
    BinanceDataConfig,
    DataConfig,
    DatasetConfig,
    context_steps,
    derive_seed,
    target_steps,
    validate_data_config,
)

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


class WindowSample(NamedTuple):
    context: torch.Tensor
    target: torch.Tensor
    sample_index: int


class WindowDataset(Dataset[Any]):
    """Eager fixed-length sequences with source-neutral context/target views."""

    def __init__(
        self,
        sequences: torch.Tensor,
        *,
        context_length: int,
        split: Split = "train",
        context_states: torch.Tensor | None = None,
        target_states: torch.Tensor | None = None,
    ) -> None:
        if sequences.ndim != 3 or sequences.shape[0] <= 0:
            raise ValueError("sequences must have shape [N, steps, features] with N > 0")
        if context_length <= 0 or context_length >= sequences.shape[1]:
            raise ValueError("context_length must split every sequence into non-empty windows")
        if not torch.isfinite(sequences).all():
            raise ValueError("sequences must be finite")
        for name, states in (("context_states", context_states), ("target_states", target_states)):
            if states is not None and (states.ndim != 2 or states.shape[0] != sequences.shape[0]):
                raise ValueError(f"{name} must have shape [N, state_dim]")
        self.sequences = sequences.contiguous()
        self.split = split
        self.contexts = self.sequences[:, :context_length]
        self.targets = self.sequences[:, context_length:]
        self.context_states = context_states
        self.target_states = target_states

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, index: int) -> Any:
        return WindowSample(self.contexts[index], self.targets[index], index)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    train: WindowDataset
    validation: WindowDataset
    test: WindowDataset
    fingerprint: str
    metadata: dict[str, Any]
    system: SyntheticSystem | None = None


DatasetSplits = DatasetBundle


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
        raise ValueError("system matrices must use the same dtype")
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


class LatentDynamicsDataset(WindowDataset):
    """A deterministic synthetic split whose samples are materialized once."""

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
            _trajectory_sample(config, self.system, split_seed=split_seed, index=index)
            for index in range(size)
        ]
        context_states = torch.stack([sample.context_state for sample in samples])
        target_states = torch.stack([sample.target_state for sample in samples])
        super().__init__(
            torch.stack([torch.cat((sample.context, sample.target)) for sample in samples]),
            context_length=config.window_size,
            split=split,
            context_states=context_states,
            target_states=target_states,
        )

    def __getitem__(self, index: int) -> TimeSeriesSample:
        assert self.context_states is not None and self.target_states is not None
        return TimeSeriesSample(
            self.contexts[index],
            self.targets[index],
            self.context_states[index],
            self.target_states[index],
        )


def _synthetic_fingerprint(config: DataConfig, system: SyntheticSystem) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode())
    digest.update(system.transition.detach().cpu().contiguous().numpy().tobytes())
    digest.update(system.observation.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_dataset_splits(config: DataConfig) -> DatasetBundle:
    system = generate_system(config)
    return DatasetBundle(
        train=LatentDynamicsDataset(config, "train", system=system),
        validation=LatentDynamicsDataset(config, "validation", system=system),
        test=LatentDynamicsDataset(config, "test", system=system),
        fingerprint=_synthetic_fingerprint(config, system),
        metadata={"source": "synthetic", "system_kind": config.system_kind},
        system=system,
    )


def build_dataset_bundle(config: DatasetConfig) -> DatasetBundle:
    if isinstance(config, BinanceDataConfig):
        from jepa.binance import load_prepared_binance

        return load_prepared_binance(config)
    return build_dataset_splits(config)


def resolved_data_config(config: DatasetConfig, bundle: DatasetBundle) -> DatasetConfig:
    """Pin immutable real data to its resolved content fingerprint."""
    if isinstance(config, BinanceDataConfig):
        from dataclasses import replace

        return replace(config, fingerprint=bundle.fingerprint)
    return config


def dataset_dimensions(config: DatasetConfig) -> tuple[int, int, int]:
    return context_steps(config), target_steps(config), config.observation_dim
