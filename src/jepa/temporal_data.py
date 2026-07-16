"""Synthetic entity/context temporal world (Sections 2-4 of the experiment spec).

Hidden generative process
--------------------------
``E_t in {0,...,K-1}`` is a slowly switching entity variable and
``C_t in R^{d_C}`` is a faster-changing stationary AR(1) context variable.
Observations are ``X_t = h(E_t, C_t)`` under a fixed linear or nonlinear map
``h`` that is generated once and shared across train/validation/test splits.

Entity and context labels are returned alongside observations for *evaluation
only*. Nothing in this module feeds them into a model's forward call; callers
must explicitly opt in (e.g. an oracle ablation) if they ever want to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from jepa.config import derive_seed
from jepa.temporal_config import EntityContextDataConfig, TargetPairing

Split = str


@dataclass(frozen=True, slots=True)
class ObservationSystem:
    """The fixed, shared observation map h(E, C) -> X."""

    mode: str
    linear_map: torch.Tensor
    hidden_map: torch.Tensor | None = None
    hidden_bias: torch.Tensor | None = None
    output_map: torch.Tensor | None = None
    output_bias: torch.Tensor | None = None

    @property
    def input_dim(self) -> int:
        return self.linear_map.shape[1]

    @property
    def observation_dim(self) -> int:
        return self.linear_map.shape[0]


def _cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def generate_observation_system(
    config: EntityContextDataConfig, *, dtype: torch.dtype = torch.float32
) -> ObservationSystem:
    """Build the fixed h(E, C) map. Shared across every split."""
    input_dim = config.num_entities + config.context_dim
    generator = _cpu_generator(config.system_seed)
    linear_map = torch.randn(config.observation_dim, input_dim, generator=generator, dtype=dtype)
    linear_map = linear_map / (input_dim**0.5)
    if config.observation_mode == "linear":
        return ObservationSystem(mode="linear", linear_map=linear_map)

    hidden_dim = config.nonlinear_hidden_dim
    hidden_map = torch.randn(hidden_dim, input_dim, generator=generator, dtype=dtype) / (
        input_dim**0.5
    )
    hidden_bias = torch.randn(hidden_dim, generator=generator, dtype=dtype) * 0.1
    output_map = torch.randn(
        config.observation_dim, hidden_dim, generator=generator, dtype=dtype
    ) / (hidden_dim**0.5)
    output_bias = torch.randn(config.observation_dim, generator=generator, dtype=dtype) * 0.1
    return ObservationSystem(
        mode="nonlinear",
        linear_map=linear_map,
        hidden_map=hidden_map,
        hidden_bias=hidden_bias,
        output_map=output_map,
        output_bias=output_bias,
    )


def observe(
    system: ObservationSystem,
    entity: torch.Tensor,
    context: torch.Tensor,
    *,
    num_entities: int,
    noise_std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute X = h(E, C) + noise for batched (entity, context) inputs.

    ``entity`` has shape [...], ``context`` has shape [..., context_dim].
    """
    onehot = F.one_hot(entity, num_entities).to(context.dtype)
    u = torch.cat([onehot, context], dim=-1)
    if system.mode == "linear":
        clean = u @ system.linear_map.T
    else:
        assert system.hidden_map is not None
        hidden = torch.tanh(u @ system.hidden_map.T + system.hidden_bias)
        clean = torch.tanh(hidden @ system.output_map.T + system.output_bias)
    if noise_std > 0:
        noise = noise_std * torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        clean = clean + noise
    return clean


class Trajectory(NamedTuple):
    observations: torch.Tensor
    entity: torch.Tensor
    context: torch.Tensor
    trajectory_id: int


def _generate_entity_context_sequence(
    config: EntityContextDataConfig, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _cpu_generator(seed)
    length = config.trajectory_length
    entities = torch.empty(length, dtype=torch.long)
    contexts = torch.empty(length, config.context_dim, dtype=torch.float32)

    entity = torch.randint(config.num_entities, (), generator=generator).item()
    context = torch.randn(config.context_dim, generator=generator, dtype=torch.float32)
    rho = config.context_rho
    diffusion = (1.0 - rho**2) ** 0.5
    for t in range(length):
        entities[t] = entity
        contexts[t] = context
        if t + 1 < length:
            if torch.rand((), generator=generator).item() < config.entity_switch_probability:
                if config.num_entities > 1:
                    offset = 1 + int(
                        torch.randint(config.num_entities - 1, (), generator=generator).item()
                    )
                    entity = (entity + offset) % config.num_entities
            noise = torch.randn(config.context_dim, generator=generator, dtype=torch.float32)
            context = rho * context + diffusion * noise
    return entities, contexts


def _split_size_and_seed(config: EntityContextDataConfig, split: Split) -> tuple[int, int]:
    if split == "train":
        return config.num_train_trajectories, config.train_sample_seed
    if split == "validation":
        return config.num_val_trajectories, config.validation_sample_seed
    if split == "test":
        return config.num_test_trajectories, config.test_sample_seed
    raise ValueError(f"unknown split: {split!r}")


class EntityContextTrajectoryDataset(Dataset):
    """A deterministic, eagerly materialized split of entity/context trajectories."""

    def __init__(
        self,
        config: EntityContextDataConfig,
        split: Split,
        *,
        system: ObservationSystem | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.system = system if system is not None else generate_observation_system(config)
        size, split_seed = _split_size_and_seed(config, split)

        entities = torch.empty(size, config.trajectory_length, dtype=torch.long)
        contexts = torch.empty(size, config.trajectory_length, config.context_dim)
        for index in range(size):
            traj_seed = derive_seed(split_seed, "trajectory", index)
            entity_seq, context_seq = _generate_entity_context_sequence(config, seed=traj_seed)
            entities[index] = entity_seq
            contexts[index] = context_seq

        noise_generator = _cpu_generator(derive_seed(split_seed, "observation-noise"))
        observations = observe(
            self.system,
            entities,
            contexts,
            num_entities=config.num_entities,
            noise_std=config.observation_noise_std,
            generator=noise_generator,
        )
        self.entities = entities
        self.contexts = contexts
        self.observations = observations

    def __len__(self) -> int:
        return self.entities.shape[0]

    def __getitem__(self, index: int) -> Trajectory:
        return Trajectory(
            observations=self.observations[index],
            entity=self.entities[index],
            context=self.contexts[index],
            trajectory_id=index,
        )


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: EntityContextTrajectoryDataset
    validation: EntityContextTrajectoryDataset
    test: EntityContextTrajectoryDataset
    system: ObservationSystem


def build_dataset_splits(config: EntityContextDataConfig) -> DatasetSplits:
    system = generate_observation_system(config)
    return DatasetSplits(
        train=EntityContextTrajectoryDataset(config, "train", system=system),
        validation=EntityContextTrajectoryDataset(config, "validation", system=system),
        test=EntityContextTrajectoryDataset(config, "test", system=system),
        system=system,
    )


@dataclass(frozen=True, slots=True)
class TemporalPairs:
    """Model-facing prediction pairs plus evaluation-only side information."""

    context_view: torch.Tensor
    target_view: torch.Tensor
    horizon: int
    source_trajectory_id: torch.Tensor
    target_trajectory_id: torch.Tensor
    context_entity: torch.Tensor
    target_entity: torch.Tensor
    context_context: torch.Tensor
    target_context: torch.Tensor


def _sample_same_entity_partners(
    dataset: EntityContextTrajectoryDataset,
    source_traj_index: torch.Tensor,
    target_time_index: torch.Tensor,
    source_entity: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """For each pair, sample a different trajectory whose entity at the target
    time index matches the source entity (used by ``shuffled_same_entity``)."""
    target_traj_index = torch.empty_like(source_traj_index)
    for t2 in torch.unique(target_time_index).tolist():
        entities_at_t2 = dataset.entities[:, t2]
        time_mask = target_time_index == t2
        for entity_value in torch.unique(source_entity[time_mask]).tolist():
            group = (entities_at_t2 == entity_value).nonzero(as_tuple=True)[0]
            positions = (time_mask & (source_entity == entity_value)).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            if group.numel() < 2:
                raise ValueError(
                    "shuffled_same_entity requires at least two trajectories sharing "
                    "an entity at every target time index; increase the trajectory count"
                )
            local_choice = torch.randint(group.numel(), (positions.numel(),), generator=generator)
            chosen = group[local_choice]
            collision = chosen == source_traj_index[positions]
            if collision.any():
                local_choice[collision] = (local_choice[collision] + 1) % group.numel()
                chosen = group[local_choice]
            target_traj_index[positions] = chosen
    return target_traj_index


def make_temporal_pairs(
    dataset: EntityContextTrajectoryDataset,
    *,
    horizon: int,
    pairing: TargetPairing,
    seed: int,
) -> TemporalPairs:
    """Build (context_view, target_view) pairs for every trajectory in the split.

    ``pairing="temporal"`` uses the same trajectory at t and t+h (the standard
    condition). ``pairing="shuffled"`` keeps the context view fixed but draws
    the target view's observation from a different trajectory at the same
    time index, destroying temporal entity persistence while preserving the
    per-timestep marginal observation distribution. ``pairing="shuffled_same_entity"``
    is a cleaner control: it also draws the target from a different trajectory,
    but constrains that trajectory to have the *same* entity at the target time
    index. This destroys the temporal continuity of context while deliberately
    *preserving* entity persistence, isolating whether an effect attributed to
    "temporal persistence" is really about the entity specifically rather than
    about training on temporally contiguous data in general.
    """
    length = dataset.config.trajectory_length
    if horizon >= length:
        raise ValueError("horizon must be less than the trajectory length")
    num_trajectories = len(dataset)
    valid_starts = length - horizon
    n_pairs = num_trajectories * valid_starts

    traj_index = torch.arange(num_trajectories).repeat_interleave(valid_starts)
    time_index = torch.arange(valid_starts).repeat(num_trajectories)

    context_view = dataset.observations[traj_index, time_index]
    context_entity = dataset.entities[traj_index, time_index]
    context_context = dataset.contexts[traj_index, time_index]

    target_time_index = time_index + horizon
    if pairing == "temporal":
        target_traj_index = traj_index
    elif pairing == "shuffled":
        generator = _cpu_generator(seed)
        offset = 1 + torch.randint(num_trajectories - 1, (n_pairs,), generator=generator)
        target_traj_index = (traj_index + offset) % num_trajectories
    elif pairing == "shuffled_same_entity":
        generator = _cpu_generator(seed)
        target_traj_index = _sample_same_entity_partners(
            dataset, traj_index, target_time_index, context_entity, generator
        )
    else:
        raise ValueError(f"unknown target pairing: {pairing!r}")

    target_view = dataset.observations[target_traj_index, target_time_index]
    target_entity = dataset.entities[target_traj_index, target_time_index]
    target_context = dataset.contexts[target_traj_index, target_time_index]

    return TemporalPairs(
        context_view=context_view,
        target_view=target_view,
        horizon=horizon,
        source_trajectory_id=traj_index,
        target_trajectory_id=target_traj_index,
        context_entity=context_entity,
        target_entity=target_entity,
        context_context=context_context,
        target_context=target_context,
    )


@dataclass(frozen=True, slots=True)
class HierarchicalPairs:
    """Aligned (context, short-horizon target, long-horizon target) triples."""

    context_view: torch.Tensor
    short_target_view: torch.Tensor
    long_target_view: torch.Tensor
    context_entity: torch.Tensor
    context_context: torch.Tensor


def make_hierarchical_pairs(
    dataset: EntityContextTrajectoryDataset,
    *,
    short_horizon: int,
    long_horizon: int,
    pairing: TargetPairing,
    seed: int,
) -> HierarchicalPairs:
    """Build triples sharing one context view t and two future targets t+h_C, t+h_E."""
    length = dataset.config.trajectory_length
    if long_horizon >= length or short_horizon >= length:
        raise ValueError("horizons must be less than the trajectory length")
    num_trajectories = len(dataset)
    valid_starts = length - long_horizon
    n_pairs = num_trajectories * valid_starts

    traj_index = torch.arange(num_trajectories).repeat_interleave(valid_starts)
    time_index = torch.arange(valid_starts).repeat(num_trajectories)

    context_view = dataset.observations[traj_index, time_index]
    context_entity = dataset.entities[traj_index, time_index]
    context_context = dataset.contexts[traj_index, time_index]

    if pairing == "temporal":
        short_traj_index = traj_index
        long_traj_index = traj_index
    else:
        generator = _cpu_generator(seed)
        short_offset = 1 + torch.randint(num_trajectories - 1, (n_pairs,), generator=generator)
        long_offset = 1 + torch.randint(num_trajectories - 1, (n_pairs,), generator=generator)
        short_traj_index = (traj_index + short_offset) % num_trajectories
        long_traj_index = (traj_index + long_offset) % num_trajectories

    short_target_view = dataset.observations[short_traj_index, time_index + short_horizon]
    long_target_view = dataset.observations[long_traj_index, time_index + long_horizon]

    return HierarchicalPairs(
        context_view=context_view,
        short_target_view=short_target_view,
        long_target_view=long_target_view,
        context_entity=context_entity,
        context_context=context_context,
    )


@dataclass(frozen=True, slots=True)
class CounterfactualPairs:
    """Controlled pairs that change exactly one hidden factor at a time."""

    same_entity: torch.Tensor
    same_entity_context_1: torch.Tensor
    same_entity_context_2: torch.Tensor
    same_entity_x1: torch.Tensor
    same_entity_x2: torch.Tensor
    diff_entity_entities: torch.Tensor
    diff_entity_shared_context: torch.Tensor
    diff_entity_x1: torch.Tensor
    diff_entity_x2: torch.Tensor


def build_counterfactual_pairs(
    config: EntityContextDataConfig,
    system: ObservationSystem,
    *,
    num_pairs: int,
    seed: int,
) -> CounterfactualPairs:
    """Build the Section 4 counterfactual evaluation pairs."""
    generator = _cpu_generator(seed)

    same_entity = torch.randint(config.num_entities, (num_pairs,), generator=generator)
    context_1 = torch.randn(num_pairs, config.context_dim, generator=generator)
    context_2 = torch.randn(num_pairs, config.context_dim, generator=generator)
    same_entity_x1 = observe(
        system, same_entity, context_1, num_entities=config.num_entities, noise_std=0.0
    )
    same_entity_x2 = observe(
        system, same_entity, context_2, num_entities=config.num_entities, noise_std=0.0
    )

    entity_1 = torch.randint(config.num_entities, (num_pairs,), generator=generator)
    offset = 1 + torch.randint(config.num_entities - 1, (num_pairs,), generator=generator)
    entity_2 = (entity_1 + offset) % config.num_entities
    shared_context = torch.randn(num_pairs, config.context_dim, generator=generator)
    diff_entity_x1 = observe(
        system, entity_1, shared_context, num_entities=config.num_entities, noise_std=0.0
    )
    diff_entity_x2 = observe(
        system, entity_2, shared_context, num_entities=config.num_entities, noise_std=0.0
    )

    return CounterfactualPairs(
        same_entity=same_entity,
        same_entity_context_1=context_1,
        same_entity_context_2=context_2,
        same_entity_x1=same_entity_x1,
        same_entity_x2=same_entity_x2,
        diff_entity_entities=torch.stack([entity_1, entity_2], dim=1),
        diff_entity_shared_context=shared_context,
        diff_entity_x1=diff_entity_x1,
        diff_entity_x2=diff_entity_x2,
    )
