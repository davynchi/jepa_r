"""Replay helpers for online metrics from official I-JEPA checkpoints."""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path

import torch
from torch.utils.data.distributed import DistributedSampler


def checkpoint_number(path: Path) -> int:
    marker = path.name.removeprefix("jepa-ep").split(".", maxsplit=1)[0]
    return int(marker)


def checkpoint_epochs(path: Path, checkpoint: dict[str, object]) -> tuple[int, int]:
    """Return displayed and sampler epochs for the official fork's save loop."""
    displayed_epoch = checkpoint_number(path)
    stored_epoch = int(checkpoint["epoch"])
    if stored_epoch + 1 != displayed_epoch:
        raise ValueError(
            f"{path.name} denotes epoch {displayed_epoch}, but stores epoch "
            f"{stored_epoch}"
        )
    sampler_epoch = stored_epoch - 1
    if sampler_epoch < 0:
        raise ValueError(f"{path.name} predates the first completed train epoch")
    return displayed_epoch, sampler_epoch


def final_epoch_indices(
    dataset: Sized,
    *,
    sampler_epoch: int,
    batch_size: int,
    online_size: int,
    world_size: int,
    rank: int,
    seed: int,
) -> torch.Tensor:
    """Reproduce the final samples retained by DataLoader(drop_last=True)."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    sampler.set_epoch(sampler_epoch)
    order = torch.tensor(list(iter(sampler)), dtype=torch.long)
    retained = (len(order) // batch_size) * batch_size
    if retained < online_size:
        raise ValueError(
            f"only {retained} samples survive drop_last, below online-size={online_size}"
        )
    return order[:retained][-online_size:]


__all__ = ["checkpoint_epochs", "checkpoint_number", "final_epoch_indices"]
