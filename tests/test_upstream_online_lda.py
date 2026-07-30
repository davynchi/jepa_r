from pathlib import Path

import torch

from jepa.analysis.upstream_online_lda import (
    checkpoint_epochs,
    final_epoch_indices,
)


def test_checkpoint_epochs_matches_official_save_loop() -> None:
    displayed, sampler = checkpoint_epochs(
        Path("jepa-ep50.pth.tar"),
        {"epoch": 49},
    )

    assert displayed == 50
    assert sampler == 48


def test_final_epoch_indices_replays_sampler_and_drop_last() -> None:
    dataset = list(range(13))
    actual = final_epoch_indices(
        dataset,
        sampler_epoch=4,
        batch_size=4,
        online_size=8,
        world_size=1,
        rank=0,
        seed=0,
    )
    order = torch.randperm(13, generator=torch.Generator().manual_seed(4))

    assert torch.equal(actual, order[:12][-8:])
