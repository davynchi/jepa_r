import torch

from jepa.training.images.ijepa_spatial import MaskConfig
from jepa.training.images.mask_loader import IndexMaskLoader, apply_prepared_crop


def _collect_epoch(
    loader: IndexMaskLoader,
    order: torch.Tensor,
    *,
    seed: int,
) -> list[tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]]:
    return [
        (
            batch.indices.clone(),
            [mask.clone() for mask in batch.context_masks],
            [mask.clone() for mask in batch.target_masks],
        )
        for batch in loader.iter_epoch(order, seed=seed)
    ]


def test_mask_loader_preserves_weighted_order_and_batch_shapes() -> None:
    order = torch.tensor([5, 5, 2, 9, 1, 1, 7, 3, 4, 8])
    loader = IndexMaskLoader(
        num_draws=len(order),
        batch_size=4,
        grid=8,
        mask_config=MaskConfig(),
        source_size=(64, 64),
        crop_scale=(0.3, 1.0),
        horizontal_flip_probability=0.5,
        num_workers=0,
        pin_memory=False,
    )

    batches = _collect_epoch(loader, order, seed=123)

    assert torch.cat([batch[0] for batch in batches]).tolist() == order.tolist()
    assert [len(batch[0]) for batch in batches] == [4, 4, 2]
    for indices, context_masks, target_masks in batches:
        assert len(context_masks) == 1
        assert len(target_masks) == 4
        assert context_masks[0].shape[0] == len(indices)
        assert all(mask.shape[0] == len(indices) for mask in target_masks)


def test_mask_loader_is_deterministic_for_epoch_seed() -> None:
    order = torch.arange(12)
    loader = IndexMaskLoader(
        num_draws=len(order),
        batch_size=4,
        grid=8,
        mask_config=MaskConfig(),
        source_size=(64, 64),
        crop_scale=(0.3, 1.0),
        horizontal_flip_probability=0.5,
        num_workers=0,
        pin_memory=False,
    )

    first = _collect_epoch(loader, order, seed=321)
    second = _collect_epoch(loader, order, seed=321)

    for first_batch, second_batch in zip(first, second, strict=True):
        assert torch.equal(first_batch[0], second_batch[0])
        for first_masks, second_masks in (
            (first_batch[1], second_batch[1]),
            (first_batch[2], second_batch[2]),
        ):
            assert all(
                torch.equal(first_mask, second_mask)
                for first_mask, second_mask in zip(first_masks, second_masks, strict=True)
            )


def test_mask_loader_prefetches_with_spawn_workers() -> None:
    order = torch.tensor([7, 2, 2, 8, 1, 9, 4, 4])
    loader = IndexMaskLoader(
        num_draws=len(order),
        batch_size=4,
        grid=8,
        mask_config=MaskConfig(),
        source_size=(64, 64),
        crop_scale=(0.3, 1.0),
        horizontal_flip_probability=0.5,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=False,
    )

    batches = _collect_epoch(loader, order, seed=456)

    assert torch.cat([batch[0] for batch in batches]).tolist() == order.tolist()


def test_mask_loader_prepares_deterministic_crop_geometry() -> None:
    order = torch.arange(4)
    loader = IndexMaskLoader(
        num_draws=4,
        batch_size=4,
        grid=8,
        mask_config=MaskConfig(),
        source_size=(64, 64),
        crop_scale=(0.3, 1.0),
        horizontal_flip_probability=0.5,
        num_workers=0,
        pin_memory=False,
    )

    first = next(loader.iter_epoch(order, seed=991))
    second = next(loader.iter_epoch(order, seed=991))

    assert first.crop_theta is not None
    assert first.horizontal_flip is not None
    assert torch.equal(first.crop_theta, second.crop_theta)
    assert torch.equal(first.horizontal_flip, second.horizontal_flip)
    images = torch.rand(4, 3, 64, 64)
    transformed = apply_prepared_crop(
        images,
        theta=first.crop_theta,
        horizontal_flip=first.horizontal_flip,
        output_size=64,
    )
    assert transformed.shape == images.shape
