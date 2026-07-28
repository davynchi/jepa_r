from __future__ import annotations

import numpy as np
import pytest
import torch

from jepa.analysis.quality_manifests import (
    QualitySplitManifest,
    build_quality_manifest,
    iter_manifest_images,
    reconstruct_counterfactual_indices,
)
from jepa.configs.images.shapes3d import Shapes3DDataConfig
from jepa.data.images.shapes3d import (
    flat_index_batch,
    sample_shapes3d_counterfactual_factor_rows,
    sample_shapes3d_static_factor_row,
)


def test_static_sampler_is_deterministic_and_flat_index_matches() -> None:
    row, flat = sample_shapes3d_static_factor_row(123, 7)
    repeated, repeated_flat = sample_shapes3d_static_factor_row(123, 7)
    assert row.equal(repeated)
    assert flat == repeated_flat == int(flat_index_batch(row.numpy()[None, :])[0])


def test_counterfactual_sampler_matches_flat_indices_and_shape_contract() -> None:
    rows = sample_shapes3d_counterfactual_factor_rows(Shapes3DDataConfig(), num_pairs=8, seed=55)
    assert np.array_equal(rows.same_flat_indices_1, flat_index_batch(rows.same_factors_1.numpy()))
    assert np.array_equal(rows.same_flat_indices_2, flat_index_batch(rows.same_factors_2.numpy()))
    shape_column = 4
    assert np.array_equal(rows.same_factors_1[:, shape_column], rows.same_entity)
    assert np.array_equal(rows.same_factors_2[:, shape_column], rows.same_entity)


def test_manifest_is_deterministic_disjoint_balanced_and_tamper_evident(tmp_path) -> None:
    config = Shapes3DDataConfig(num_train_samples=8, num_val_samples=4, num_test_samples=4)
    sizes = {
        "unlabeled_fit_bank": 40,
        "unlabeled_metric_bank": 24,
        "classifier_train": 20,
        "classifier_validation": 8,
        "classifier_test": 8,
    }
    first = build_quality_manifest(
        config,
        data_config_hash="data",
        manifest_seed=7,
        transformation_model_seeds=(11, 12),
        transformation_ref_size=4,
        bank_sizes=sizes,
    )
    second = build_quality_manifest(
        config,
        data_config_hash="data",
        manifest_seed=7,
        transformation_model_seeds=(11, 12),
        transformation_ref_size=4,
        bank_sizes=sizes,
    )
    assert first.content_hash == second.content_hash
    roots = [set(first.banks[name]) for name in sizes]
    assert all(
        not left.intersection(right) for i, left in enumerate(roots) for right in roots[i + 1 :]
    )
    assert all(not set(first.excluded_indices).intersection(bank) for bank in roots)
    for name in ("classifier_train", "classifier_validation", "classifier_test"):
        shapes = [((index // 15) % 4) for index in first.banks[name]]
        assert [shapes.count(shape) for shape in range(4)] == [len(shapes) // 4] * 4
    path = tmp_path / "manifest.json"
    first.write(path)
    assert QualitySplitManifest.read(path).content_hash == first.content_hash
    payload = path.read_text().replace('"manifest_seed": 7', '"manifest_seed": 8')
    path.write_text(payload)
    with pytest.raises(ValueError, match="hash"):
        QualitySplitManifest.read(path)


def test_counterfactual_union_changes_with_model_seed() -> None:
    config = Shapes3DDataConfig()
    one = set(
        reconstruct_counterfactual_indices(config, transformation_model_seeds=(1,), ref_size=8)
    )
    two = set(
        reconstruct_counterfactual_indices(config, transformation_model_seeds=(2,), ref_size=8)
    )
    assert one
    assert two
    assert one != two


class _FakeSource:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def images_batch(self, indices: np.ndarray) -> np.ndarray:
        self.calls.append(len(indices))
        return np.broadcast_to(indices[:, None, None, None], (len(indices), 3, 2, 2)).astype(
            np.float32
        )


def test_streaming_preserves_order_and_bounds_reads() -> None:
    source = _FakeSource()
    chunks = list(iter_manifest_images(source, (9, 2, 7, 1, 8), batch_size=2))  # type: ignore[arg-type]
    assert source.calls == [2, 2, 1]
    assert tuple(index for indices, _ in chunks for index in indices) == (9, 2, 7, 1, 8)
    assert torch.cat([images for _, images in chunks])[:, 0, 0, 0].tolist() == [9, 2, 7, 1, 8]
