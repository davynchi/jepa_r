#!/usr/bin/env python3
"""Evaluate Shapes3D quality metrics from immutable spatial checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jepa.analysis.quality_autograd import (  # noqa: E402
    context_latent_gradients,
    fused_q8_q15,
    predictor_jacobian,
)
from jepa.analysis.quality_features import (  # noqa: E402
    CheckpointFeatureBundle,
    LiveSpatialAdapter,
    SpatialCheckpointAdapter,
    extract_feature_bundle,
    feature_metadata,
)
from jepa.analysis.quality_manifests import (  # noqa: E402
    QualitySplitManifest,
    build_quality_manifest,
    iter_manifest_images,
)
from jepa.analysis.quality_metrics import (  # noqa: E402
    METRIC_SPEC_BY_NAME,
    NullReason,
    q1_cross_covariance,
    q2_projector_interaction,
    q3_projector_overlap,
    q4_partition_incompleteness,
    q5_q6_subspace_stability,
    q7_cross_jacobian_energy,
    q9_subspace_velocity,
    q10_weighted_shape_entropy,
    q11_cross_subspace_gaussian_mi,
    q12_entropy_change,
    q13_surprise_locality,
    q14_gradient_locality,
    q16_entity_consistency,
    q17_transformation_residual,
    q18_perturbation_concentration,
    q19_jacobian_simplicity,
    q20_reconstruction_nmse,
)
from jepa.analysis.quality_store import QualityStore  # noqa: E402
from jepa.analysis.quality_timing import QualityTimingRecorder  # noqa: E402
from jepa.analysis.spatial_decomposition import (  # noqa: E402
    EigenBoundaryTieError,
    Panel,
    SubspacePartition,
    fit_label_free_partition,
    fit_supervised_partition,
)
from jepa.analysis.subspace import classifier_accuracy, fit_entity_classifier  # noqa: E402
from jepa.configs.base import derive_seed  # noqa: E402
from jepa.configs.images.shapes3d import (  # noqa: E402
    Shapes3DExperimentConfig,
    shapes3d_config_from_dict,
    shapes3d_config_identity_hash,
)
from jepa.data.images.shapes3d import Shapes3DSource  # noqa: E402
from jepa.models.patches import patchify  # noqa: E402
from jepa.training.images.ijepa_spatial import (  # noqa: E402
    MaskConfig,
    sample_masks,
    spatial_ijepa_per_sample_loss,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", default=None)
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--checkpoint-pattern", default="epoch_*.pt")
    parser.add_argument("--quality-root", default=None)
    parser.add_argument("--manifest-seed", type=int, default=58_031)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-expensive", action="store_true")
    parser.add_argument("--only-metrics", default=None)
    return parser.parse_args()


def _discover_runs(arguments: argparse.Namespace) -> list[Path]:
    runs = [Path(path).expanduser().resolve() for path in arguments.run_dir]
    if arguments.grid_root:
        root = Path(arguments.grid_root).expanduser().resolve()
        runs.extend(path.parent for path in root.rglob("config.json"))
    unique = sorted(set(runs))
    if not unique:
        raise ValueError("provide --run-dir or --grid-root")
    return unique


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _checkpoints(run: Path, pattern: str) -> list[Path]:
    checkpoints = sorted((run / "checkpoints").glob(pattern))
    if not checkpoints:
        raise ValueError(f"no checkpoints matching {pattern!r} in {run}")
    return checkpoints


def _config(run: Path) -> tuple[dict, Shapes3DExperimentConfig]:
    payload = json.loads((run / "config.json").read_text())
    if "config" not in payload:
        raise ValueError(f"{run} is not a Shapes3D run")
    config = shapes3d_config_from_dict(payload["config"])
    return payload, config


def _quality_root(arguments: argparse.Namespace, runs: list[Path]) -> Path:
    if arguments.quality_root:
        return Path(arguments.quality_root).expanduser().resolve()
    base = Path(arguments.grid_root).resolve() if arguments.grid_root else runs[0]
    return base / "quality"


def _inventory(arguments: argparse.Namespace, runs: list[Path], output: Path) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        for checkpoint in _checkpoints(run, arguments.checkpoint_pattern):
            adapter = SpatialCheckpointAdapter.load(run, checkpoint)
            rows.append(
                {
                    "run_id": run.name,
                    "checkpoint": str(checkpoint),
                    "checkpoint_hash": adapter.checkpoint_hash,
                    "epoch": adapter.epoch,
                    "global_step": adapter.global_step,
                    "model_seed": adapter.model_seed,
                    "curriculum": adapter.curriculum,
                    "training_uses_shape_metadata": adapter.training_uses_shape_metadata,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoint_inventory.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    return rows


def _manifest(
    arguments: argparse.Namespace, runs: list[Path], output: Path
) -> QualitySplitManifest:
    path = output / "quality_split_manifest.json"
    first_payload, first_config = _config(runs[0])
    del first_payload
    identity = shapes3d_config_identity_hash(first_config)
    for run in runs[1:]:
        _, config = _config(run)
        if shapes3d_config_identity_hash(config) != identity:
            raise ValueError("grid_data_config_mismatch")
    if path.exists():
        manifest = QualitySplitManifest.read(path)
        if manifest.data_config_hash != identity:
            raise ValueError("grid_data_config_mismatch")
        return manifest
    transformation_seeds: list[int] = []
    ref_size = 1024
    for run in runs:
        payload, _ = _config(run)
        spatial = payload["spatial"]
        weighting = spatial["weighting"]
        if (
            weighting["method"] == "coord"
            and weighting["coordinate_importance"] == "transformation"
        ):
            transformation_seeds.append(int(spatial["seed"]))
            ref_size = int(weighting["ref_size"])
    manifest = build_quality_manifest(
        first_config.data,
        data_config_hash=identity,
        manifest_seed=arguments.manifest_seed,
        transformation_model_seeds=transformation_seeds,
        transformation_ref_size=ref_size,
    )
    manifest.write(path)
    return manifest


def _bundle(
    adapter: SpatialCheckpointAdapter | LiveSpatialAdapter,
    source: Shapes3DSource,
    manifest: QualitySplitManifest,
    output: Path,
    bank: str,
    view: str,
    batch_size: int,
    *,
    cache_features: bool = True,
    timing: QualityTimingRecorder | None = None,
) -> CheckpointFeatureBundle:
    indices = manifest.banks[bank]
    mask_seed = manifest.named_seeds.get(f"label_free_{view}_seed") if view != "full" else None
    metadata = feature_metadata(
        adapter,
        manifest_hash=manifest.content_hash,
        bank=bank,
        view=view,
        indices=indices,
        mask_seed=mask_seed,
    )
    key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = output / "caches" / f"{key}.npz"
    if cache_features and path.exists():
        try:
            return CheckpointFeatureBundle.load(path, expected_metadata=metadata)
        except ValueError:
            path.unlink()
    timer = (
        timing.stage(f"features:{bank}:{view}")
        if timing is not None
        else nullcontext()
    )
    with timer:
        bundle = extract_feature_bundle(
            adapter,
            source,
            indices,
            manifest_hash=manifest.content_hash,
            bank=bank,
            view=view,
            batch_size=batch_size,
            mask_seed=mask_seed,
        )
    if cache_features:
        bundle.save(output / "caches")
    return bundle


def _labels(indices) -> torch.Tensor:
    return torch.tensor([((int(index) // 15) % 4) for index in indices], dtype=torch.long)


def _image_targets(source: Shapes3DSource, indices, batch_size: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for _, images in iter_manifest_images(source, indices, batch_size=batch_size):
        rows.append(F.adaptive_avg_pool2d(images, (16, 16)).flatten(1))
    return torch.cat(rows)


@torch.no_grad()
def _heldout_loss(
    adapter: SpatialCheckpointAdapter | LiveSpatialAdapter,
    source: Shapes3DSource,
    manifest: QualitySplitManifest,
    *,
    batch_size: int,
) -> float:
    device = next(adapter.core.context_encoder.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(manifest.manifest_seed, "quality-heldout-loss-masks")
    )
    total = 0.0
    count = 0
    for _, images in iter_manifest_images(
        source, manifest.banks["classifier_test"], batch_size=batch_size
    ):
        patches = patchify(images, 8).to(device)
        context_masks, target_masks = sample_masks(8, 8, MaskConfig(), generator)
        losses = torch.zeros(patches.shape[0], device=device)
        for context_mask in context_masks:
            for target_mask in target_masks:
                losses += spatial_ijepa_per_sample_loss(
                    adapter.core,
                    patches,
                    context_mask.to(device),
                    target_mask.to(device),
                )
        losses /= len(context_masks) * len(target_masks)
        total += float(losses.sum().detach().cpu())
        count += patches.shape[0]
    if count == 0:
        raise ValueError("held-out loss bank is empty")
    return total / count


def _write_checkpoint_loss(
    output: Path,
    *,
    adapter: SpatialCheckpointAdapter | LiveSpatialAdapter,
    loss: float,
    sample_count: int,
) -> None:
    path = output / "checkpoint_losses.jsonl"
    by_key: dict[tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            by_key[(row["run_id"], row["checkpoint_id"])] = row
    row = {
        "run_id": adapter.run_dir.name,
        "checkpoint_id": adapter.checkpoint_path.name,
        "epoch": adapter.epoch,
        "global_step": adapter.global_step,
        "curriculum": adapter.curriculum,
        "loss": loss,
        "sample_count": sample_count,
    }
    by_key[(row["run_id"], row["checkpoint_id"])] = row
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, sort_keys=True) + "\n"
            for _, value in sorted(by_key.items())
        )
    )
    temporary.replace(path)


def _patch_bank(
    source: Shapes3DSource, indices, batch_size: int, device: torch.device
) -> torch.Tensor:
    rows = [
        patchify(images, 8)
        for _, images in iter_manifest_images(source, indices, batch_size=batch_size)
    ]
    if not rows:
        raise ValueError("expensive metric bank is empty")
    return torch.cat(rows).to(device)


def _fixed_masks(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    context_masks, target_masks = sample_masks(8, 8, MaskConfig(), generator)
    return context_masks[0].to(device), target_masks[0].to(device)


def _mean_metric(values) -> tuple[float | None, str | None]:
    valid = [item.value for item in values if item.value is not None]
    if not valid:
        return None, next((item.reason for item in values if item.reason), "no_valid_samples")
    return sum(valid) / len(valid), None


def _timed_value(
    timing: QualityTimingRecorder | None,
    name: str,
    function: Callable[[], object],
    *,
    metric: bool = False,
):
    timer = timing.stage(name, metric=metric) if timing is not None else nullcontext()
    with timer:
        return function()


def _expensive_metrics(
    adapter: SpatialCheckpointAdapter | LiveSpatialAdapter,
    source: Shapes3DSource,
    manifest: QualitySplitManifest,
    partition: SubspacePartition,
    *,
    batch_size: int,
    timing: QualityTimingRecorder | None = None,
) -> dict[str, tuple[float | None, str | None, dict]]:
    device = next(adapter.core.context_encoder.parameters()).device
    core = adapter.core
    active_timing = timing or QualityTimingRecorder(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = active_timing.stages.get("jacobian_construction", 0.0)
    with active_timing.stage("jacobian_construction"):
        jacobian_patches = _patch_bank(
            source, manifest.banks["jacobian_bank"], batch_size, device
        )
        q7_context, q7_targets = _fixed_masks(
            manifest.named_seeds["q7_target_mask_seed"], device
        )
        with torch.no_grad():
            summaries = core.context_encoder(jacobian_patches[:, q7_context]).mean(dim=1)
        jacobians = [
            predictor_jacobian(core, summary, int(target_index))
            for summary in summaries
            for target_index in q7_targets
        ]
    jacobian_seconds = active_timing.stages["jacobian_construction"] - before
    with active_timing.stage("q7_cross_jacobian_energy", metric=True):
        q7 = _mean_metric(
            [q7_cross_jacobian_energy(value, partition) for value in jacobians]
        )
    with active_timing.stage("q19a_q19b_shared_formula"):
        q19_values = [q19_jacobian_simplicity(value) for value in jacobians]
    q19a = _mean_metric([value[0] for value in q19_values])
    q19b = _mean_metric([value[1] for value in q19_values])

    before = active_timing.stages.get("gradient_construction", 0.0)
    with active_timing.stage("gradient_construction"):
        gradient_patches = _patch_bank(
            source, manifest.banks["gradient_bank"], batch_size, device
        )
        q14_context, q14_targets = _fixed_masks(
            manifest.named_seeds["q14_loss_mask_seed"], device
        )
        gradients = context_latent_gradients(
            core, gradient_patches, q14_context, q14_targets
        )
    gradient_seconds = active_timing.stages["gradient_construction"] - before
    with active_timing.stage("q14_gradient_locality", metric=True):
        q14, shares, maximum = q14_gradient_locality(gradients, partition)

    before = active_timing.stages.get("virtual_update_q8_q15", 0.0)
    with active_timing.stage("virtual_update_q8_q15"):
        conditioning = _patch_bank(
            source, manifest.banks["virtual_conditioning_bank"], batch_size, device
        )
        refit = _patch_bank(
            source, manifest.banks["decomposition_refit_bank"], batch_size, device
        )
        replay = _patch_bank(source, manifest.banks["replay_bank"], batch_size, device)
        q8_context, q8_targets = _fixed_masks(
            manifest.named_seeds["q8_loss_mask_seed"], device
        )
        refit_a_mask, _ = _fixed_masks(
            manifest.named_seeds["label_free_mask_a_seed"], device
        )
        refit_b_mask, _ = _fixed_masks(
            manifest.named_seeds["label_free_mask_b_seed"], device
        )
        fused_error = None
        try:
            fused = fused_q8_q15(
                core,
                conditioning,
                q8_context,
                q8_targets,
                refit[:, refit_a_mask],
                refit[:, refit_b_mask],
                replay,
                checkpoint_id=adapter.checkpoint_hash,
                eta=1e-3,
                microbatch_size=batch_size,
            )
        except EigenBoundaryTieError as error:
            fused = None
            fused_error = error
    virtual_seconds = active_timing.stages["virtual_update_q8_q15"] - before
    peak_device_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    results = {
        "q7_cross_jacobian_energy": (
            *q7,
            {
                "jacobian_count": len(jacobians),
                "stage_seconds": jacobian_seconds,
                "peak_device_bytes": peak_device_bytes,
            },
        ),
        "q14_gradient_locality": (
            q14.value,
            q14.reason,
            {
                "gradient_energy_shares": shares,
                "q14_max": maximum,
                "stage_seconds": gradient_seconds,
                "peak_device_bytes": peak_device_bytes,
            },
        ),
        "q19a_jacobian_effective_rank": (
            *q19a,
            {"jacobian_count": len(jacobians), "stage_seconds": jacobian_seconds},
        ),
        "q19b_jacobian_density_tau_1em3": (
            *q19b,
            {"jacobian_count": len(jacobians), "stage_seconds": jacobian_seconds},
        ),
    }
    if fused is None:
        diagnostics = {
            "partition_error": str(fused_error),
            "stage_seconds": virtual_seconds,
            "peak_device_bytes": peak_device_bytes,
        }
        results["q8_virtual_sensitivity_magnitude"] = (
            None,
            NullReason.DEGENERATE_PARTITION.value,
            diagnostics,
        )
        results["q15_virtual_interference"] = (
            None,
            NullReason.DEGENERATE_PARTITION.value,
            diagnostics,
        )
    else:
        with active_timing.stage("q8_virtual_sensitivity_magnitude", metric=True):
            q8_value = float(fused.q8)
        with active_timing.stage("q15_virtual_interference", metric=True):
            q15_value = float(fused.q15)
        results["q8_virtual_sensitivity_magnitude"] = (
            q8_value,
            None,
            {
                "gradient_calls": fused.gradient_calls,
                "per_band_mean_delta": tuple(
                    float(value) for value in fused.per_sample_deltas.mean(dim=0)
                ),
                "stage_seconds": virtual_seconds,
                "peak_device_bytes": peak_device_bytes,
            },
        )
        results["q15_virtual_interference"] = (
            q15_value,
            None,
            {
                "gradient_calls": fused.gradient_calls,
                "stage_seconds": virtual_seconds,
                "peak_device_bytes": peak_device_bytes,
            },
        )
    return results


def _cell_key(run_id: str, checkpoint: str, panel: Panel, metric: str, manifest: str) -> str:
    raw = "|".join((run_id, checkpoint, panel.value, metric, manifest, "1"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _store_metric(
    store: QualityStore,
    *,
    run_id: str,
    checkpoint: str,
    panel: Panel,
    metric: str,
    value: float | None,
    null_reason: str | None,
    manifest: QualitySplitManifest,
    context: dict,
    retry_failed: bool,
) -> None:
    key = _cell_key(run_id, checkpoint, panel, metric, manifest.content_hash)
    store.ensure_cell(
        {
            "cell_key": key,
            "run_id": run_id,
            "checkpoint_id": checkpoint,
            "panel": panel.value,
            "metric_name": metric,
            "metric_version": METRIC_SPEC_BY_NAME[metric].version,
        }
    )
    if not store.claim(key, retry_failed=retry_failed):
        return
    store.complete(key, value=value, null_reason=null_reason, diagnostics=context)


LABEL_FREE_METRICS = {
    spec.name for spec in METRIC_SPEC_BY_NAME.values() if spec.name != "q16_entity_consistency"
}
SUPERVISED_METRICS = {
    "q1_cross_covariance",
    "q2_projector_interaction",
    "q3_projector_overlap",
    "q4_partition_incompleteness",
    "q5_subspace_similarity",
    "q6_subspace_distance",
    "q9_subspace_velocity",
    "q10_weighted_shape_entropy",
    "q11_cross_subspace_gaussian_mi",
    "q12_absolute_entropy_change",
    "q13_realized_surprise_locality",
    "q16_entity_consistency",
}


def _store_degenerate_partition(
    store: QualityStore,
    *,
    adapter: SpatialCheckpointAdapter | LiveSpatialAdapter,
    panel: Panel,
    manifest: QualitySplitManifest,
    context: dict,
    enabled_metrics: set[str],
    error: EigenBoundaryTieError,
    retry_failed: bool,
) -> None:
    expected = LABEL_FREE_METRICS if panel is Panel.LABEL_FREE else SUPERVISED_METRICS
    for metric in sorted(expected.intersection(enabled_metrics)):
        _store_metric(
            store,
            run_id=adapter.run_dir.name,
            checkpoint=adapter.checkpoint_path.name,
            panel=panel,
            metric=metric,
            value=None,
            null_reason=NullReason.DEGENERATE_PARTITION.value,
            manifest=manifest,
            context={**context, "partition_error": str(error)},
            retry_failed=retry_failed,
        )


def _evaluate_checkpoint(
    adapter: SpatialCheckpointAdapter | LiveSpatialAdapter,
    source: Shapes3DSource,
    manifest: QualitySplitManifest,
    output: Path,
    store: QualityStore,
    *,
    batch_size: int,
    previous: dict[Panel, tuple[SubspacePartition, float, int]] | None,
    enabled_metrics: set[str],
    run_expensive: bool,
    retry_failed: bool,
    cache_features: bool = True,
    timing: QualityTimingRecorder | None = None,
    temporal_null_reason: str = "no_previous_checkpoint",
) -> dict[Panel, tuple[SubspacePartition, float, int]]:
    run_id = adapter.run_dir.name
    checkpoint_id = adapter.checkpoint_path.name
    phase_prefix = run_id.split("_", 1)[0]
    phase = phase_prefix if phase_prefix in {"pilot", "discovery", "replication"} else "unspecified"
    context = {
        "epoch": adapter.epoch,
        "global_step": adapter.global_step,
        "model_seed": adapter.model_seed,
        "seed_block": adapter.model_seed,
        "curriculum": adapter.curriculum,
        "training_uses_shape_metadata": adapter.training_uses_shape_metadata,
        "phase": phase,
    }
    bundle_options = {"cache_features": cache_features, "timing": timing}
    fit_a = _bundle(
        adapter,
        source,
        manifest,
        output,
        "unlabeled_fit_bank",
        "mask_a",
        batch_size,
        **bundle_options,
    )
    fit_b = _bundle(
        adapter,
        source,
        manifest,
        output,
        "unlabeled_fit_bank",
        "mask_b",
        batch_size,
        **bundle_options,
    )
    metric_a = _bundle(
        adapter,
        source,
        manifest,
        output,
        "metric_bank",
        "mask_a",
        batch_size,
        **bundle_options,
    )
    metric_b = _bundle(
        adapter,
        source,
        manifest,
        output,
        "metric_bank",
        "mask_b",
        batch_size,
        **bundle_options,
    )
    metric_full = _bundle(
        adapter,
        source,
        manifest,
        output,
        "metric_bank",
        "full",
        batch_size,
        **bundle_options,
    )
    label_free_error = None
    try:
        label_free = _timed_value(
            timing,
            "factorization_label_free",
            lambda: fit_label_free_partition(
                fit_a.features, fit_b.features, checkpoint_id=checkpoint_id
            ),
        )
    except EigenBoundaryTieError as error:
        label_free = None
        label_free_error = error

    classifier_train = _bundle(
        adapter,
        source,
        manifest,
        output,
        "classifier_train",
        "full",
        batch_size,
        **bundle_options,
    )
    classifier_validation = _bundle(
        adapter,
        source,
        manifest,
        output,
        "classifier_validation",
        "full",
        batch_size,
        **bundle_options,
    )
    classifier_validation_a = _bundle(
        adapter,
        source,
        manifest,
        output,
        "classifier_validation",
        "mask_a",
        batch_size,
        **bundle_options,
    )
    classifier_validation_b = _bundle(
        adapter,
        source,
        manifest,
        output,
        "classifier_validation",
        "mask_b",
        batch_size,
        **bundle_options,
    )
    classifier_test = _bundle(
        adapter,
        source,
        manifest,
        output,
        "classifier_test",
        "full",
        batch_size,
        **bundle_options,
    )
    train_labels = _labels(manifest.banks["classifier_train"])
    test_labels = _labels(manifest.banks["classifier_test"])
    supervised_error = None
    try:
        supervised = _timed_value(
            timing,
            "factorization_supervised",
            lambda: fit_supervised_partition(
                classifier_train.features,
                train_labels,
                checkpoint_id=checkpoint_id,
            ),
        )
    except EigenBoundaryTieError as error:
        supervised = None
        supervised_error = error
    classifier = _timed_value(
        timing,
        "classifier_fit",
        lambda: fit_entity_classifier(
            classifier_train.features, train_labels, 4, ridge=1e-6
        ),
    )
    with (
        timing.stage("classifier_test")
        if timing is not None
        else nullcontext()
    ):
        accuracy = classifier_accuracy(classifier, classifier_test.features, test_labels)
        predictions = classifier.probe.predict(classifier_test.features).argmax(dim=1)
    per_class = [
        float((predictions[test_labels == shape] == shape).double().mean()) for shape in range(4)
    ]
    store.upsert_accuracy(
        run_id, checkpoint_id, accuracy=accuracy, balanced_accuracy=sum(per_class) / 4
    )

    outputs: dict[
        Panel, tuple[SubspacePartition | None, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {
        Panel.LABEL_FREE: (label_free, metric_full.features, metric_a.features, metric_b.features),
        Panel.SUPERVISED_LDA: (
            supervised,
            classifier_validation.features,
            classifier_validation_a.features,
            classifier_validation_b.features,
        ),
    }
    current: dict[Panel, tuple[SubspacePartition, float, int]] = {}
    for panel, (partition, features, view_a, view_b) in outputs.items():
        partition_error = (
            label_free_error if panel is Panel.LABEL_FREE else supervised_error
        )
        if partition is None:
            if partition_error is None:
                raise AssertionError("missing partition error")
            _store_degenerate_partition(
                store,
                adapter=adapter,
                panel=panel,
                manifest=manifest,
                context=context,
                enabled_metrics=enabled_metrics,
                error=partition_error,
                retry_failed=retry_failed,
            )
            continue
        metric_diagnostics: dict[str, dict] = {}
        panel_context = dict(context)
        scalar_values: dict[str, tuple[float | None, str | None]] = {}
        for metric_name, function in (
            (
                "q1_cross_covariance",
                lambda features=features, partition=partition: q1_cross_covariance(
                    features, partition
                ),
            ),
            (
                "q2_projector_interaction",
                lambda features=features, partition=partition: q2_projector_interaction(
                    features, partition
                ),
            ),
            (
                "q3_projector_overlap",
                lambda partition=partition: q3_projector_overlap(partition),
            ),
            (
                "q4_partition_incompleteness",
                lambda partition=partition: q4_partition_incompleteness(partition),
            ),
        ):
            scalar_values[metric_name] = (
                _timed_value(timing, metric_name, function, metric=True),
                None,
            )
        entropy, _ = _timed_value(
            timing,
            "q10_weighted_shape_entropy",
            lambda features=features, partition=partition: q10_weighted_shape_entropy(
                features, partition
            ),
            metric=True,
        )
        mi, clamped = _timed_value(
            timing,
            "q11_cross_subspace_gaussian_mi",
            lambda features=features, partition=partition: q11_cross_subspace_gaussian_mi(
                features, partition
            ),
            metric=True,
        )
        scalar_values["q10_weighted_shape_entropy"] = (entropy.value, entropy.reason)
        scalar_values["q11_cross_subspace_gaussian_mi"] = (mi.value, mi.reason)
        if panel is Panel.SUPERVISED_LDA:
            entity = partition.bands[0]
            entity_a = view_a.to(dtype=entity.basis.dtype) @ entity.basis
            entity_b = view_b.to(dtype=entity.basis.dtype) @ entity.basis
            q16 = _timed_value(
                timing,
                "q16_entity_consistency",
                lambda entity_a=entity_a, entity_b=entity_b: q16_entity_consistency(
                    entity_a, entity_b
                ),
                metric=True,
            )
            scalar_values["q16_entity_consistency"] = (q16.value, q16.reason)
        if panel is Panel.LABEL_FREE:
            forward = _timed_value(
                timing,
                "q17_mask_transformation_residual",
                lambda: q17_transformation_residual(
                    fit_a.features,
                    fit_b.features,
                    metric_a.features,
                    metric_b.features,
                ),
                metric=True,
            )
            backward = _timed_value(
                timing,
                "q17_mask_transformation_residual",
                lambda: q17_transformation_residual(
                    fit_b.features,
                    fit_a.features,
                    metric_b.features,
                    metric_a.features,
                ),
                metric=True,
            )
            q17_value = (
                None
                if forward.value is None or backward.value is None
                else (forward.value + backward.value) / 2
            )
            scalar_values["q17_mask_transformation_residual"] = (
                q17_value,
                forward.reason or backward.reason,
            )
            q18, _, _ = _timed_value(
                timing,
                "q18_mask_perturbation_concentration",
                lambda view_a=view_a, view_b=view_b, partition=partition: (
                    q18_perturbation_concentration(view_a, view_b, partition)
                ),
                metric=True,
            )
            scalar_values["q18_mask_perturbation_concentration"] = (q18.value, q18.reason)
            fit_full = _bundle(
                adapter,
                source,
                manifest,
                output,
                "unlabeled_fit_bank",
                "full",
                batch_size,
                **bundle_options,
            )
            train_targets = _timed_value(
                timing,
                "reconstruction_targets",
                lambda: _image_targets(
                    source, manifest.banks["unlabeled_fit_bank"], batch_size
                ),
            )
            metric_targets = _timed_value(
                timing,
                "reconstruction_targets",
                lambda: _image_targets(source, manifest.banks["metric_bank"], batch_size),
            )
            q20 = _timed_value(
                timing,
                "q20_reconstruction_nmse",
                lambda fit_full=fit_full,
                train_targets=train_targets,
                metric_full=metric_full,
                metric_targets=metric_targets: q20_reconstruction_nmse(
                    fit_full.features,
                    train_targets,
                    metric_full.features,
                    metric_targets,
                ),
                metric=True,
            )
            scalar_values["q20_reconstruction_nmse"] = (q20.value, q20.reason)
            expensive_names = {
                "q7_cross_jacobian_energy",
                "q8_virtual_sensitivity_magnitude",
                "q14_gradient_locality",
                "q15_virtual_interference",
                "q19a_jacobian_effective_rank",
                "q19b_jacobian_density_tau_1em3",
            }
            requested_expensive = expensive_names.intersection(enabled_metrics)
            if requested_expensive and run_expensive:
                for metric, (value, reason, diagnostics) in _expensive_metrics(
                    adapter,
                    source,
                    manifest,
                    partition,
                    batch_size=batch_size,
                    timing=timing,
                ).items():
                    scalar_values[metric] = (value, reason)
                    metric_diagnostics[metric] = diagnostics
            else:
                reason = "not_scheduled"
                for metric in requested_expensive:
                    scalar_values[metric] = (None, reason)
        if previous and panel in previous and entropy.value is not None:
            prior_partition, prior_entropy, prior_step = previous[panel]
            q5, q6, distances = _timed_value(
                timing,
                "q5_q6_shared_formula",
                lambda partition=partition, prior_partition=prior_partition: (
                    q5_q6_subspace_stability(partition, prior_partition)
                ),
            )
            scalar_values["q5_subspace_similarity"] = (q5, None)
            scalar_values["q6_subspace_distance"] = (q6, None)
            scalar_values["q9_subspace_velocity"] = (
                _timed_value(
                    timing,
                    "q9_subspace_velocity",
                    lambda q6=q6, prior_step=prior_step: q9_subspace_velocity(
                        q6, adapter.global_step, prior_step
                    ),
                    metric=True,
                ),
                None,
            )
            q12, signed = _timed_value(
                timing,
                "q12_absolute_entropy_change",
                lambda entropy=entropy, prior_entropy=prior_entropy: q12_entropy_change(
                    float(entropy.value), prior_entropy
                ),
                metric=True,
            )
            scalar_values["q12_absolute_entropy_change"] = (q12, None)
            q13 = _timed_value(
                timing,
                "q13_realized_surprise_locality",
                lambda distances=distances: q13_surprise_locality(distances),
                metric=True,
            )
            scalar_values["q13_realized_surprise_locality"] = (q13.value, q13.reason)
            panel_context["q12_signed"] = signed
        else:
            for metric in (
                "q5_subspace_similarity",
                "q6_subspace_distance",
                "q9_subspace_velocity",
                "q12_absolute_entropy_change",
                "q13_realized_surprise_locality",
            ):
                scalar_values[metric] = (None, temporal_null_reason)
        for metric, (value, reason) in scalar_values.items():
            if metric not in enabled_metrics:
                continue
            _store_metric(
                store,
                run_id=run_id,
                checkpoint=checkpoint_id,
                panel=panel,
                metric=metric,
                value=value,
                null_reason=reason,
                manifest=manifest,
                context={
                    **panel_context,
                    "clamped_roundoff": clamped,
                    **metric_diagnostics.get(metric, {}),
                },
                retry_failed=retry_failed,
            )
        if entropy.value is not None:
            current[panel] = (partition, entropy.value, adapter.global_step)
    return current


def main() -> None:
    arguments = _args()
    runs = _discover_runs(arguments)
    output = _quality_root(arguments, runs)
    inventory = _inventory(arguments, runs, output)
    if arguments.inventory:
        print(f"inventory={len(inventory)} path={output / 'checkpoint_inventory.json'}")
        return
    manifest = _manifest(arguments, runs, output)
    (output / "metric_dictionary.json").write_text(
        json.dumps(
            [
                {
                    "name": spec.name,
                    "q_number": spec.q_number,
                    "version": spec.version,
                    "direction": spec.direction.value,
                    "role": spec.role.value,
                    "family": spec.family,
                    "cost_tier": spec.cost_tier.value,
                    "primary_eligible": spec.primary_eligible,
                }
                for spec in METRIC_SPEC_BY_NAME.values()
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    enabled_metrics = set(METRIC_SPEC_BY_NAME)
    if arguments.only_metrics:
        enabled_metrics = {
            name.strip() for name in arguments.only_metrics.split(",") if name.strip()
        }
        unknown = enabled_metrics.difference(METRIC_SPEC_BY_NAME)
        if unknown:
            raise ValueError(f"unregistered metrics: {sorted(unknown)}")
    _, config = _config(runs[0])
    source = Shapes3DSource(config.data.h5_path)
    store = QualityStore(output / "quality.sqlite")
    store.recover_interrupted()
    for run in runs:
        previous = None
        checkpoints = _checkpoints(run, arguments.checkpoint_pattern)
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            adapter = SpatialCheckpointAdapter.load(run, checkpoint)
            device = _resolve_device(arguments.device)
            adapter.core.context_encoder.to(device)
            adapter.core.predictor.to(device)
            adapter.core.target_encoder.to(device)
            heldout_loss = _heldout_loss(
                adapter,
                source,
                manifest,
                batch_size=arguments.feature_batch_size,
            )
            _write_checkpoint_loss(
                output,
                adapter=adapter,
                loss=heldout_loss,
                sample_count=len(manifest.banks["classifier_test"]),
            )
            previous = _evaluate_checkpoint(
                adapter,
                source,
                manifest,
                output,
                store,
                batch_size=arguments.feature_batch_size,
                previous=previous,
                enabled_metrics=enabled_metrics,
                run_expensive=(
                    not arguments.skip_expensive
                    and (
                        checkpoint_index == 0
                        or checkpoint_index == len(checkpoints) - 1
                        or checkpoint_index % 4 == 0
                    )
                ),
                retry_failed=arguments.resume,
            )
    store.export(output)
    store.close()
    print(f"quality_root={output}")


if __name__ == "__main__":
    main()
