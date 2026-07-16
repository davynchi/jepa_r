# JEPA for Time Series

A compact PyTorch research codebase for studying which JEPA asymmetries prevent
representation collapse on time-series data.

The project replaces the original video/Transformer runtime with a small, explicit core.
V1 contains:

- a strictly affine encoder/predictor family;
- a matched nonlinear MLP family using smooth `tanh` activations;
- four executable stop-gradient/EMA policies;
- strict typed YAML configuration with CLI-style overrides;
- paired replicate seed derivation that is independent of SG/EMA axes;
- eagerly materialized linear/nonlinear state-space datasets with contraction dynamics,
  orthonormal observation maps, and burn-in;
- full-matrix covariance, effective-rank, latent-variance, vector-norm, and gradient-norm
  diagnostics with explicit null reasons;
- train-centered multi-output ridge probes in CPU `float64`, using an unregularized intercept
  and Cholesky solve;
- a deterministic trainer with wide epoch metrics, final-only test/probe evaluation,
  atomic run artifacts, and exact CPU checkpoint resume;
- a command-line interface for single runs and failure-isolated 8/16-cell sweeps, with
  raw CSV, aggregated CSV, and dependency-free SVG reports;
- tests for module identity, gradient flow, optimizer membership, EMA order, failure
  artifacts, and uninterrupted-versus-resumed equivalence.

Together, the two model families and four policies define the eight JEPA variants.

## Setup

```bash
uv sync
uv run pytest
uv run jepa --version
```

The reference experiment configuration is [configs/base.yaml](configs/base.yaml).

## Programmatic training

```python
from jepa import load_config, train_experiment

config = load_config("configs/base.yaml", overrides={"architecture": "linear", "sg": "on"})
result = train_experiment(config)
print(result.run_dir)
```

Every completed run contains `config.yaml`, `status.json`, `history.csv`,
`metrics.json`, and a mandatory final `checkpoint.pt`. A failed run keeps a structured
failure status. Exact resume is deliberately restricted to CPU checkpoints:

```python
result = train_experiment(config, resume_from="outputs/<run-id>/checkpoint.pt")
```

## Command line and sweeps

Run one of the eight model/policy variants:

```bash
uv run jepa train --config configs/base.yaml \
  --architecture linear --sg on --ema off
```

Run all eight variants for the configured dynamics, or all sixteen combinations across
both linear and nonlinear dynamics:

```bash
uv run jepa sweep --config configs/base.yaml --seeds 0 1 2
uv run jepa sweep --config configs/base.yaml --system-kind all --seeds 0
```

Each sweep gets its own directory containing per-cell run artifacts plus `summary.csv`,
`summary_by_variant.csv`, `summary.svg`, and `status.json`. A failed cell is recorded and
marked in the report without stopping its peers; the command returns a non-zero status
after all cells finish.

Arbitrary strict config overrides are available through repeated `--set KEY=VALUE`
arguments. For example:

```bash
uv run jepa train --config configs/base.yaml \
  --set training.epochs=20 --set training.learning_rate=0.0005
```

## Policy semantics

| SG | EMA | Target encoder | Target gradient | Target update |
|---|---|---|---|---|
| off | off | shared with context | enabled | shared Adam step |
| on | off | shared with context | detached | context-side Adam step |
| on | on | separate copy | disabled | EMA only |
| off | on | separate copy | enabled | Adam, then EMA pull |

The final row is deliberately non-standard and is labeled
`hybrid-gradient-ema-target` in code.

## Reproducibility contract

Synthetic systems, materialized samples, model initialization, and epoch ordering have
separate deterministic seed ownership. Sweep replicates keep system and sample seeds paired
across all SG/EMA policies; the model seed changes only with the architecture.

Bitwise checkpoint resume is supported only on CPU with the committed `uv.lock`, the same
OS and processor architecture, identical PyTorch/BLAS builds, one PyTorch thread, and
deterministic algorithms enabled. CUDA and MPS can be used for fresh training, but exact
resume across those devices is intentionally rejected. Cross-environment results should be
compared numerically rather than assumed to be bitwise identical.

## Running on GPU

All `configs/temporal_*.yaml` / `configs/spatial_semantics_*.yaml` set `training.device: auto`,
so copying the repository to a CUDA machine and running the same commands automatically picks
up the GPU (`jepa.training._resolve_device` prefers CUDA, then MPS, then CPU) -- no config
edits needed. To force a device explicitly instead, use the same strict override mechanism as
every other setting:

```bash
python scripts/train_temporal.py experiment=temporal_hierarchy_full training.device=cuda
python scripts/train_temporal_image.py experiment=temporal_hierarchy_image_full training.device=cuda
```

Two things to check before running on a CUDA server:

- **Determinism**: every run seeds via `torch.use_deterministic_algorithms(True)`. Some cuBLAS
  operations only support this with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) set in the
  environment; without it, training raises at the first affected op instead of silently being
  non-deterministic. Export it before launching:
  ```bash
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  ```
- **Matching torch build**: `pyproject.toml` pins `torch>=2.2` without a CUDA index, so `uv sync`
  may resolve a CPU-only wheel. On the GPU machine, install/sync against the CUDA build that
  matches its driver (e.g. via `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`
  or whatever CUDA version applies) before running `uv sync` again.
- Exact bitwise checkpoint **resume** is still CPU-only by design (see the reproducibility
  contract below) -- GPU runs are for fresh training/throughput, not exact resume.

For this specific experiment family (linear/small-MLP models on 32-d vectors, batch sizes in
the hundreds), GPU is unlikely to speed up the vector-world grid -- the model is too small for
kernel-launch and host/device transfer overhead to pay off. GPU matters much more for the
rendered-image world if the encoder or dataset scale grows (larger batches, a convolutional
encoder, bigger `image_size`).

## Current scope and limitations

- V1 uses controlled synthetic state-space time series only; a real-data adapter is a
  separate future benchmark.
- Context and target are fixed-size adjacent windows, and the training objective is MSE in
  representation space.
- Sweeps execute sequentially in one local process. There is no distributed scheduler.
- The affine family has no activation anywhere; the nonlinear family uses `tanh` hidden
  layers. Transformers, masking, image/video processing, and distributed V-JEPA code are
  deliberately absent.
- `SG off + EMA on` is an explicit experimental hybrid: the target receives an Adam update
  and is then pulled toward the context encoder by EMA. It should not be presented as the
  conventional JEPA recipe.
- Collapse diagnostics and probes are research measurements, not evidence of downstream
  usefulness on real data.

## Continuous integration

GitHub Actions runs linting, static type checks, tests, and package builds on Python
3.11–3.13. A separate five-minute job executes the complete tiny 16-cell sweep across both
dynamics and validates finite metrics, every per-run artifact, both CSV summaries, and the
SVG report. The same smoke contract can be run locally:

```bash
uv run jepa sweep --config configs/smoke.yaml \
  --system-kind all --seeds 0 --output-root smoke-outputs --overwrite
uv run python scripts/verify_smoke.py smoke-outputs
```

## Attribution

This repository started from Meta's V-JEPA research repository. The original license is
preserved. The new time-series runtime is a fresh implementation that retains only the
high-level JEPA/EMA training idea; video, image, masking, ViT, and distributed code are not
part of the new package.

## Entity/context temporal-persistence experiment

A second, self-contained experiment (`jepa.temporal_*`) tests whether **temporal
persistence causes JEPA to discover a low-dimensional, context-invariant entity
subspace**. It reuses the model families, the strict-config/override machinery, the
SG/EMA policy resolver, the atomic artifact writers, and the ridge-probe/effective-rank
diagnostics above; it adds nothing to `ExperimentConfig` and changes no existing config,
run, or test.

### Hypothesis

> When one hidden generative factor changes significantly more slowly than the rest and
> is useful for future prediction, JEPA should concentrate information about that factor
> in a low-dimensional latent subspace that is invariant to faster contextual changes.

This is stronger than "the representation contains entity information" (any linear probe
on a sufficiently large latent can decode almost anything). The experiment is designed to
tell the two apart:

1. **Decodability** — can a linear probe on the *full* latent `z` predict entity `E`?
   A "yes" here is necessary but not sufficient evidence for the hypothesis.
2. **A genuine subspace** — is there a *specific, low-dimensional* set of directions
   `P_E` such that (a) entity is decodable from `P_E z`, (b) entity is much harder to
   decode from the complementary directions `(I - P_E) z`, and (c) `P_E z` is invariant
   to context changes at fixed entity (Section 7.5's counterfactual `Q_E` metric)?

Only (2) is evidence for the hypothesis. Because JEPA's objective is invariant to any
rotation of the latent space, there is generally no single "entity neuron" to find —
individual coordinates are not identifiable, only subspaces are. **Failing to find one
axis that "is" the entity is expected, not a negative result;** `jepa.temporal_analysis`
never looks for one. It solves a regularized generalized eigenproblem
(`S_B v = λ (S_W + εI) v`, via `scipy.linalg.eigh`) between the between-entity and
within-entity scatter matrices of the frozen latent and takes the top-`k` eigenvectors as
the post-hoc entity subspace `P_E = V_k V_k^T`.

### Hidden generative process

Every trajectory has a slow entity `E_t ∈ {0,…,K-1}` (switches with probability `p_E` per
step) and a fast context `C_t ∈ R^{d_C}` (a stationary AR(1) process with correlation
`context_rho`), sampled **independently** so entity decoding cannot piggyback on
accidental entity/context correlation. Observations are `X_t = h(E_t, C_t) + noise` under
a fixed linear or `tanh`-MLP map `h`, generated once per dataset instance and shared
across train/validation/test splits (`jepa.temporal_data.generate_observation_system`).
**Entity and context labels are returned only for evaluation** — `jepa.temporal_training`
never passes them to an encoder's forward call; `test_entity_labels_are_not_passed_into_
the_training_forward_call` in `tests/test_temporal_training.py` asserts this by hooking
the encoder and checking every captured input is a float observation tensor of the right
width, not an entity/context tensor.

### Controls

- **Temporal-target control** (`training.target_pairing: temporal`, the standard
  condition): the predictor's target is the same trajectory's observation at `t+h`.
- **Shuffled-target control** (`training.target_pairing: shuffled`): the target
  observation instead comes from a *different* trajectory at the same time index. The
  per-timestep marginal observation distribution is unchanged, but temporal entity
  persistence is destroyed. If entity selectivity survives shuffling, persistence was
  never the cause — this control is what makes the hypothesis falsifiable.
- **Random-encoder baseline** (`model.kind: random`): an untrained, randomly initialized
  encoder run through the same analysis pipeline with zero optimizer steps.

### Explicit entity/context model

`model.kind: hierarchical` (`jepa.temporal_models.HierarchicalEncoderPredictor`) splits
the latent into `z = (z_E, z_C)` and adds two heads reading the *full* latent `z_t`: a
long-horizon head predicting the entity block of the target `h_E` steps ahead, and a
short-horizon head predicting the context block `h_C` steps ahead
(`jepa.temporal_training.hierarchical_loss`). The loss is
`λ_E L_E + λ_C L_C + λ_var L_var + λ_cross L_cross`, where `L_E`/`L_C` are normalized by
the batch's own `z_E`/`z_C` covariance trace, `L_var` is a per-dimension standard-deviation
floor, and `L_cross` penalizes the Frobenius norm of `Cov(z_E, z_C)`. Each of the horizon
split, the variance floor, and the cross-covariance penalty can be disabled independently
(`hierarchical.use_horizon_split`, `use_variance_reg`, `use_cross_cov_reg`) — see
`configs/temporal_hierarchy_ablation_*.yaml`. This model is evaluated both via its
explicit `z_E`/`z_C` blocks and via the same post-hoc subspace analysis as the standard
model.

### Configs

| File | Purpose |
|---|---|
| `configs/temporal_base.yaml` | Field-for-field defaults (`TemporalExperimentConfig`) |
| `configs/temporal_hierarchy_quick.yaml` | Small single-run smoke config |
| `configs/temporal_hierarchy_full.yaml` | Full-scale single-run config |
| `configs/temporal_hierarchy_explicit.yaml` | Hierarchical model, default regularizers |
| `configs/temporal_hierarchy_ablation_no_horizon_split.yaml` | Ablation: one shared horizon |
| `configs/temporal_hierarchy_ablation_no_variance_reg.yaml` | Ablation: variance floor off |
| `configs/temporal_hierarchy_ablation_no_cross_cov_reg.yaml` | Ablation: cross-cov penalty off |
| `configs/temporal_grids/quick.yaml` | Grid axes for the quick preset (Section 8) |
| `configs/temporal_grids/full.yaml` | Grid axes for the full preset (Section 8) |

### Commands

Smoke test (single run, seconds):

```bash
python scripts/train_temporal.py experiment=temporal_hierarchy_quick
```

Full single run:

```bash
python scripts/train_temporal.py experiment=temporal_hierarchy_full
```

Full experimental grid (5 seeds × {linear, nonlinear} × {linear, nonlinear obs} × 4
switch probabilities × 3 horizons × {temporal, shuffled}, plus the random baseline and the
hierarchical model at `p_E ∈ {0.01, 0.05, 0.20}`) — **never launched implicitly**, only via:

```bash
python scripts/run_temporal_grid.py --preset quick   # sanity-check the grid machinery
python scripts/run_temporal_grid.py --preset full
```

Evaluate one completed run — encodes train/validation/test, estimates the post-hoc entity
subspace from train+validation, fits probes, computes every Section 7 metric on test,
writes the per-run plots and `analysis/summary.json`:

```bash
python scripts/evaluate_temporal_hierarchy.py --run-dir outputs/temporal/<run-id>
```

Aggregate a grid across seeds/conditions (auto-runs evaluation on any un-analyzed run,
then writes `aggregate_summary.csv`/`.json`, `entity_selectivity_vs_timescale.png`,
`temporal_vs_shuffled.png`, and a `scientific_summary.md` answering Section 14's ten
questions):

```bash
python scripts/aggregate_temporal_hierarchy.py --root outputs/temporal_grid_full
```

### Expected outputs

Each run directory (`outputs/.../temporal_kind-<...>_seed-<...>_cfg-<hash>/`) contains
`config.yaml`, `status.json`, `history.json`, `metrics.json`, and `checkpoint.pt`
(weights + config + RNG state + the shared observation-system matrices, for exact
reproduction). Running the evaluate script adds `analysis/summary.json` and the per-run
PNGs (`generalized_eigenvalue_spectrum.png`, `probe_matrix.png`,
`counterfactual_distances.png`, `effective_rank_over_training.png`,
`prediction_loss_curves.png`, `latent_autocorrelation.png`). Aggregating a grid adds
`aggregate_summary.csv`/`.json`, `entity_selectivity_vs_timescale.png`,
`temporal_vs_shuffled.png`, and `scientific_summary.md` at the grid root.

### Implementation notes

- PNG plotting needs a raster backend, so `matplotlib` and `scipy` (for the symmetric
  generalized eigensolver) were added to `pyproject.toml`; nothing else in the base
  repository changed.
- The entity linear "classifier" is ridge regression onto one-hot labels plus `argmax`,
  reusing `jepa.metrics.fit_ridge_probe` rather than adding a new dependency (e.g.
  scikit-learn) for logistic regression.
- `evaluation.entity_subspace_dims` is swept in full on validation to select the reported
  default `k` (highest validation entity accuracy); the full curve over `k` is still
  reported, per the spec's instruction not to hide it behind the selected value.

## Rendered-image entity/context experiment

A third variant of the same experiment (`jepa.temporal_image_*`) replaces the vector
observation map with a small rasterizer: `E_t in {none, circle, square, triangle}`,
`C_t = (x, y, scale, rotation, r, g, b, visibility, background)`, `X_t = h(E_t, C_t)` is a
64x64x3 rendered frame. It answers the same scientific question as the vector world and
reuses almost everything from it.

### What's actually new vs. reused

- **New**: `jepa.temporal_image_config` (config), `jepa.temporal_image_data` (a
  dependency-free, vectorized NumPy rasterizer with 2x-supersample anti-aliasing;
  the entity/context temporal process; a static independent-sample dataset; block
  masking for the spatial-JEPA control; image counterfactual pairs).
- **Reused unmodified**: `jepa.temporal_data.make_temporal_pairs` /
  `make_hierarchical_pairs`, and `jepa.temporal_training`'s `_standard_epoch_loss`,
  `_hierarchical_epoch_loss`, `build_jepa_core`, `build_hierarchical_core`,
  `encode_split_standard`. None of these functions were touched or duplicated for
  images — `EntityContextImageTrajectoryDataset` simply exposes `.observations`
  (flattened frames), `.entities`, `.contexts`, and `config.trajectory_length` /
  `config.observation_dim` with the exact same meaning the vector dataset uses, so the
  pair builders and per-epoch training loops run against it untouched. The entire
  post-hoc analysis pipeline (`jepa.temporal_analysis`: scatter matrices, the
  generalized eigenproblem, probes, counterfactual invariance, autocorrelation) and
  every plot in `jepa.temporal_plots` are equally generic over latent tensors and are
  reused as-is; only one new auxiliary plot was added (`plot_latent_pca_scatter`).

### Hidden-factor design choices (documented assumptions)

- **Clipping vs. occlusion, as two independent factors.** Position `x, y` range over
  an extended `[-0.15, 1.15]` normalized window, so objects are naturally cropped by
  the 64x64 canvas edge when they drift near the border (real geometric clipping).
  `visibility` is a *separate* continuous opacity/alpha factor in `[0.3, 1.0]`
  blending the shape into the background (simulating partial occlusion/haze
  independent of position). This covers both aspects of Section 2.4.1 without
  conflating them into one variable.
- **Anti-aliasing**: frames are rasterized at 2x resolution with vectorized NumPy
  boolean masks (circle: distance test; square: rotated half-plane test; triangle:
  barycentric sign test against an equilateral triangle) and then average-pooled
  down — no image library dependency was added.
- **Static spatial-JEPA control (Section 2.4.3/2.4.6)**: `block_mask` splits an image
  into two *complementary* full-size views — `visible` (the image with one random
  rectangular block zeroed out) and `target` (only that block's original pixels,
  everything else zeroed). Both share the same `[C, H, W]` shape, so the existing
  single-global-latent `JEPACore` (`build_jepa_core`, `compute_loss`) is reused
  as-is: the context encoder reads the visible view, the target encoder (SG/EMA per
  the usual policy) reads the target-block view, and the predictor is trained to
  match the two — "predict the latent of the masked block from the visible part"
  at the level of one global vector latent rather than a patch-embedding sequence
  (a full per-patch I-JEPA-style architecture was out of scope for this iteration).

### Commands

```bash
# smoke test
python scripts/train_temporal_image.py experiment=temporal_hierarchy_image_quick
python scripts/train_temporal_image.py experiment=temporal_hierarchy_image_quick_shuffled

# full preset
python scripts/train_temporal_image.py experiment=temporal_hierarchy_image_full

# static spatial-JEPA semantics control (Section 2.4.7)
python scripts/train_temporal_image.py experiment=spatial_semantics_image_quick
python scripts/train_temporal_image.py experiment=spatial_semantics_image_quick_control

# evaluate any completed image run (same Section 7 metrics + an auxiliary PCA-by-entity plot)
python scripts/evaluate_temporal_image.py --run-dir outputs/temporal_image_quick/<run-id>

# preview grids: static samples, temporal trajectories, counterfactual pairs, masking
python scripts/visualize_temporal_image_dataset.py \
    --config configs/temporal_hierarchy_image_quick.yaml \
    --output-dir outputs/temporal_image_previews
```

### Configs

| File | Purpose |
|---|---|
| `configs/temporal_hierarchy_image_quick.yaml` | Smoke test, `temporal_image` |
| `configs/temporal_hierarchy_image_quick_shuffled.yaml` | Shuffled-target control |
| `configs/temporal_hierarchy_image_full.yaml` | Full-scale `temporal_image` |
| `configs/spatial_semantics_image_quick.yaml` | Static spatial-JEPA masking |
| `configs/spatial_semantics_image_quick_control.yaml` | Same, but excludes `none` so there is no object/no-object shortcut |

### Expected outputs

Same artifact set as the vector world (`config.yaml`, `status.json`, `history.json`,
`metrics.json`, `checkpoint.pt`), plus `analysis/summary.json` and PNGs
(`generalized_eigenvalue_spectrum.png`, `probe_matrix.png`,
`counterfactual_distances.png`, `effective_rank_over_training.png`,
`prediction_loss_curves.png`, `latent_pca_by_entity.png`) after running the evaluate
script. `visualize_temporal_image_dataset.py` writes `preview_*.png` grids into
`--output-dir` for a visual sanity check of the generator before spending a training
budget on it.

## Shapes3D-backed entity/context world

A third image variant (`jepa.temporal_shapes3d_*`) swaps the procedural rasterizer for an
**exact lookup** into DeepMind's [3D Shapes dataset](https://github.com/deepmind/3d-shapes)
(480,000 real rendered images, every combination of 6 factors: `floor_hue`, `wall_hue`,
`object_hue`, `scale`, `shape`, `orientation`). It is *additive*: nothing in
`jepa.temporal_image_*` changed, and both stay usable independently.

- **Entity** = `shape` (4 classes: cube, cylinder, sphere, capsule). **Context** = the other
  5 factors.
- Because every reachable image is a real row already in the dataset, the context process is
  a **discrete random walk over factor indices** (`context_step_probability` chance of moving
  `±1..context_step_max` per factor per step, clipped to the valid range) rather than the
  continuous AR(1) process used by the procedural renderer -- there is no snapping or
  interpolation, no approximation error, and indexing is exact (`jepa.temporal_shapes3d_data.flat_index`
  implements the dataset's own row-major formula).
- Reuses `jepa.temporal_image_data.block_mask` / `build_static_spatial_dataset` unmodified for
  the spatial-JEPA masking control, and (like the procedural image world) is duck-type
  compatible with `jepa.temporal_data.make_temporal_pairs` / `jepa.temporal_training`'s
  per-epoch training helpers, so training needed no new core logic -- only a new dataset and a
  thin `train_shapes3d_experiment` orchestration wrapper mirroring `temporal_image_training.py`.

### Setup (one-time download)

```bash
mkdir -p data
curl -o data/3dshapes.h5 https://storage.googleapis.com/3d-shapes/3dshapes.h5   # ~267 MB
```

`data/` is gitignored -- nothing here is ever committed. Two performance issues were found and
fixed while building this, in case the dataset is re-used elsewhere:

- **Lazy remote (HTTP range-request) access** via `h5py` + `fsspec` works but is impractically
  slow (~190s for a single image), because HDF5 metadata is scattered across the file and every
  seek is a separate round trip. Download the file once locally instead.
- **Even local random access is slow** on the file as published: `3dshapes.h5`'s `images`
  dataset is gzip-chunked as `(15000, 4, 4, 1)` -- each chunk spans 15,000 images but only a
  4x4x1 pixel patch, so reading one full image touches 768 chunks (~30-50s for a few thousand
  scattered images, even with an enlarged h5py chunk cache). `Shapes3DSource` works around this
  automatically: on first use it does one *sequential* full read of the file (chunks are hit
  cleanly in order, ~40-60s total) and caches the result as an uncompressed memory-mapped
  `data/3dshapes.npy` (~5.9 GB) next to it. Every later random access -- training, evaluation,
  counterfactual pairs, however scattered -- then hits that memmap instead of the `.h5` file and
  is effectively free (measured: 6,000 scattered images in ~1.7s after the cache is built, vs.
  ~99s against the raw file). The one-time conversion happens automatically the first time any
  script constructs a `Shapes3DSource`/dataset; nothing to run by hand.

### Commands

```bash
python scripts/train_temporal_shapes3d.py experiment=temporal_shapes3d_quick
python scripts/train_temporal_shapes3d.py experiment=temporal_shapes3d_full

python scripts/evaluate_temporal_shapes3d.py --run-dir outputs/temporal_shapes3d_quick/<run-id>

python scripts/visualize_temporal_shapes3d_dataset.py \
    --config configs/temporal_shapes3d_quick.yaml \
    --output-dir outputs/temporal_shapes3d_previews
```

`tests/test_temporal_shapes3d.py` auto-skips if `data/3dshapes.h5` is missing or incomplete
(checked by exact byte size), so the rest of the suite is unaffected on machines without the
dataset.
