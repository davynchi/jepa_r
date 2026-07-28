# Shapes3D quality–accuracy study

This pipeline tests whether checkpoint-level representation diagnostics predict a frozen linear
shape classifier's accuracy. The primary label-free panel discovers four orthogonal latent
subspaces from two fixed mask views. A supervised LDA partition is retained only as a control.

## Plan and run the grid

Planning is deterministic and does not launch training:

```bash
uv run python scripts/images/run_spatial_quality_grid.py plan \
  --config configs/images/shapes3d/quality.yaml \
  --grid-root outputs/ijepa_spatial/quality-grid
```

The manifest contains six pilot cells, 30 discovery cells, and 30 replication cells. Each
five-block phase crosses the same block seed with all six curricula. Local execution advances
one phase at a time:

```bash
uv run python scripts/images/run_spatial_quality_grid.py run-local \
  --grid-root outputs/ijepa_spatial/quality-grid --max-concurrency 2

uv run python scripts/images/run_spatial_quality_grid.py resume \
  --grid-root outputs/ijepa_spatial/quality-grid --max-concurrency 2
```

`resume` retries only failures explicitly marked retryable. `reconcile` imports status JSON from
an external scheduler and rejects a mismatched command hash or changes to a completed cell.

## Evaluate immutable checkpoints

```bash
uv run python scripts/analysis/evaluate_spatial_quality.py \
  --grid-root outputs/ijepa_spatial/quality-grid/runs \
  --quality-root outputs/ijepa_spatial/quality-grid/quality \
  --resume --device cuda
```

Cheap metrics run at every selected checkpoint. Q7, Q8, Q14, Q15, and Q19 run at the first,
final, and every fourth selected checkpoint; use `--skip-expensive` to record them as
`not_scheduled`. `--only-metrics` accepts a comma-separated list of registry names. Results are
stored transactionally in SQLite and exported to JSONL/CSV with a versioned metric dictionary.

The quantities discussed in the proposal enter the study as follows:

- `delta_i(x)` is retained per sample and band; its sum averaged over conditioning images is Q8.
- `H(V_i)` contributes rank-weighted band entropy to Q10; its checkpoint change is Q12.
- subspace similarity `S_i` and Grassmann distance `G_i` aggregate to Q5 and Q6; the distance per
  optimizer step is Q9, and concentration of realized band movement is Q13.
- the predictor Jacobian supplies cross-band energy Q7. Its normalized effective rank and
  thresholded nonzero density are Q19a and Q19b; raw numerical rank and exact `||J||_0` are not
  used because they are unstable without a threshold.
- projected loss-gradient energy gives Q14, while replay drift after the same virtual update used
  by Q8 gives Q15.

## Analyze association

```bash
uv run python scripts/analysis/analyze_quality_accuracy.py \
  --quality-root outputs/ijepa_spatial/quality-grid/quality
```

Discovery and replication are analyzed separately. The primary coefficient is partial Spearman
correlation after removing seed-block effects; uncertainty and p-values resample or permute whole
blocks. Benjamini–Hochberg correction is applied independently in each phase to eligible
label-free metrics. Pilot rows are never pooled into either confirmatory phase.

## Live-epoch uniform profiling

The live path measures the same registered Q metrics directly from the in-memory
model after every epoch. It does not load epoch checkpoints. It is intentionally
restricted to Shapes3D with uniform sampling:

```bash
uv run python scripts/images/train_ijepa_spatial.py \
  --run-name uniform_live_quality_seed7111473167050986645 \
  --dataset shapes3d \
  --device mps \
  --epochs 100 \
  --batch-size 128 \
  --weighting-method uniform \
  --eval-every-epochs 100 \
  --checkpoint-every-epochs 100 \
  --online-quality \
  --online-quality-every-epochs 1 \
  --online-quality-feature-batch-size 128
```

The run writes live observations under
`outputs/ijepa_spatial/<run-name>/online-quality/`. Logical identifiers such as
`live_epoch_0001` preserve the transactional metric-cell contract, but do not
name checkpoint files.

The evaluator restores module modes plus CPU/MPS RNG state after every diagnostic.
All expensive metrics run at every live observation. Temporal metrics compare
strictly adjacent observed epochs; epoch 1 and an epoch following a degenerate
partition receive `no_previous_epoch`.

Timing uses `time.perf_counter_ns()` with device synchronization before and after
each measured accelerator stage. Shared prerequisites are reported separately:

- Jacobian construction is shared by Q7, Q19a, and Q19b.
- The virtual-update/refit stage is shared by Q8 and Q15.
- Q5 and Q6 share one subspace-comparison calculation.
- Q19a and Q19b share one Jacobian-simplicity pass.

`timings.jsonl` distinguishes `stage` time from `metric_formula` time; shared
work is not counted multiple times.

After training, build the epoch-level correlation and timing report:

```bash
uv run python scripts/analysis/analyze_epoch_quality.py \
  --quality-root \
  outputs/ijepa_spatial/uniform_live_quality_seed7111473167050986645/online-quality
```

The report includes Pearson and Spearman correlations with classification
accuracy and held-out JEPA loss, first-difference correlations, scatter pages,
correlation summaries, timing-by-epoch plots, median/p95 timing tables, and a
Russian interpretation file. These are descriptive correlations: the 100 epochs
belong to one autocorrelated training trajectory and are not treated as 100
independent samples.
