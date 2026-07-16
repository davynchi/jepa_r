# JEPA for Time Series

A compact PyTorch research codebase for studying which JEPA asymmetries prevent
representation collapse on time-series data. It retains the high-level JEPA/EMA idea while
removing video processing and Transformers.

The repository provides:

- strictly affine and smooth `tanh` encoder/predictor families;
- four explicit stop-gradient/EMA policies;
- deterministic synthetic state-space data and an immutable Binance Spot 15-minute benchmark;
- causal `future_window` and transformer-free `local-patch masked JEPA` objectives;
- collapse diagnostics, gradient metrics, ridge probes, atomic artifacts, and exact CPU resume;
- failure-isolated, objective-aware sweeps with CSV and SVG reports.

For each objective, two architectures × two SG values × two EMA values define eight cells.

## Setup

```bash
uv sync
uv run pytest
uv run jepa --version
```

The synthetic reference configuration is [configs/base.yaml](configs/base.yaml). The Binance
test preset is [configs/binance_spot_15m_test.yaml](configs/binance_spot_15m_test.yaml).

## Binance 15-minute benchmark

The preset uses `BTCUSDT`, `ETHUSDT`, and `BNBUSDT` from 2021 through 2024. Preparation
downloads official monthly ZIP and checksum files, aligns the three symbols by UTC timestamp,
creates 18 features, fits normalization on train only, and atomically publishes content-addressed
`[N, 64, 18]` NPZ arrays.

Prepare once and run all 16 cells:

```bash
uv run jepa prepare-data --config configs/binance_spot_15m_test.yaml

uv run jepa sweep \
  --config configs/binance_spot_15m_test.yaml \
  --objectives all \
  --seeds 0
```

Preparation is resumable through the verified raw cache. `--offline` forbids network access;
`--force` redownloads source files; the two flags are mutually exclusive. A repeated sweep
reuses its immutable dataset and completed cells.

Training progress is written to `stderr`, so the final JSON in `stdout` remains safe for scripts:

```text
Cell 3/16 | masked_patches | nonlinear | SG on | EMA off
  Epoch 7/10 | train MSE 0.0312 | elapsed 01:24 | ETA 00:36
  Complete | elapsed 02:01
```

Each invocation is stored in a Moscow-time directory rather than a hash-named directory:

```text
outputs/binance_spot_15m/
└── 2026-07-16_14-32-08_MSK/
    ├── learning_curves.svg
    ├── summary.svg
    ├── summary.csv
    └── runs/
        └── <descriptive-run-id>/
            ├── history.svg
            ├── history.csv
            └── ...
```

`history.svg` plots objective MSE, effective rank, latent standard deviation, and gradient norms
across evaluation epochs. `learning_curves.svg` compares validation MSE across all sweep cells,
with a separate panel for each objective. The Binance preset evaluates every epoch, so its plots
contain all ten epochs. Semantic hashes remain inside metadata and checkpoints for safe resume.

The splits are:

- train: `[2021-01-01, 2024-01-01)`;
- validation: `[2024-01-02, 2024-07-01)`;
- test: `[2024-07-02, 2025-01-01)`.

The one-day gaps are embargoes. Training windows use stride 8; validation and test use stride
32. Dataset preparation is outside the runtime target. The preset targets less than 30 minutes
per full cell, while the sequential 16-cell sweep has no hard timeout.

## Objectives

`future_window` preserves the causal adjacent-window task. In the Binance preset it flattens
the first 32 steps and predicts the representation of the flattened next 32 steps.

`masked_patches` splits all 64 steps into eight temporal patches of eight steps. Exactly four
sorted patches are targets. The context encoder physically receives only the other four patches.
The vectorized predictor receives a zero-filled latent grid, an eight-slot visibility mask, and
fixed 16-dimensional sin/cos target positions, producing `[B, 4, latent_dim]` in one call.

This is deliberately called **local-patch masked JEPA**. It is not an exact reproduction of
Transformer-based TS-JEPA or V-JEPA. Cross-objective losses are not a direct ranking: the masked
task is non-causal and uses different targets and predictor capacity. Forecast R² and downstream
probes are descriptive comparisons only.

Select one objective for a run:

```bash
uv run jepa train --config configs/binance_spot_15m_test.yaml \
  --objective masked_patches --architecture nonlinear --sg on --ema on
```

Old YAML without `objective` remains `future_window`; its config hash, run ID, and ordinary
eight-cell synthetic sweep remain unchanged.

## Artifacts and resume

Every completed run contains `config.yaml`, `status.json`, `history.csv`, `history.svg`,
`metrics.json`, and `checkpoint.pt`. Artifacts record the objective, dataset fingerprint,
replicate seed, unique parameter counts, and wall-clock time. Failed cells keep structured
failure labels and do not stop their peers.

```python
from jepa import load_config, train_experiment

config = load_config("configs/base.yaml", overrides={"architecture": "linear", "sg": "on"})
result = train_experiment(config)
print(result.run_dir)
```

Exact resume is restricted to CPU checkpoints from an equivalent environment:

```python
result = train_experiment(config, resume_from="outputs/<moscow-timestamp>/checkpoint.pt")
```

Schema-v1 synthetic checkpoints remain readable. Objective or dataset-fingerprint mismatches
are rejected. Mask schedules derive from replicate seed, dataset fingerprint, split, sample
index, and epoch (or `eval`), so resume does not depend on DataLoader order or global RNG state.

## SG/EMA policy semantics

| SG | EMA | Target encoder | Target gradient | Target update |
|---|---|---|---|---|
| off | off | shared with context | enabled | shared Adam step |
| on | off | shared with context | detached | context-side Adam step |
| on | on | separate copy | disabled | EMA only |
| off | on | separate copy | enabled | Adam, then EMA pull |

The last row is an explicit experimental hybrid named `hybrid-gradient-ema-target`.

## Scope

Arbitrary CSV/NPZ adapters, Transformers, feature-tube masks, trading/PnL, live APIs, direction
classifiers, distributed sweeps, and hard timeouts are outside this slice. CUDA and MPS are
available for fresh runs, but bitwise resume is CPU-only. Collapse diagnostics and probes are
research measurements, not evidence of profitable or production-ready representations.

## Attribution

The repository began from Meta's V-JEPA research repository and preserves its original license.
The time-series runtime is a fresh implementation; video, image, ViT, and distributed V-JEPA
code are not part of this package.
