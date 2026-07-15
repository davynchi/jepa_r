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
