# JEPA for Synthetic Time Series — Reviewed Implementation Plan

Status: APPROVED
Reviewed: 2026-07-15
Source design: `/Users/davynci/.gstack/projects/facebookresearch-jepa/davynci-main-design-20260715-153734.md`

## Outcome

Replace the existing video/Transformer-oriented runtime with a compact PyTorch research package for two JEPA model families:

- linear: affine encoder and affine predictor only;
- nonlinear: matched MLP encoder and predictor with `tanh` after every hidden layer.

The package must run all combinations of architecture, stop-gradient (SG), and exponential moving average (EMA): `2 × 2 × 2 = 8` variants for each synthetic dynamics kind and paired replicate.

MSE is the only training objective. Collapse is measured and reported, not prevented by an auxiliary loss.

## In scope

- Linear and nonlinear synthetic state-space dynamics.
- Linear and `tanh` JEPA model families.
- Four explicit SG/EMA optimization policies.
- Deterministic paired experiment replicates.
- Train/validation metrics, final test metrics, collapse diagnostics, and linear probes.
- Checkpointing and deterministic CPU resume in a pinned environment.
- Eight- and sixteen-cell sweeps, partial-failure isolation, CSV/JSON artifacts, and a static SVG report.
- `uv` packaging, tests, source distribution through GitHub, and GitHub Actions.

## NOT in scope

- Video, image, masking, Vision Transformer, and Transformer code.
- Real-world time-series adapters in v1; this is tracked in `TODOS.md`.
- Alternate losses, optimizers, schedulers, gradient clipping, AMP, distributed training, SLURM, Hydra, or Lightning.
- Plugin registries or abstract dataset frameworks before a second concrete dataset exists.
- Visualization dashboard or notebook framework; v1 emits one static sweep SVG.
- CUDA/MPS resume guarantees.
- PyPI publication.

## What already exists

- `app/vjepa/train.py` demonstrates the conceptual online encoder → predictor → target encoder order and post-optimizer EMA update. Reuse the verified idea, not the video-specific implementation.
- The existing license and upstream attribution remain.
- Existing video datasets, masks, ViTs, evaluation apps, old configs, `setup.py`, and video-heavy requirements do not fit the new runtime and will be removed.
- `AGENTS.md` already records the required `uv` workflow and Codex skill routing.

## Compact repository structure

```text
.
├── AGENTS.md
├── LICENSE
├── README.md
├── TODOS.md
├── pyproject.toml
├── uv.lock
├── configs/
│   └── base.yaml
├── src/jepa/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   ├── reporting.py
│   └── training.py
├── tests/
│   ├── test_cli_sweep_reporting.py
│   ├── test_config_data.py
│   ├── test_metrics_artifacts.py
│   ├── test_models_policies.py
│   ├── test_training_resume.py
│   └── test_repository_hygiene.py
└── .github/workflows/ci.yml
```

Use functions and typed dataclasses for SG/EMA behavior. Do not introduce `GradientPolicy` or `EMAPolicy` class hierarchies in v1.

## Runtime data flow

```text
resolved config
     │
     ├── derive paired replicate seeds
     │        ├── system A, C
     │        ├── train/validation/test samples
     │        └── model initialization + epoch order
     │
     ├── materialize split windows and latent states
     │
     ├── build encoder, predictor, and optional target encoder
     │
     └── epoch loop
              ├── context → encoder → predictor ──────┐
              ├── target  → target encoder → [SG] ───┤ MSE
              ├── backward → Adam step → [EMA step]  │
              ├── scheduled train/validation metrics │
              └── mandatory final checkpoint ─────────┘
                         │
                         ├── final train-fitted probes
                         ├── first and only test evaluation
                         └── CSV/JSON/checkpoint artifacts

sweep: run all cells independently → aggregate completed cells → summary.svg
```

Add a shortened version of the SG/EMA state transition diagram as an inline ASCII comment beside the policy-selection function in `training.py`. Other modules are straightforward enough not to need inline diagrams.

## Configuration contract

`configs/base.yaml` uses nested `data`, `model`, `training`, `evaluation`, and `output` sections. CLI overrides YAML; YAML overrides typed defaults. Reject unknown keys and conflicting aliases.

Adam and MSE are fixed implementation properties and are not enum fields.

```yaml
data:
  system_kind: linear          # linear | nonlinear
  num_samples: 4096
  validation_samples: 1024
  test_samples: 1024
  sequence_length: 64
  window_size: 16
  burn_in_steps: 64
  latent_state_dim: 4
  observation_dim: 4           # must be >= latent_state_dim
  process_noise: 0.01
  observation_noise: 0.01
  transition_norm: 0.9         # operator norm, strictly between 0 and 1
  system_seed: 123
  train_sample_seed: 1000
  validation_sample_seed: 2000
  test_sample_seed: 3000

model:
  architecture: linear         # linear | nonlinear
  latent_dim: 16
  hidden_dim: 64
  hidden_layers: 1             # nonlinear only, must be >= 1

training:
  seed: 0
  learning_rate: 0.001
  betas: [0.9, 0.999]
  epsilon: 1.0e-8
  amsgrad: false
  batch_size: 128
  epochs: 100
  shuffle: true
  device: auto                 # cpu | cuda | mps | auto
  evaluation_every_epochs: 1
  checkpoint_every_epochs: 10
  stop_gradient: true
  ema:
    enabled: true
    decay: 0.99

evaluation:
  covariance_epsilon: 1.0e-8
  collapse_std_threshold: 1.0e-3
  collapse_rank_threshold: 0.1
  probe_ridge: 1.0e-6

output:
  root: outputs
  overwrite: false
```

Validate positive dimensions and intervals, `sequence_length >= 2 * window_size`, non-negative noise, `0 < transition_norm < 1`, `0 <= ema.decay < 1`, and `observation_dim >= latent_state_dim`.

## Randomness and reproducibility

Use stable SHA-256-based 64-bit derivation; never use Python's process-randomized `hash()`.

For a single `train` command, explicit config seeds are authoritative. For `sweep --seeds 0 1 2`, each value is a paired replicate seed that derives:

- the system seed;
- train, validation, and test sample seeds;
- architecture-specific model initialization seed;
- epoch shuffle seeds.

All eight policy cells inside one dynamics/replicate pair use identical generated data and identical initialization/order within an architecture. Different replicates use different systems, samples, initialization, and order. Aggregate summaries therefore estimate variation across complete synthetic replicates rather than one fixed system.

Exact replay is guaranteed only with the committed `uv.lock`, identical OS/architecture, PyTorch and BLAS builds, `num_workers=0`, `torch.set_num_threads(1)`, and deterministic PyTorch algorithms. Save `A` and `C` in run artifacts. Across other supported environments, compare samples and metrics with documented `rtol=1e-5`, `atol=1e-7`; do not claim bitwise equality.

## Synthetic dynamics

For each paired replicate:

```text
raw_A ~ Normal(0, 1)
A = transition_norm * raw_A / operator_norm(raw_A)
raw_C ~ Normal(0, 1), shape [observation_dim, latent_state_dim]
Q, R = thin_qr(raw_C)
C = Q * canonical_sign(diag(R))    # orthonormal columns, fixed QR sign

z[0] ~ Normal(0, I)
z[t+1] = transition(A @ z[t]) + process_noise
x[t] = C @ z[t] + observation_noise
```

- `transition=identity` for linear dynamics; `transition=tanh` for nonlinear dynamics.
- Canonicalize each QR column so the corresponding diagonal entry of `R` is non-negative; treat an exact zero sign as positive.
- Generate `burn_in_steps + sequence_length` observations and discard burn-in before selecting windows.
- Each dataset index has its own deterministic trajectory RNG.
- Draw exactly one deterministic valid start index for that item.
- Return adjacent context and target windows plus the last latent state represented by each window.
- Materialize only returned windows and latent states once when each split is constructed. Do not retain full trajectories.
- Train, validation, and test share `A,C` inside a replicate but never share sample RNGs or items.

## Model families

Both families flatten `[B, window_size, observation_dim]` to `[B, window_size * observation_dim]` and emit `[B, latent_dim]`.

- Linear encoder: one `nn.Linear(input_dim, latent_dim)`.
- Linear predictor: one `nn.Linear(latent_dim, latent_dim)`.
- Nonlinear encoder: `hidden_layers` repetitions of `Linear → Tanh`, followed by `Linear(hidden_dim, latent_dim)`.
- Nonlinear predictor: the same pattern from `latent_dim` to `latent_dim`.

No normalization, dropout, residual path, output normalization, or implicit anti-collapse feature is allowed.

## SG/EMA truth table

| SG | EMA | Target module | Target gradient | Optimizer membership | Post-Adam update |
|---|---|---|---|---|---|
| off | off | Same encoder object | Yes | Shared encoder | None |
| on | off | Same encoder object | Detached latent | Shared encoder | None |
| on | on | Separate copied encoder | No | Excluded | `target = d*target + (1-d)*context` |
| off | on | Separate copied encoder | Yes | Included | Adam target step, then EMA pull |

The fourth cell is intentionally non-standard. Label it `hybrid-gradient-ema-target` in config metadata, summaries, plots, and README.

Exact step order:

```text
optimizer.zero_grad(set_to_none=True)
context_latent = context_encoder(context)
target_latent = target_encoder(target)
prediction = predictor(context_latent)
loss = mse(prediction, detach(target_latent) if SG else target_latent)
loss.backward()
optimizer.step()
if EMA: update target parameters from context parameters
```

In the hybrid cell, retain Adam first/second moments after the EMA pull and checkpoint them unchanged. Resetting moments would define another policy.

Use PyTorch's small built-ins where they fit, but keep the hybrid post-optimizer update explicit because its semantics differ from a conventional frozen EMA model.

## Evaluation protocol

Scheduled evaluation during training computes only train/validation MSE, branch representation diagnostics, norms, and gradient aggregates. The test split is not touched.

After the final optimizer step:

1. Perform mandatory final train/validation evaluation even if the epoch is not divisible by the configured interval.
2. Write a mandatory final checkpoint under the same rule.
3. Freeze encoders.
4. Fit probes only on train representations.
5. Evaluate probes on validation/test.
6. Compute test MSE and representation diagnostics for the first and only time.

### Epoch aggregation

- MSE: total squared error divided by total scalar target elements.
- Covariance/effective-rank metrics: concatenate the full split representation matrix first.
- Vector norms: sample-weighted mean of per-sample L2 norms.
- Parameter and latent gradient norms: sample-weighted mean of per-batch norms.
- Missing gradients: JSON `null` plus a stable reason such as `stop_gradient` or `shared_target_module`.

### Collapse metrics

For centered `H ∈ R^(N×D)`, use the symmetrized covariance `Hc.T @ Hc / (N-1)`. Clamp negative numerical eigenvalues and variances to zero. If `N < 2`, values are non-finite, or the eigenvalue sum is below epsilon, emit `null`/collapsed with an explicit reason. Otherwise normalize eigenvalues by their exact positive sum and calculate:

- mean latent standard deviation;
- effective rank and normalized effective rank;
- largest-eigenvalue fraction;
- context, target, and prediction mean vector norms.

Collapse thresholds are reporting rules only.

### Ridge probes

Run on CPU in `float64`.

- Center features and targets using train means only.
- Do not variance-normalize.
- Solve `(XcᵀXc + αI)W = XcᵀYc` by Cholesky.
- Recover the unregularized intercept as `y_mean - x_mean @ W`.
- Reuse train means and fitted parameters unchanged for validation/test.
- Report multi-output aggregate `R²`; constant targets return `null` with reason `constant_target`.

Fit one probe from context representation to context latent state, one from target representation to target latent state, and one forecast head from context representation to the flattened raw target window.

## Commands and sweep behavior

```bash
uv sync
uv run jepa train --config configs/base.yaml --architecture linear --sg on --ema off
uv run jepa sweep --config configs/base.yaml --seeds 0 1 2
uv run jepa sweep --config configs/base.yaml --system-kind all --seeds 0
uv run pytest
uv build
```

For one configured dynamics kind, sweep produces eight cells per replicate. `--system-kind all` produces sixteen.

Each cell runs in an isolated exception boundary. A failed cell writes `state=failed`; the sweep continues, excludes it from completed aggregates, records completed/failed counts, builds all possible outputs, and returns a non-zero exit code after the matrix finishes.

## Run identity, artifacts, and resume

Run IDs retain readable axes and append the first eight hex characters of the immutable initial resolved-config hash:

```text
dynamics-linear_model-linear_sg-on_ema-off_seed-0_cfg-a1b2c3d4
```

The initial identity hash includes experiment semantics, including the initially requested epochs, but excludes `output.root` and `output.overwrite`. YAML key order cannot affect it. A resumed run keeps this immutable run ID even when its epoch limit increases. Artifacts then retain both `initial_config_hash` and the updated `current_config_hash`, plus a resume-history entry, so the directory lineage is explicit rather than pretending the original hash describes the extended request.

Each run writes:

- `config.yaml`: current resolved config, initial/current hashes, resume history, schemas, package/git/environment versions, derived seeds, and saved `A,C`.
- `status.json`: atomic `incomplete | running | failed | complete` state transitions, timestamps, epoch, and structured failure.
- `history.csv`: one unique `(run_id, epoch, split)` row with explicitly prefixed context/target/prediction fields.
- `metrics.json`: final/best validation, final test/probe metrics, collapse flags/reasons, and artifact paths.
- `checkpoint.pt`: schema, epoch, all model and optimizer states, original resolved config, and Python/NumPy/Torch RNG states.

The wide history schema includes scalar MSE and elapsed time; `context_*` and `target_*` latent standard deviation, effective rank, normalized rank, top-eigen fraction, representation norm, latent-gradient norm; `prediction_norm_mean`; parameter-gradient norms for context encoder, predictor, and a distinct target encoder; and context-target parameter distance where applicable.

The sweep writes `summary.csv`, `summary_by_variant.csv`, and `summary.svg`. The SVG contains panels for validation MSE, context normalized effective rank, and target normalized effective rank; all variants have stable labels and failed runs are marked.

### Resume state machine

Resume always continues the original run directory containing the checkpoint. It bypasses the ordinary non-empty-directory refusal only after verifying run ID, artifact schema, original config, and checkpoint epoch.

- Only `training.epochs` may increase.
- Refuse resume when status is `complete`.
- Truncate/reject history inconsistent with the checkpoint; never duplicate `(epoch, split)` rows.
- CPU resume is supported only in the pinned deterministic environment.
- Mandatory comparison test: uninterrupted `N+M` epochs equals `N` then resume to `N+M` within tolerance, including parameters, metrics, and history.

Fresh runs refuse any existing non-empty run directory unless `--overwrite` is explicit. Overwrite applies only to the exact hashed run directory, never to `output.root` as a whole.

## Distribution

- GitHub source distribution only.
- `uv sync` installs from a clean clone; `uv.lock` is committed.
- `uv build` must produce wheel and source archive.
- No PyPI workflow.
- GitHub Actions runs lint, type checks, unit/integration tests, and build on supported Python versions with CPU execution.
- A Python 3.11 smoke job runs one tiny 16-cell sweep across both dynamics, verifies finite metrics and all artifacts including SVG, and has a five-minute timeout.

Reference implementation behavior against official documentation for [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [DataLoader generators](https://docs.pytorch.org/docs/stable/data.html), and [uv packaged projects](https://docs.astral.sh/uv/concepts/projects/init/).

## Failure modes

| Codepath | Realistic failure | Planned test | Handling and user signal |
|---|---|---|---|
| Config | Unknown key or impossible dimensions | Validation table tests | CLI exits non-zero with field path and reason |
| Data | Degenerate raw `A`/QR input or non-finite trajectory | Seeded edge tests | Construction fails explicitly; no partial run |
| Model factory | Nonlinear model built without `tanh` | Structural tests | Config/factory error before training |
| SG/EMA policy | Wrong module identity, detach, optimizer set, or update order | Four-cell truth-table test | Test blocks ship; metadata identifies hybrid |
| Training | Non-finite loss | Injected failure test | Run becomes `failed`; error is visible in status |
| Metrics | Too few samples, zero spectrum, or non-finite values | Analytic edge tests | `null` with reason, never silent NaN |
| Probe | Singular/constant target | Analytic ridge tests | Ridge remains solvable; constant target reports reason |
| Artifacts | Interrupted write | Atomic-write tests | Previous valid file remains; status is not complete |
| Resume | Mismatched config/history/checkpoint | Full resume integration test | Refuse before mutation with precise message |
| Sweep | One cell raises | Partial-failure integration test | Remaining cells run; final exit non-zero |
| Reporting | Missing/failed cell data | SVG integration test | Failed cell is marked; report generation continues |

No planned path has a silent failure without both handling and a test.

## Test coverage map

```text
CODE PATHS                                               EXPERIMENT FLOWS
[+] config.py                                             [+] Single run
 ├── defaults/YAML/CLI precedence [★★★ PLANNED]            ├── linear/nonlinear model [★★★]
 ├── strict validation/errors [★★★]                       ├── four SG/EMA cells [★★★]
 └── canonical identity hash [★★★]                        ├── scheduled + final eval [★★★]
[+] data.py                                               └── artifacts + final test [★★★]
 ├── linear/nonlinear dynamics [★★★]                     [+] Resume
 ├── paired replicates/split isolation [★★★]              ├── uninterrupted equivalence [★★★]
 └── contraction/QR/burn-in/materialization [★★★]         ├── no duplicate history [★★★]
[+] models.py                                             └── mismatch/complete refusal [★★★]
 ├── affine-only family [★★★]                            [+] Sweep
 └── hidden_layers × tanh structure [★★★]                 ├── 8/16 unique cells [★★★]
[+] training.py                                           ├── isolated failure [★★★]
 ├── four-cell gradient truth table [★★★]                 ├── aggregate counts/exit code [★★★]
 ├── hybrid Adam→EMA order [★★★]                          └── summary.svg [★★★]
 └── aggregation/non-finite failure [★★★]
[+] metrics.py/reporting.py
 ├── covariance/rank boundaries [★★★]
 ├── float64 ridge contract [★★★]
 └── wide artifact schema/branch mapping [★★★]

PLANNED COVERAGE: all identified branches
IMPLEMENTED COVERAGE: all identified v1 branches have executable coverage
Legend: ★★★ = behavior + boundary + error-path requirement
```

Detailed executable cases live in `docs/test-plan.md`.

## Worktree parallelization

| Step | Modules touched | Depends on |
|---|---|---|
| Foundation | project metadata, package skeleton | — |
| Config + data | `src/jepa/config.py`, `src/jepa/data.py` | Foundation |
| Models + policies | `src/jepa/models.py`, policy portion of `training.py` | Foundation |
| Metrics | `src/jepa/metrics.py` | Foundation |
| Trainer + artifacts + resume | `src/jepa/training.py` | Config + data; models + policies; metrics |
| CLI + sweep + reporting | `src/jepa/cli.py`, `src/jepa/reporting.py` | Trainer + artifacts + resume |
| Docs + CI + hygiene | documentation and workflow | All runtime paths stable |

Parallel lanes:

- Lane A: Foundation → Config + data.
- Lane B: after Foundation, Models + policies.
- Lane C: after Foundation, Metrics.
- Merge A+B+C, then implement Trainer + artifacts + resume.
- Finish with CLI + sweep + reporting, then Docs + CI + hygiene.

`training.py` is shared by the policy and trainer steps. If separate worktrees are used, land Models + policies before beginning the trainer to avoid a merge conflict in that module.

## Implementation Tasks

Synthesized from this review. Execute in order unless the dependency table allows parallel work.

- [x] **T1 (P1, human: ~2h / Codex: ~20min)** — Foundation — Replace the video runtime with the compact `uv` package skeleton.
  - Surfaced by: Scope review D1 and existing-code audit.
  - Files: `pyproject.toml`, `uv.lock`, `src/jepa/`, old video runtime/config paths.
  - Verify: `uv sync`, package import, repository-hygiene test.
- [x] **T2 (P1, human: ~4h / Codex: ~35min)** — Config/data — Implement strict config, paired seed derivation, controlled dynamics, burn-in, and eager split materialization.
  - Surfaced by: D18, D19, and reproducibility review D17.
  - Files: `src/jepa/config.py`, `src/jepa/data.py`, `configs/base.yaml`.
  - Verify: `uv run pytest tests/test_config_data.py`.
- [x] **T3 (P1, human: ~4h / Codex: ~35min)** — Models/policies — Implement affine and tanh families plus the executable four-cell SG/EMA truth table.
  - Surfaced by: Core design, D6, and policy semantics review.
  - Files: `src/jepa/models.py`, `src/jepa/training.py`.
  - Verify: `uv run pytest tests/test_models_policies.py`.
- [x] **T4 (P1, human: ~6h / Codex: ~50min)** — Trainer/artifacts — Implement aggregation, atomic artifacts, final evaluation/checkpoint, run identity, and safe same-directory resume.
  - Surfaced by: D2, D3, D7-D9, D12-D14, D20.
  - Files: `src/jepa/training.py`, `src/jepa/config.py`.
  - Verify: `uv run pytest tests/test_training_resume.py tests/test_metrics_artifacts.py`.
- [x] **T5 (P1, human: ~4h / Codex: ~35min)** — Metrics — Implement collapse diagnostics and the exact float64 ridge-probe contract without test leakage.
  - Surfaced by: D16, D20, D21.
  - Files: `src/jepa/metrics.py`.
  - Verify: analytic metric/probe tests in `tests/test_metrics_artifacts.py`.
- [x] **T6 (P2, human: ~5h / Codex: ~45min)** — Experiments — Implement CLI, failure-isolated 8/16-cell sweeps, aggregate tables, and static SVG reporting.
  - Surfaced by: D4, D8, D10, D22.
  - Files: `src/jepa/cli.py`, `src/jepa/reporting.py`.
  - Verify: `uv run pytest tests/test_cli_sweep_reporting.py`.
- [x] **T7 (P2, human: ~3h / Codex: ~25min)** — Distribution — Document the hybrid policy, commands, artifacts, limitations, and add cross-version CI plus the two-dynamics smoke sweep.
  - Surfaced by: Distribution review and D23.
  - Files: `README.md`, `.github/workflows/ci.yml`, `tests/test_repository_hygiene.py`.
  - Verify: `uv run pytest`, `uv build`, local tiny 16-cell sweep.

## Acceptance criteria

- `uv sync` succeeds from a clean clone with no video dependencies.
- `uv run pytest` passes on CPU.
- Every SG/EMA cell satisfies module identity, gradient flow, optimizer membership, and EMA-order tests.
- Linear models contain no activation; nonlinear models contain exactly the configured hidden `tanh` layers.
- One dynamics/replicate sweep yields exactly eight unique completed-or-failed run records; both dynamics yield sixteen.
- Failed cells do not stop the sweep and cause a final non-zero exit.
- Test data remains untouched until final evaluation.
- Every completed run has resolved config, status, unique wide history rows, metrics, and final checkpoint.
- Resume is equivalent to uninterrupted pinned-CPU training and cannot mutate a completed run.
- `summary.svg` visibly compares MSE and context/target normalized effective rank.
- CI smoke-tests both synthetic dynamics end to end.
- Runtime and dependency graph contain no video/image/Transformer infrastructure.

## GSTACK REVIEW REPORT

### Decisions

- D1 A — Flatten runtime modules while preserving experimental completeness.
- D2 A — Use one wide, branch-prefixed history CSV.
- D3 A — Resume in the original run directory with integrity checks.
- D4 A — Isolate sweep cell failures and return non-zero after completion.
- D5 A — Remove single-value optimizer/objective config enums.
- D6 A — Replace ambiguous depth with `hidden_layers >= 1`.
- D7 A — Always evaluate and checkpoint the final epoch.
- D8 A — Append a canonical config hash to readable run IDs.
- D9-D14 A — Add full contract tests for resume, sweep failures, nonlinear structure, history schema, final artifacts, and identity hashing.
- D15 A — Eagerly materialize returned synthetic samples.
- D16 A — Keep test/probe evaluation final-only.
- D17 A — Limit exact replay to the pinned environment and save system matrices.
- D18 A — Treat sweep seeds as complete paired replicate seeds.
- D19 A — Use controlled contraction dynamics, orthonormal observation maps, and burn-in.
- D20 A — Define deterministic full-epoch metric aggregation.
- D21 A — Implement the exact ridge probe in PyTorch float64.
- D22 A — Emit a static sweep SVG.
- D23 A — Smoke-test both dynamics in CI.
- D24 A — Track repository ownership/remote selection in `TODOS.md`.
- D25 A — Track the first real time-series benchmark in `TODOS.md`.
- D26 B — Do not track or implement PyPI publication.

### Section result

- Scope: reduced from a deeply nested package to seven substantive runtime modules.
- Architecture: approved after resolving artifact identity, resume, sweep isolation, and research protocol boundaries.
- Code quality: approved after removing fake configurability and clarifying nonlinear depth/final events.
- Tests: every identified branch has a behavior, boundary, and error-path requirement.
- Performance: approved with eager data materialization and final-only probes/test evaluation.
- Outside voice: external Codex export was rejected for privacy; an in-task independent agent raised seven findings, all explicitly accepted as D17-D23.
- Remaining blockers: none for local implementation.
