#!/usr/bin/env bash
set -euo pipefail

LEGACY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(dirname -- "$LEGACY_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/shapes3d_corrupted_legacy}"
EPOCHS="${EPOCHS:-700}"

COMMON_ARGS=(
  --output-root "$OUTPUT_ROOT"
  --epochs "$EPOCHS"
  --batch-size 128
  --device cuda
  --architecture cnn
  --num-train-samples 16000
  --num-val-samples 1000
  --num-test-samples 1000
  --eval-every-epochs 10
  --checkpoint-every-epochs 25
  --log-every-steps 25
  --resident-device-data
  --corruption-fraction 0.2
  --corruption-mode noise
  --corruption-seed 1701
  --corruption-noise-std 0.75
  --weighting-warmup-epochs 100
  --weighting-update-every-epochs 100
  --weighting-score-batch-size 128
  --weighting-ref-size 1024
)

run_experiment() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHON_BIN="$PYTHON_BIN" \
    DIAGNOSTICS_DEVICE=cpu \
    legacy/train_shapes3d_corrupted_with_diagnostics.sh \
    "${COMMON_ARGS[@]}" "$@"
}

queue_a() {
  run_experiment "$GPU_A" \
    --run-name shapes3d_noise20_legacy_uniform_seed0_e700 \
    --weighting-method uniform
  run_experiment "$GPU_A" \
    --run-name shapes3d_noise20_legacy_ras_pr_seed0_e700 \
    --weighting-method ras \
    --weighting-richness pr
}

queue_b() {
  run_experiment "$GPU_B" \
    --run-name shapes3d_noise20_legacy_loss_seed0_e700 \
    --weighting-method loss
  run_experiment "$GPU_B" \
    --run-name shapes3d_noise20_legacy_oracle_clean_seed0_e700 \
    --weighting-method oracle-clean
}

queue_a &
PID_A=$!
queue_b &
PID_B=$!

STATUS=0
wait "$PID_A" || STATUS=$?
wait "$PID_B" || STATUS=$?
exit "$STATUS"
