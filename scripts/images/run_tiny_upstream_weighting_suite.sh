#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ijepa_upstream_tiny}"
EPOCHS="${EPOCHS:-700}"

COMMON_ARGS=(
  --output-root "$OUTPUT_ROOT"
  --epochs "$EPOCHS"
  --batch-size 512
  --device cuda
  --model-name vit_tiny
  --image-size 64
  --patch-size 8
  --predictor-embed-dim 192
  --predictor-depth 6
  --bfloat16
  --crop-scale 0.3 1.0
  --horizontal-flip-prob 0.5
  --dataset tiny-imagenet
  --tiny-imagenet-root data/tiny-imagenet-200/tiny-imagenet-200
  --num-train-samples 100000
  --num-val-samples 1000
  --num-test-samples 1000
  --eval-every-epochs 10
  --checkpoint-every-epochs 25
  --log-every-steps 50
  --resident-device-data
  --weighting-warmup-epochs 100
  --weighting-update-every-epochs 100
  --weighting-score-batch-size 512
  --weighting-ref-size 1024
)

run_experiment() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHON_BIN="$PYTHON_BIN" \
    DIAGNOSTICS_DEVICE=cpu \
    scripts/images/train_ijepa_spatial_with_diagnostics.sh \
    "${COMMON_ARGS[@]}" "$@"
}

queue_a() {
  run_experiment "$GPU_A" \
    --run-name tiny_vit_tiny_upstream_uniform_seed0_e700_b512 \
    --weighting-method uniform
  run_experiment "$GPU_A" \
    --run-name tiny_vit_tiny_upstream_ras_pr_seed0_e700_b512_score256 \
    --weighting-method ras \
    --weighting-richness pr \
    --ras-score-granularity batch \
    --weighting-score-batch-size 256
}

queue_b() {
  run_experiment "$GPU_B" \
    --run-name tiny_vit_tiny_upstream_loss_seed0_e700_b512 \
    --weighting-method loss
  run_experiment "$GPU_B" \
    --run-name tiny_vit_tiny_upstream_ras_logdet_seed0_e700_b512_score256 \
    --weighting-method ras \
    --weighting-richness logdet \
    --ras-score-granularity batch \
    --weighting-score-batch-size 256
}

queue_a &
PID_A=$!
queue_b &
PID_B=$!

STATUS=0
wait "$PID_A" || STATUS=$?
wait "$PID_B" || STATUS=$?
exit "$STATUS"
