#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {uniform|matched-1pct|matched-5pct|shuffled-5pct}" >&2
  exit 2
fi

MODE="$1"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ijepa_spatial}"
DATA_ROOT="${DATA_ROOT:-data/tiny-imagenet-200/tiny-imagenet-200}"
EPOCHS="${EPOCHS:-300}"
SEED="${SEED:-0}"

COMMON_ARGS=(
  --output-root "$OUTPUT_ROOT"
  --epochs "$EPOCHS"
  --batch-size 1024
  --device cuda
  --model-name vit_small
  --image-size 64
  --patch-size 4
  --predictor-embed-dim 192
  --predictor-depth 6
  --amp-dtype bfloat16
  --crop-scale 0.3 1.0
  --horizontal-flip-prob 0.5
  --dataset tiny-imagenet
  --tiny-imagenet-root "$DATA_ROOT"
  --num-train-samples 100000
  --num-val-samples 1000
  --num-test-samples 10000
  --eval-every-epochs 25
  --probe-every-epochs 25
  --probe-train-size 0
  --probe-test-size 0
  --checkpoint-every-epochs 25
  --log-every-steps 20
  --mask-loader-workers 10
  --resident-device-data
  --weighting-method uniform
  --seed "$SEED"
)

case "$MODE" in
  uniform)
    RUN_NAME="tiny_vits_p4_uniform_seed${SEED}_e${EPOCHS}_b1024"
    EXTRA_ARGS=(--residual-q17-gradient-ratio 0)
    ;;
  matched-1pct)
    RUN_NAME="tiny_vits_p4_residual_q17_matched_g1_seed${SEED}_e${EPOCHS}_b1024"
    EXTRA_ARGS=(
      --residual-q17-gradient-ratio 0.01
      --residual-q17-mode matched
    )
    ;;
  matched-5pct)
    RUN_NAME="tiny_vits_p4_residual_q17_matched_g5_seed${SEED}_e${EPOCHS}_b1024"
    EXTRA_ARGS=(
      --residual-q17-gradient-ratio 0.05
      --residual-q17-mode matched
    )
    ;;
  shuffled-5pct)
    RUN_NAME="tiny_vits_p4_residual_q17_shuffled_g5_seed${SEED}_e${EPOCHS}_b1024"
    EXTRA_ARGS=(
      --residual-q17-gradient-ratio 0.05
      --residual-q17-mode shuffled
    )
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ "$MODE" != "uniform" ]]; then
  EXTRA_ARGS+=(
    --residual-q17-transforms flip blur color
    --residual-q17-batch-size 0
    --residual-q17-statistics-decay 0.99
    --residual-q17-operator-lr 0.001
    --residual-q17-operator-weight-decay 0.0001
    --residual-q17-blur-sigma 1.0
    --residual-q17-color-strength 0.2
  )
fi

echo "mode=$MODE run_name=$RUN_NAME gpu=$GPU dataset=tiny-imagenet model=vit_small/4"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/images/train_ijepa_spatial.py \
  --run-name "$RUN_NAME" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
