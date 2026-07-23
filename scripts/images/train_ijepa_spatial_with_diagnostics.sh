#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DIAGNOSTICS_DEVICE="${DIAGNOSTICS_DEVICE:-cpu}"
DIAGNOSTICS_POLL_SECONDS="${DIAGNOSTICS_POLL_SECONDS:-15}"
DIAGNOSTICS_TENSORBOARD="${DIAGNOSTICS_TENSORBOARD:-1}"

RUN_NAME=""
OUTPUT_ROOT="outputs/ijepa_spatial"
DATASET="shapes3d"
TRAIN_ARGS=("$@")

while (($# > 0)); do
  case "$1" in
    --run-name)
      RUN_NAME="${2:-}"
      shift 2
      ;;
    --run-name=*)
      RUN_NAME="${1#--run-name=}"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --output-root=*)
      OUTPUT_ROOT="${1#--output-root=}"
      shift
      ;;
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    --dataset=*)
      DATASET="${1#--dataset=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  echo "error: --run-name is required so the diagnostics watcher can find the run" >&2
  exit 2
fi

case "$OUTPUT_ROOT" in
  /*)
    RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
    ;;
  *)
    RUN_DIR="$(pwd -P)/$OUTPUT_ROOT/$RUN_NAME"
    ;;
esac
TRAIN_LOG="$RUN_DIR/train.log"
DIAGNOSTICS_LOG="$RUN_DIR/diagnostics_watcher.log"

mkdir -p "$RUN_DIR"

"$PYTHON_BIN" scripts/images/train_ijepa_spatial.py "${TRAIN_ARGS[@]}" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

cleanup() {
  if [[ "${DIAGNOSTICS_PID:-}" ]]; then
    kill "$DIAGNOSTICS_PID" 2>/dev/null || true
    wait "$DIAGNOSTICS_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

until [[ -f "$RUN_DIR/config.json" ]] || ! kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 1
done

if [[ ! -f "$RUN_DIR/config.json" ]]; then
  wait "$TRAIN_PID"
  exit $?
fi

case "$DATASET" in
  shapes3d)
    DIAGNOSTICS_SCRIPT="scripts/analysis/track_spatial_diagnostics.py"
    ;;
  tiny-imagenet)
    DIAGNOSTICS_SCRIPT="scripts/analysis/track_tiny_imagenet_diagnostics.py"
    ;;
  *)
    echo "error: no diagnostics watcher for dataset $DATASET" > "$DIAGNOSTICS_LOG"
    wait "$TRAIN_PID"
    exit $?
    ;;
esac

DIAGNOSTICS_ARGS=(
  "$DIAGNOSTICS_SCRIPT"
  --run-dir "$RUN_DIR"
  --poll-seconds "$DIAGNOSTICS_POLL_SECONDS"
  --device "$DIAGNOSTICS_DEVICE"
)
if [[ "$DIAGNOSTICS_TENSORBOARD" == "0" ]]; then
  DIAGNOSTICS_ARGS+=(--no-tensorboard)
fi

"$PYTHON_BIN" "${DIAGNOSTICS_ARGS[@]}" > "$DIAGNOSTICS_LOG" 2>&1 &
DIAGNOSTICS_PID=$!

if wait "$TRAIN_PID"; then
  TRAIN_STATUS=0
else
  TRAIN_STATUS=$?
fi
cleanup
if [[ "$TRAIN_STATUS" -eq 0 ]]; then
  "$PYTHON_BIN" "${DIAGNOSTICS_ARGS[@]}" --once >> "$DIAGNOSTICS_LOG" 2>&1
fi
exit "$TRAIN_STATUS"
