#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
CONFIG="configs/tiny_vits4_q17_matched_g5_ep300_b256_q64_datasphere.yaml"
DATA_ROOT="../../data/tiny-imagenet-200"

if [[ ! -f "$DATA_ROOT/wnids.txt" ]]; then
  echo "Tiny ImageNet not found: $ROOT/$DATA_ROOT/wnids.txt" >&2
  exit 1
fi

"$PYTHON_BIN" -c \
  'import torch, torchvision, yaml; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

exec "$PYTHON_BIN" main.py --fname "$CONFIG" --devices cuda:0
