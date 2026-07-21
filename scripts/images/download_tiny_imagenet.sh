#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data}"
URL="${TINY_IMAGENET_URL:-https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1}"
ARCHIVE="$DATA_ROOT/tiny-imagenet-200.zip"
TARGET="$DATA_ROOT/tiny-imagenet-200"

mkdir -p "$DATA_ROOT"

if [[ -d "$TARGET" ]]; then
  echo "Tiny ImageNet already exists at $TARGET"
  exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
  curl -L -o "$ARCHIVE" "$URL"
fi

python - "$ARCHIVE" "$DATA_ROOT" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys

archive = Path(sys.argv[1])
data_root = Path(sys.argv[2])
with ZipFile(archive) as zip_file:
    zip_file.extractall(data_root)
print(data_root / "tiny-imagenet-200")
PY
