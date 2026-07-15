"""Validate the complete artifact contract of a tiny 16-cell sweep."""

from __future__ import annotations

import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RUN_ARTIFACTS = {
    "checkpoint.pt",
    "config.yaml",
    "history.csv",
    "metrics.json",
    "status.json",
}
FINITE_SUMMARY_FIELDS = (
    "validation_mse",
    "test_mse",
    "context_effective_rank_normalized",
    "target_effective_rank_normalized",
)


def verify_smoke(output_root: Path) -> Path:
    sweep_dirs = sorted(path for path in output_root.glob("sweep-*") if path.is_dir())
    if len(sweep_dirs) != 1:
        raise AssertionError(f"expected one sweep directory, found {len(sweep_dirs)}")
    sweep_dir = sweep_dirs[0]
    status = json.loads((sweep_dir / "status.json").read_text())
    if status != {
        "schema_version": 1,
        "state": "complete",
        "completed_count": 16,
        "failed_count": 0,
        "cell_count": 16,
    }:
        raise AssertionError(f"unexpected sweep status: {status}")

    with (sweep_dir / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 16 or len({row["run_id"] for row in rows}) != 16:
        raise AssertionError("smoke sweep must contain 16 unique cells")
    if {row["dynamics"] for row in rows} != {"linear", "nonlinear"}:
        raise AssertionError("smoke sweep did not cover both dynamics")
    for row in rows:
        if row["state"] != "complete":
            raise AssertionError(f"cell did not complete: {row['run_id']}")
        for field in FINITE_SUMMARY_FIELDS:
            if not math.isfinite(float(row[field])):
                raise AssertionError(f"non-finite {field} in {row['run_id']}")
        run_dir = Path(row["run_dir"])
        if {path.name for path in run_dir.iterdir()} != RUN_ARTIFACTS:
            raise AssertionError(f"incomplete artifact set in {run_dir}")
        run_status = json.loads((run_dir / "status.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if run_status["state"] != "complete" or run_status["completed_epoch"] != 1:
            raise AssertionError(f"invalid run status in {run_dir}")
        if not all(math.isfinite(float(value)) for value in metrics["objective"].values()):
            raise AssertionError(f"non-finite objective in {run_dir}")

    with (sweep_dir / "summary_by_variant.csv").open(newline="") as stream:
        aggregates = list(csv.DictReader(stream))
    if len(aggregates) != 16:
        raise AssertionError("expected one aggregate row per dynamics/variant cell")
    ET.parse(sweep_dir / "summary.svg")
    return sweep_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_smoke.py OUTPUT_ROOT", file=sys.stderr)
        return 2
    try:
        sweep_dir = verify_smoke(Path(sys.argv[1]))
    except (AssertionError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"smoke verification failed: {error}", file=sys.stderr)
        return 1
    print(f"verified {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
