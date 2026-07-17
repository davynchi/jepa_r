#!/usr/bin/env python3
"""Train one entity/context temporal-persistence run.

Usage (Hydra-style, resolved against ``configs/<name>.yaml``):

    python scripts/train_temporal.py experiment=temporal_hierarchy_quick
    python scripts/train_temporal.py experiment=temporal_hierarchy_full

Equivalent explicit form:

    python scripts/train_temporal.py --config configs/temporal_hierarchy_quick.yaml

Strict dotted overrides are supported either way:

    python scripts/train_temporal.py experiment=temporal_hierarchy_quick \\
        data.entity_switch_probability=0.20 training.epochs=10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jepa.temporal_config import load_temporal_config  # noqa: E402
from jepa.temporal_training import train_temporal_experiment  # noqa: E402


def _parse_args(argv: list[str]) -> tuple[str | None, dict[str, str], int | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--seed-label", type=int, default=None)
    parser.add_argument(
        "hydra_style",
        nargs="*",
        help="KEY=VALUE tokens; experiment=<name> resolves to configs/<name>.yaml",
    )
    args = parser.parse_args(argv)

    config_path = args.config
    overrides: dict[str, str] = {}
    for token in args.hydra_style:
        if "=" not in token:
            raise SystemExit(f"expected KEY=VALUE, got {token!r}")
        key, _, value = token.partition("=")
        if key == "experiment":
            config_path = f"configs/{value}.yaml"
        else:
            overrides[key] = value
    return config_path, overrides, args.seed_label


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    config_path, overrides, seed_label = _parse_args(argv)
    config = load_temporal_config(config_path, overrides=overrides)
    result = train_temporal_experiment(config, seed_label=seed_label)
    print(json.dumps({"run_id": result.run_id, "run_dir": str(result.run_dir), **result.metrics}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
