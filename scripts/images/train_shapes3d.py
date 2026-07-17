#!/usr/bin/env python3
"""Train one Shapes3D-backed entity/context run.

    python scripts/images/train_shapes3d.py experiment=temporal_shapes3d_quick

Requires ``data/3dshapes.h5`` (downloaded once):
    curl -o data/3dshapes.h5 https://storage.googleapis.com/3d-shapes/3dshapes.h5

Same Hydra-style ``experiment=<name>`` / strict-override interface as
``train_temporal.py`` / ``train_temporal_image.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from jepa.configs.images.shapes3d import load_shapes3d_config  # noqa: E402
from jepa.training.images.shapes3d import train_shapes3d_experiment  # noqa: E402


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
    config = load_shapes3d_config(config_path, overrides=overrides)
    result = train_shapes3d_experiment(config, seed_label=seed_label)
    print(json.dumps({"run_id": result.run_id, "run_dir": str(result.run_dir), **result.metrics}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
