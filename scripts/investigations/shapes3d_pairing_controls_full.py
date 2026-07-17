#!/usr/bin/env python3
"""Full-scale variant of ``shapes3d_pairing_controls.py``.

Same controls/metrics, but trained against ``temporal_shapes3d_full.yaml``
(4000/1000/1000 trajectories, 60 epochs) instead of the quick preset, since
the quick-preset run showed no separation between any condition (including
untrained random baselines) -- consistent with 5 epochs / 300 trajectories
simply being too little training budget for this nonlinear-rendering world,
rather than a genuine bottleneck confound. d_z grid extended to {8, 32, 64}
since compute is not the constraint here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import shapes3d_pairing_controls as base  # noqa: E402

from jepa.configs.images.shapes3d import load_shapes3d_config  # noqa: E402

base.OUTPUT_ROOT = str(
    Path(__file__).resolve().parents[2] / "outputs" / "shapes3d_bottleneck_grid_full"
)
base.D_Z_VALUES = (8, 32, 64)
base.SEEDS = (0, 1, 2)


def base_config():
    return load_shapes3d_config(
        "configs/images/shapes3d/full.yaml",
        overrides={"output.root": base.OUTPUT_ROOT, "output.overwrite": "on"},
    )


base.base_config = base_config

if __name__ == "__main__":
    base.main()
