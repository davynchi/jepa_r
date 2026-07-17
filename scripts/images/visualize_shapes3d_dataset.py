#!/usr/bin/env python3
"""Preview-grid visualizations for the Shapes3D-backed entity/context world.

    python scripts/images/visualize_shapes3d_dataset.py \\
        --config configs/images/shapes3d/quick.yaml \\
        --output-dir outputs/temporal_shapes3d_previews

Mirrors ``visualize_temporal_image_dataset.py``: random static samples, random
temporal trajectories, same-entity/different-context and
and different-entity/same-context counterfactual pairs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jepa.configs.images.shapes3d import SHAPES3D_ENTITY_NAMES, load_shapes3d_config  # noqa: E402
from jepa.data.images.shapes3d import (  # noqa: E402
    Shapes3DEntityContextTrajectoryDataset,
    Shapes3DStaticImageDataset,
    build_shapes3d_counterfactual_pairs,
)


def _to_hw3(frame):
    return frame.permute(1, 2, 0).clamp(0, 1).numpy()


def _save_grid(rows: int, cols: int, cell_size: float):
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * cell_size, rows * cell_size), squeeze=False
    )
    for axis in axes.flat:
        axis.axis("off")
    return fig, axes


def visualize_static_samples(config, out_dir: Path, *, n: int = 8) -> None:
    dataset = Shapes3DStaticImageDataset(config.data, "train")
    n = min(n, len(dataset))
    fig, axes = _save_grid(1, n, 2.0)
    for i in range(n):
        axes[0, i].imshow(_to_hw3(dataset.images[i]))
        axes[0, i].set_title(SHAPES3D_ENTITY_NAMES[int(dataset.entities[i])], fontsize=8)
    fig.suptitle("Random static Shapes3D samples")
    fig.tight_layout()
    fig.savefig(out_dir / "preview_static_samples.png", dpi=150)
    plt.close(fig)


def visualize_temporal_trajectories(
    config, out_dir: Path, *, n: int = 4, frames_shown: int = 8
) -> None:
    dataset = Shapes3DEntityContextTrajectoryDataset(config.data, "train")
    n = min(n, len(dataset))
    frames_shown = min(frames_shown, config.data.trajectory_length)
    step = max(1, config.data.trajectory_length // frames_shown)
    time_indices = list(range(0, config.data.trajectory_length, step))[:frames_shown]
    fig, axes = _save_grid(n, len(time_indices), 1.6)
    for row in range(n):
        for col, t in enumerate(time_indices):
            axes[row, col].imshow(_to_hw3(dataset.frames[row, t]))
            if row == 0:
                axes[row, col].set_title(f"t={t}", fontsize=8)
            if col == 0:
                entity_name = SHAPES3D_ENTITY_NAMES[int(dataset.entities[row, t])]
                axes[row, col].set_ylabel(f"traj {row}\n({entity_name})", fontsize=7)
                axes[row, col].axis("on")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
    fig.suptitle("Random Shapes3D trajectories (shape persists, other factors drift)")
    fig.tight_layout()
    fig.savefig(out_dir / "preview_temporal_trajectories.png", dpi=150)
    plt.close(fig)


def visualize_counterfactual_pairs(config, out_dir: Path, *, n: int = 4) -> None:
    dataset = Shapes3DStaticImageDataset(config.data, "train")
    pairs = build_shapes3d_counterfactual_pairs(
        config.data, dataset.source, num_pairs=n, seed=config.data.counterfactual_seed
    )

    fig, axes = _save_grid(n, 2, 2.0)
    for i in range(n):
        axes[i, 0].imshow(_to_hw3(pairs.same_entity_x1[i]))
        axes[i, 1].imshow(_to_hw3(pairs.same_entity_x2[i]))
        entity_name = SHAPES3D_ENTITY_NAMES[int(pairs.same_entity[i])]
        axes[i, 0].set_title(f"{entity_name}, C1", fontsize=8)
        axes[i, 1].set_title(f"{entity_name}, C2", fontsize=8)
    fig.suptitle("Same shape, different context")
    fig.tight_layout()
    fig.savefig(out_dir / "preview_same_entity_diff_context.png", dpi=150)
    plt.close(fig)

    fig, axes = _save_grid(n, 2, 2.0)
    for i in range(n):
        axes[i, 0].imshow(_to_hw3(pairs.diff_entity_x1[i]))
        axes[i, 1].imshow(_to_hw3(pairs.diff_entity_x2[i]))
        e1, e2 = (int(v) for v in pairs.diff_entity_entities[i])
        axes[i, 0].set_title(SHAPES3D_ENTITY_NAMES[e1], fontsize=8)
        axes[i, 1].set_title(SHAPES3D_ENTITY_NAMES[e2], fontsize=8)
    fig.suptitle("Different shape, same context")
    fig.tight_layout()
    fig.savefig(out_dir / "preview_diff_entity_same_context.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_shapes3d_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    visualize_static_samples(config, args.output_dir)
    visualize_temporal_trajectories(config, args.output_dir)
    visualize_counterfactual_pairs(config, args.output_dir)
    print(f"wrote previews to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
