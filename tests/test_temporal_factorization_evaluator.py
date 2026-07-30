from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jepa.training.images.ijepa_spatial import build_spatial_ijepa_core


def _tiny_imagenet_fixture(root: Path, *, count: int = 32) -> None:
    classes = ("n00000001", "n00000002")
    root.mkdir(parents=True)
    (root / "wnids.txt").write_text("\n".join(classes) + "\n")
    image_dir = root / "val" / "images"
    image_dir.mkdir(parents=True)
    annotations = []
    generator = np.random.default_rng(17)
    for index in range(count):
        filename = f"val_{index:04d}.JPEG"
        pixels = generator.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(image_dir / filename)
        annotations.append(f"{filename}\t{classes[index % len(classes)]}\t0\t0\t64\t64")
    (root / "val" / "val_annotations.txt").write_text("\n".join(annotations) + "\n")


def _official_checkpoints(root: Path) -> None:
    root.mkdir(parents=True)
    torch.manual_seed(23)
    core = build_spatial_ijepa_core(
        "vit_tiny",
        image_size=64,
        patch_size=8,
        predictor_embed_dim=32,
        predictor_depth=1,
    )
    for epoch in (50, 100):
        if epoch == 100:
            with torch.no_grad():
                for parameter in core.target_encoder.parameters():
                    parameter.add_(1e-4 * torch.randn_like(parameter))
        torch.save(
            {
                "encoder": core.context_encoder.state_dict(),
                "target_encoder": core.target_encoder.state_dict(),
                "epoch": epoch - 1,
            },
            root / f"jepa-ep{epoch}.pth.tar",
        )


def test_temporal_evaluator_runs_end_to_end(tmp_path: Path) -> None:
    data_root = tmp_path / "tiny-imagenet-200"
    checkpoint_root = tmp_path / "checkpoints"
    output_root = tmp_path / "output"
    _tiny_imagenet_fixture(data_root)
    _official_checkpoints(checkpoint_root)
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    command = [
        sys.executable,
        str(
            repository
            / "scripts"
            / "analysis"
            / "evaluate_upstream_ijepa_temporal_factorization.py"
        ),
        "--checkpoint-dir",
        str(checkpoint_root),
        "--epochs",
        "50",
        "100",
        "--reference-epoch",
        "50",
        "--tiny-imagenet-root",
        str(data_root),
        "--output-dir",
        str(output_root),
        "--encoder",
        "target",
        "--model-name",
        "vit_tiny",
        "--patch-size",
        "8",
        "--predictor-embed-dim",
        "32",
        "--predictor-depth",
        "1",
        "--test-size",
        "32",
        "--operator-train-size",
        "16",
        "--operator-validation-size",
        "8",
        "--pca-dimension",
        "8",
        "--num-blocks",
        "2",
        "4",
        "--transforms",
        "color",
        "flip",
        "--jbd-restarts",
        "1",
        "--jbd-steps",
        "3",
        "--batch-size",
        "16",
        "--device",
        "cpu",
        "--amp-dtype",
        "none",
    ]

    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    records = json.loads((output_root / "records.json").read_text())
    assert [record["epoch"] for record in records] == [50, 100]
    assert records[0]["blocks"]["2"]["temporal"]["q5_similarity"] is None
    assert records[1]["blocks"]["2"]["temporal"]["q5_similarity"] is not None
    assert records[1]["blocks"]["4"]["q12"]["value"] is not None
    assert set(records[1]["q17"]) == {"color", "flip"}
    assert "q21_vs_reference" in records[1]["q21"]
    assert (output_root / "states" / "epoch_0100.pt").exists()
    assert (output_root / "temporal_factorization.png").exists()
