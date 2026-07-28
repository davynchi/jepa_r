#!/usr/bin/env python3
"""Convert an official I-JEPA checkpoint for the project's probe tools."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jepa.training.core import SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--model-name", default="vit_small")
    parser.add_argument("--predictor-embed-dim", type=int, default=384)
    parser.add_argument("--predictor-depth", type=int, default=12)
    return parser.parse_args()


def strip_ddp_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state.items()
    }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    required = {"encoder", "predictor", "target_encoder", "epoch"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"upstream checkpoint is missing keys: {sorted(missing)}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "epoch": int(checkpoint["epoch"]),
        "global_step": None,
        "context_encoder": strip_ddp_prefix(checkpoint["encoder"]),
        "predictor": strip_ddp_prefix(checkpoint["predictor"]),
        "target_encoder": strip_ddp_prefix(checkpoint["target_encoder"]),
        "metadata": {
            "model_family": "upstream_ijepa",
            "model_name": args.model_name,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "predictor_embed_dim": args.predictor_embed_dim,
            "predictor_depth": args.predictor_depth,
            "source": "facebookresearch/ijepa",
        },
        "upstream": {
            key: checkpoint.get(key)
            for key in ("batch_size", "world_size", "lr", "loss")
        },
    }
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"converted epoch={payload['epoch']} to {output}")


if __name__ == "__main__":
    main()
