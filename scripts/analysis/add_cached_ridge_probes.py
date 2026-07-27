#!/usr/bin/env python3
"""Add ridge results to probe benchmarks that already cache encoder features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from jepa.analysis.subspace import classifier_accuracy, fit_entity_classifier  # noqa: E402
from jepa.data.images.tiny_imagenet import (  # noqa: E402
    TinyImageNetDataConfig,
    build_tiny_imagenet_static_dataset_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=("context", "target"), default="target")
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    data_config = TinyImageNetDataConfig(**config["tiny_imagenet"]["data"])
    datasets = build_tiny_imagenet_static_dataset_splits(data_config)
    train_labels = datasets.train.entities.to(torch.long)
    test_labels = datasets.test.entities.to(torch.long)
    for result_path in sorted(args.benchmark_root.glob("epoch_*/results.json")):
        directory = result_path.parent
        train_features = torch.load(
            directory / "features" / f"{args.encoder}_train.pt",
            map_location="cpu",
            weights_only=True,
        )
        test_features = torch.load(
            directory / "features" / f"{args.encoder}_test.pt",
            map_location="cpu",
            weights_only=True,
        )
        classifier = fit_entity_classifier(
            train_features,
            train_labels,
            data_config.num_entities,
            ridge=args.ridge,
        )
        scores = classifier.probe.predict(test_features).float()
        top5 = (
            scores.topk(min(5, scores.shape[1]), dim=-1)
            .indices.eq(test_labels[:, None])
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )
        payload = json.loads(result_path.read_text())
        payload["probes"][f"{args.encoder}/ridge"] = {
            "loss": torch.nn.functional.cross_entropy(scores, test_labels).item(),
            "top1": classifier_accuracy(classifier, test_features, test_labels),
            "top5": top5,
        }
        result_path.write_text(json.dumps(payload, indent=2))
        print(
            f"{directory.name} {args.encoder}/ridge "
            f"top1={payload['probes'][f'{args.encoder}/ridge']['top1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
