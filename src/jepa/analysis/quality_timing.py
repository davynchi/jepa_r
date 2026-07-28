"""Synchronized wall-clock timing and state isolation for live quality diagnostics."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch

from jepa.training.images.ijepa_spatial import SpatialIJEPACore


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work before reading a wall clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(slots=True)
class QualityTimingRecorder:
    device: torch.device
    stages: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    metric_formulas: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @contextmanager
    def stage(self, name: str, *, metric: bool = False) -> Iterator[None]:
        synchronize_device(self.device)
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            synchronize_device(self.device)
            seconds = (time.perf_counter_ns() - started) / 1_000_000_000
            target = self.metric_formulas if metric else self.stages
            target[name] += seconds

    def rows(self, *, run_id: str, epoch: int, global_step: int) -> list[dict]:
        common = {
            "run_id": run_id,
            "epoch": epoch,
            "global_step": global_step,
            "device": self.device.type,
        }
        return [
            {**common, "kind": kind, "name": name, "seconds": seconds}
            for kind, values in (
                ("stage", self.stages),
                ("metric_formula", self.metric_formulas),
            )
            for name, seconds in sorted(values.items())
        ]

    def upsert_jsonl(
        self,
        path: str | Path,
        *,
        run_id: str,
        epoch: int,
        global_step: int,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        by_key: dict[tuple[str, int, str, str], dict] = {}
        if destination.exists():
            for line in destination.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                by_key[(row["run_id"], int(row["epoch"]), row["kind"], row["name"])] = row
        for row in self.rows(run_id=run_id, epoch=epoch, global_step=global_step):
            by_key[(run_id, epoch, row["kind"], row["name"])] = row
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                for _, row in sorted(by_key.items())
            )
        )
        temporary.replace(destination)
        return destination


@contextmanager
def isolated_quality_evaluation(
    core: SpatialIJEPACore,
    device: torch.device,
) -> Iterator[None]:
    """Restore module modes and RNG even when a diagnostic fails."""
    modules = (core.context_encoder, core.predictor, core.target_encoder)
    modes = tuple(module.training for module in modules)
    cpu_rng = torch.random.get_rng_state()
    accelerator_rng = None
    if device.type == "cuda":
        accelerator_rng = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        accelerator_rng = torch.mps.get_rng_state()
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        synchronize_device(device)
        torch.random.set_rng_state(cpu_rng)
        if device.type == "cuda" and accelerator_rng is not None:
            torch.cuda.set_rng_state(accelerator_rng, device)
        elif device.type == "mps" and accelerator_rng is not None:
            torch.mps.set_rng_state(accelerator_rng)
        for module, training in zip(modules, modes, strict=True):
            module.train(training)


__all__ = [
    "QualityTimingRecorder",
    "isolated_quality_evaluation",
    "synchronize_device",
]
