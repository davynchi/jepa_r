"""Run logging utilities for spatial I-JEPA experiments."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

TensorBoardSummaryWriter: Any
try:
    from torch.utils.tensorboard import SummaryWriter as TensorBoardSummaryWriter
except ModuleNotFoundError:  # pragma: no cover - depends on optional local install.
    TensorBoardSummaryWriter = None


class SpatialRunLogger:
    """Write durable JSONL metrics and optional TensorBoard summaries."""

    def __init__(self, run_dir: str | Path, *, enable_tensorboard: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.writer: Any | None = None
        if enable_tensorboard and TensorBoardSummaryWriter is not None:
            self.writer = TensorBoardSummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

    def write_config(self, payload: Mapping[str, Any]) -> None:
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")

    def log(
        self,
        *,
        step: int,
        epoch: int,
        event: str,
        scalars: Mapping[str, float | int | bool | None],
        histograms: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "time": time.time(),
            "step": step,
            "epoch": epoch,
            "event": event,
            "scalars": dict(scalars),
        }
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

        if self.writer is None:
            return
        for name, value in scalars.items():
            if value is None or isinstance(value, bool):
                continue
            self.writer.add_scalar(name, float(value), step)
        for name, values in (histograms or {}).items():
            self.writer.add_histogram(name, values.detach().float().cpu(), step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def eigenvalue_scalars(eigenvalues: Any, *, top_k: int = 10) -> dict[str, float]:
    values = [float(v) for v in eigenvalues[:top_k]]
    scalars = {f"repr/top_eig_{index + 1:02d}": value for index, value in enumerate(values)}
    total = float(sum(float(v) for v in eigenvalues))
    if total > 0:
        for k in (1, 4, 8):
            scalars[f"repr/eig_mass_top_{k}"] = float(
                sum(float(v) for v in eigenvalues[:k]) / total
            )
    return scalars
