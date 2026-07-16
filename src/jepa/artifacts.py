"""Artifact directory naming shared by single runs and sweeps."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MOSCOW_TIMEZONE = ZoneInfo("Europe/Moscow")


def moscow_timestamp(value: datetime | None = None) -> str:
    """Return a filesystem-safe wall-clock label in Moscow time."""
    current = datetime.now(MOSCOW_TIMEZONE) if value is None else value.astimezone(MOSCOW_TIMEZONE)
    return current.strftime("%Y-%m-%d_%H-%M-%S_MSK")


def create_timestamped_directory(root: str | Path) -> Path:
    """Atomically create a timestamp directory, adding a numeric collision suffix."""
    directory = Path(root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    base = moscow_timestamp()
    for index in range(1, 10_000):
        name = base if index == 1 else f"{base}_{index:02d}"
        candidate = directory / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not allocate a timestamped output directory under {directory}")
