from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jepa.artifacts as artifacts_module
from jepa.artifacts import create_timestamped_directory, moscow_timestamp


def test_moscow_timestamp_converts_from_utc() -> None:
    assert moscow_timestamp(datetime(2026, 1, 2, 10, 11, 12, tzinfo=UTC)) == (
        "2026-01-02_13-11-12_MSK"
    )


def test_timestamp_directory_uses_deterministic_collision_suffix(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        artifacts_module,
        "moscow_timestamp",
        lambda: "2026-01-02_13-11-12_MSK",
    )

    first = create_timestamped_directory(tmp_path)
    second = create_timestamped_directory(tmp_path)

    assert first.name == "2026-01-02_13-11-12_MSK"
    assert second.name == "2026-01-02_13-11-12_MSK_02"
