"""Transactional canonical store for offline quality evaluation."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class UnsupportedQualityStoreSchema(ValueError):
    pass


class QualityStore:
    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=busy_timeout_ms / 1000)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        self._initialize()

    def _initialize(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, SCHEMA_VERSION):
            raise UnsupportedQualityStoreSchema(f"unsupported quality store schema: {version}")
        if version == 0:
            with self.transaction(immediate=True):
                self.connection.executescript(
                    """
                    CREATE TABLE metric_cells (
                      cell_key TEXT PRIMARY KEY,
                      run_id TEXT NOT NULL,
                      checkpoint_id TEXT NOT NULL,
                      panel TEXT NOT NULL,
                      metric_name TEXT NOT NULL,
                      metric_version TEXT NOT NULL,
                      state TEXT NOT NULL CHECK(state IN ('pending','running','complete','failed')),
                      value REAL,
                      null_reason TEXT,
                      error_type TEXT,
                      error_message TEXT,
                      diagnostics_json TEXT NOT NULL DEFAULT '{}',
                      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      CHECK (state != 'complete' OR ((value IS NULL) != (null_reason IS NULL)))
                    );
                    CREATE INDEX metric_cells_state_idx ON metric_cells(state);
                    CREATE TABLE accuracy_cells (
                      run_id TEXT NOT NULL,
                      checkpoint_id TEXT NOT NULL,
                      accuracy REAL NOT NULL,
                      balanced_accuracy REAL NOT NULL,
                      PRIMARY KEY(run_id, checkpoint_id)
                    );
                    """
                )
                self.connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def ensure_cell(self, identity: Mapping[str, str]) -> None:
        required = ("cell_key", "run_id", "checkpoint_id", "panel", "metric_name", "metric_version")
        missing = [name for name in required if not identity.get(name)]
        if missing:
            raise ValueError(f"missing cell identity fields: {missing}")
        with self.transaction(immediate=True):
            self.connection.execute(
                """INSERT OR IGNORE INTO metric_cells
                (cell_key,run_id,checkpoint_id,panel,metric_name,metric_version,state)
                VALUES (?,?,?,?,?,?,'pending')""",
                tuple(identity[name] for name in required),
            )

    def claim(self, cell_key: str, *, retry_failed: bool = True) -> bool:
        eligible = ("pending", "failed") if retry_failed else ("pending",)
        placeholders = ",".join("?" for _ in eligible)
        with self.transaction(immediate=True):
            cursor = self.connection.execute(
                f"""UPDATE metric_cells SET state='running', error_type=NULL,
                error_message=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE cell_key=? AND state IN ({placeholders})""",
                (cell_key, *eligible),
            )
        return cursor.rowcount == 1

    def complete(
        self,
        cell_key: str,
        *,
        value: float | None,
        null_reason: str | None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        if (value is None) == (null_reason is None):
            raise ValueError("completion requires exactly one of value or null_reason")
        payload = json.dumps(diagnostics or {}, sort_keys=True, allow_nan=False)
        with self.transaction(immediate=True):
            cursor = self.connection.execute(
                """UPDATE metric_cells SET state='complete', value=?, null_reason=?,
                diagnostics_json=?, updated_at=CURRENT_TIMESTAMP
                WHERE cell_key=? AND state='running'""",
                (value, null_reason, payload, cell_key),
            )
            if cursor.rowcount != 1:
                raise ValueError("cell is not running")

    def fail(self, cell_key: str, *, error_type: str, message: str) -> None:
        with self.transaction(immediate=True):
            cursor = self.connection.execute(
                """UPDATE metric_cells SET state='failed', error_type=?, error_message=?,
                updated_at=CURRENT_TIMESTAMP WHERE cell_key=? AND state='running'""",
                (error_type, message, cell_key),
            )
            if cursor.rowcount != 1:
                raise ValueError("cell is not running")

    def recover_interrupted(self) -> int:
        with self.transaction(immediate=True):
            cursor = self.connection.execute(
                """UPDATE metric_cells SET state='failed', error_type='interrupted',
                error_message='recovered stale running cell', updated_at=CURRENT_TIMESTAMP
                WHERE state='running'"""
            )
        return cursor.rowcount

    def rows(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute("SELECT * FROM metric_cells ORDER BY cell_key")
        ]

    def upsert_accuracy(
        self, run_id: str, checkpoint_id: str, *, accuracy: float, balanced_accuracy: float
    ) -> None:
        with self.transaction(immediate=True):
            self.connection.execute(
                """INSERT INTO accuracy_cells
                (run_id,checkpoint_id,accuracy,balanced_accuracy) VALUES (?,?,?,?)
                ON CONFLICT(run_id,checkpoint_id) DO UPDATE SET
                accuracy=excluded.accuracy, balanced_accuracy=excluded.balanced_accuracy""",
                (run_id, checkpoint_id, accuracy, balanced_accuracy),
            )

    def accuracy_rows(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM accuracy_cells ORDER BY run_id, checkpoint_id"
            )
        ]

    def export(self, directory: str | Path) -> tuple[Path, Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        rows = self.rows()
        jsonl = target / "records.jsonl"
        csv_path = target / "records.csv"
        json_tmp = jsonl.with_suffix(".jsonl.tmp")
        csv_tmp = csv_path.with_suffix(".csv.tmp")
        json_tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        with csv_tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["cell_key"])
            writer.writeheader()
            writer.writerows(rows)
        json_tmp.replace(jsonl)
        csv_tmp.replace(csv_path)
        accuracy_path = target / "accuracy.jsonl"
        accuracy_tmp = accuracy_path.with_suffix(".jsonl.tmp")
        accuracy_tmp.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.accuracy_rows())
        )
        accuracy_tmp.replace(accuracy_path)
        return jsonl, csv_path


__all__ = ["QualityStore", "SCHEMA_VERSION", "UnsupportedQualityStoreSchema"]
