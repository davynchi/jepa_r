from __future__ import annotations

import sqlite3

import pytest

from jepa.analysis.quality_store import QualityStore, UnsupportedQualityStoreSchema


def _identity(key: str = "cell") -> dict[str, str]:
    return {
        "cell_key": key,
        "run_id": "run",
        "checkpoint_id": "ckpt",
        "panel": "label_free",
        "metric_name": "q1_cross_covariance",
        "metric_version": "1",
    }


def test_lifecycle_is_idempotent_and_completed_cells_are_immutable(tmp_path) -> None:
    store = QualityStore(tmp_path / "quality.sqlite")
    store.ensure_cell(_identity())
    store.ensure_cell(_identity())
    assert store.claim("cell")
    assert not store.claim("cell")
    store.complete("cell", value=0.25, null_reason=None, diagnostics={"n": 4})
    assert not store.claim("cell")
    assert store.rows()[0]["value"] == 0.25


def test_transaction_rolls_back_and_recovery_retries(tmp_path) -> None:
    store = QualityStore(tmp_path / "quality.sqlite")
    store.ensure_cell(_identity())
    with pytest.raises(RuntimeError):
        with store.transaction(immediate=True):
            store.connection.execute("UPDATE metric_cells SET state='running'")
            raise RuntimeError("boom")
    assert store.rows()[0]["state"] == "pending"
    assert store.claim("cell")
    assert store.recover_interrupted() == 1
    assert store.rows()[0]["error_type"] == "interrupted"
    assert store.claim("cell")


def test_two_connections_cannot_claim_same_cell(tmp_path) -> None:
    path = tmp_path / "quality.sqlite"
    first = QualityStore(path)
    second = QualityStore(path)
    first.ensure_cell(_identity())
    assert first.claim("cell")
    assert not second.claim("cell")


def test_schema_rejection_and_export(tmp_path) -> None:
    path = tmp_path / "future.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version=99")
    connection.close()
    with pytest.raises(UnsupportedQualityStoreSchema):
        QualityStore(path)

    store = QualityStore(tmp_path / "quality.sqlite")
    store.ensure_cell(_identity())
    assert store.claim("cell")
    store.complete("cell", value=None, null_reason="zero_denominator")
    jsonl, csv_path = store.export(tmp_path / "exports")
    assert '"null_reason": "zero_denominator"' in jsonl.read_text()
    assert "zero_denominator" in csv_path.read_text()
