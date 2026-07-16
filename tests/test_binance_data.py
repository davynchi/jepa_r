from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

import jepa.binance as binance_module
from jepa.binance import archive_url, download_archives, prepare_binance_data
from jepa.cli import main
from jepa.config import (
    BinanceDataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from jepa.data import build_dataset_bundle
from jepa.training import train_experiment


def _timestamp(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _archive_bytes(path: Path, symbol: str, *, gap: bool = False) -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    rows: list[str] = []
    for index in range(6 * 96):
        if gap and index == 80:
            continue
        time = start + timedelta(minutes=15 * index)
        split_scale = 1.0 if time.day <= 2 else 20.0
        price = (100.0 + index * 0.01) * (1.0 + 0.1 * (symbol != "BTCUSDT"))
        opening = price
        close = price * (1.0 + 0.0001 * ((index % 5) - 2))
        high = max(opening, close) * 1.001
        low = min(opening, close) * 0.999
        base_volume = 10.0 * split_scale + index % 7
        quote_volume = 1000.0 * split_scale + index
        row = [
            str(_timestamp(time)),
            f"{opening:.8f}",
            f"{high:.8f}",
            f"{low:.8f}",
            f"{close:.8f}",
            f"{base_volume:.8f}",
            str(_timestamp(time + timedelta(minutes=15)) - 1),
            f"{quote_volume:.8f}",
            str(10 + index % 3),
            f"{base_volume * 0.4:.8f}",
            f"{quote_volume * 0.4:.8f}",
            "0",
        ]
        rows.append(",".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{symbol}-15m-2021-01.csv", "\n".join(rows) + "\n")


def _checksum(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.CHECKSUM").write_text(f"{digest}  {path.name}\n")


def _config(tmp_path: Path) -> BinanceDataConfig:
    return BinanceDataConfig(
        start="2021-01-01",
        end="2021-02-01",
        train_start="2021-01-01",
        train_end="2021-01-02",
        validation_start="2021-01-03",
        validation_end="2021-01-04",
        test_start="2021-01-05",
        test_end="2021-01-06",
        raw_cache_dir=str(tmp_path / "raw"),
        processed_root=str(tmp_path / "processed"),
        download_workers=2,
        download_retries=2,
    )


def _populate_offline(config: BinanceDataConfig, *, gap: bool = False) -> None:
    for symbol in config.symbols:
        url = archive_url(symbol, config.interval, datetime(2021, 1, 1).date())
        path = Path(config.raw_cache_dir) / symbol / config.interval / url.rsplit("/", 1)[-1]
        _archive_bytes(path, symbol, gap=gap and symbol == "BTCUSDT")
        _checksum(path)


def test_offline_preparation_alignment_splits_normalization_and_fingerprint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _populate_offline(config, gap=True)

    first = prepare_binance_data(config, offline=True)
    second = prepare_binance_data(config, offline=True)
    bundle = build_dataset_bundle(replace(config, fingerprint=first.fingerprint))
    manifest = json.loads((first.directory / "manifest.json").read_text())

    assert first.reused is False
    assert second.reused is True
    assert first.fingerprint == second.fingerprint == bundle.fingerprint
    assert bundle.train.sequences.shape[1:] == (64, 18)
    assert bundle.validation.sequences.shape[1:] == (64, 18)
    assert bundle.test.sequences.shape[1:] == (64, 18)
    assert manifest["schema_version"] == 2
    assert manifest["normalization"]["fit_split"] == "train"
    assert max(manifest["normalization"]["mean"][index] for index in (3, 9, 15)) < 8.0
    assert manifest["counts"]["timestamp_gaps"] >= 1
    assert manifest["counts"]["rejected_windows"]["train"] >= 1
    assert np.isfinite(bundle.train.sequences.numpy()).all()
    assert not list(Path(config.processed_root).glob(".staging-*"))
    assert not list(Path(config.raw_cache_dir).rglob("*.part"))

    experiment = ExperimentConfig(
        data=replace(config, fingerprint=first.fingerprint),
        model=ModelConfig(latent_dim=4),
        training=TrainingConfig(
            epochs=1,
            batch_size=64,
            device="cpu",
            evaluation_every_epochs=1,
            checkpoint_every_epochs=1,
        ),
        output=OutputConfig(root=str(tmp_path / "outputs")),
    )
    trained = train_experiment(experiment, datasets=bundle)
    assert trained.metrics["dataset_fingerprint"] == first.fingerprint
    assert trained.metrics["probes"]["context_to_context_state"]["test"] == {
        "value": None,
        "reason": "not_available_for_dataset",
    }
    assert trained.metrics["probes"]["context_to_target_window"]["test"]["value"] is not None


def test_download_retries_and_rejects_corrupt_offline_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        _config(tmp_path),
        symbols=("BTCUSDT",),
        observation_dim=6,
        download_workers=1,
    )
    source = tmp_path / "source.zip"
    _archive_bytes(source, "BTCUSDT")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    attempts = 0

    def fake_download(url: str, destination: Path) -> None:
        nonlocal attempts
        if url.endswith(".CHECKSUM"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"{digest}  BTCUSDT-15m-2021-01.zip\n")
            return
        attempts += 1
        if attempts == 1:
            raise OSError("temporary failure")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    monkeypatch.setattr(binance_module, "_atomic_download", fake_download)
    archives = download_archives(config)

    assert attempts == 2
    assert len(archives) == 1
    assert hashlib.sha256(archives[0][0].read_bytes()).hexdigest() == digest

    archives[0][0].write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        download_archives(config, offline=True)


def test_offline_and_force_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        download_archives(_config(tmp_path), offline=True, force=True)


def test_prepare_data_cli_uses_local_fixtures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    _populate_offline(config)
    config_path = tmp_path / "binance.yaml"
    config_path.write_text(yaml.safe_dump({"data": asdict(config)}))

    exit_code = main(["prepare-data", "--config", str(config_path), "--offline"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["state"] == "complete"
    assert output["sample_counts"]["train"] > 0
    assert Path(output["dataset_dir"]).name == output["fingerprint"]
