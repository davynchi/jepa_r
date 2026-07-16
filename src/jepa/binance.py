"""Official Binance Spot kline download, validation, preprocessing, and caching."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import urllib.request
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jepa.config import BinanceDataConfig, validate_data_config
from jepa.data import DatasetBundle, WindowDataset

_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
_INTERVAL_MS = 15 * 60 * 1000
_FEATURE_NAMES = (
    "log_return",
    "candle_range",
    "candle_body",
    "log1p_quote_asset_volume",
    "log1p_number_of_trades",
    "taker_quote_imbalance",
)


@dataclass(frozen=True, slots=True)
class PreparationResult:
    fingerprint: str
    directory: Path
    split_samples: dict[str, int]
    reused: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "directory": str(self.directory),
            "split_samples": self.split_samples,
            "reused": self.reused,
        }


@dataclass(frozen=True, slots=True)
class _Klines:
    timestamps: np.ndarray
    opening: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray
    trades: np.ndarray
    taker_buy_quote_volume: np.ndarray


def _month_starts(start: str, end: str) -> tuple[date, ...]:
    current = date.fromisoformat(start).replace(day=1)
    stop = date.fromisoformat(end)
    months: list[date] = []
    while current < stop:
        months.append(current)
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
    return tuple(months)


def _next_month(value: date) -> date:
    return date(value.year + (value.month == 12), value.month % 12 + 1, 1)


def _timestamp_ms(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp() * 1000)


def archive_url(symbol: str, interval: str, month: date) -> str:
    filename = f"{symbol}-{interval}-{month:%Y-%m}.zip"
    return f"{_BASE_URL}/{symbol}/{interval}/{filename}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _checksum_value(path: Path, expected_filename: str) -> str:
    fields = path.read_text().strip().split()
    if len(fields) < 2 or fields[1].lstrip("*") != expected_filename:
        raise ValueError(f"invalid checksum file: {path}")
    value = fields[0].lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid SHA-256 in checksum file: {path}")
    return value


def _ensure_archive(
    config: BinanceDataConfig,
    symbol: str,
    month: date,
    *,
    offline: bool,
    force: bool,
) -> tuple[Path, str, str]:
    url = archive_url(symbol, config.interval, month)
    filename = url.rsplit("/", 1)[-1]
    directory = Path(config.raw_cache_dir).expanduser().resolve() / symbol / config.interval
    archive = directory / filename
    checksum = directory / f"{filename}.CHECKSUM"
    last_error: Exception | None = None
    for attempt in range(config.download_retries):
        try:
            if offline:
                if not archive.is_file() or not checksum.is_file():
                    raise FileNotFoundError(f"offline cache is missing {archive} or {checksum}")
            elif force or not checksum.is_file():
                _atomic_download(f"{url}.CHECKSUM", checksum)
            expected = _checksum_value(checksum, filename)
            if archive.is_file() and not force and _sha256_file(archive) == expected:
                return archive, expected, url
            if offline:
                raise ValueError(f"offline cached archive failed checksum: {archive}")
            candidate = archive.with_name(f".{archive.name}.{uuid.uuid4().hex}.candidate")
            try:
                _atomic_download(url, candidate)
                actual = _sha256_file(candidate)
                if actual != expected:
                    raise ValueError(
                        f"checksum mismatch for {filename}: expected {expected}, got {actual}"
                    )
                os.replace(candidate, archive)
            finally:
                candidate.unlink(missing_ok=True)
            return archive, expected, url
        except Exception as error:
            last_error = error
            if offline or attempt + 1 == config.download_retries:
                break
            time.sleep(0.25 * (2**attempt))
    assert last_error is not None
    raise RuntimeError(f"failed to prepare {symbol} {month:%Y-%m}: {last_error}") from last_error


def download_archives(
    config: BinanceDataConfig, *, offline: bool = False, force: bool = False
) -> tuple[tuple[Path, str, str, str], ...]:
    """Download and verify all configured monthly archives."""
    validate_data_config(config)
    if offline and force:
        raise ValueError("--offline and --force cannot be used together")
    jobs = [
        (symbol, month)
        for symbol in config.symbols
        for month in _month_starts(config.start, config.end)
    ]

    def run(job: tuple[str, date]) -> tuple[Path, str, str, str]:
        symbol, month = job
        path, checksum, url = _ensure_archive(config, symbol, month, offline=offline, force=force)
        return path, checksum, url, symbol

    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        return tuple(executor.map(run, jobs))


def _parse_float(path: Path, line: int, name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{path}:{line}: {name} must be numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{line}: {name} must be finite")
    return parsed


def _parse_archive(path: Path, month: date) -> _Klines:
    columns: dict[str, list[float | int]] = {
        "timestamps": [],
        "opening": [],
        "high": [],
        "low": [],
        "close": [],
        "quote_volume": [],
        "trades": [],
        "taker_buy_quote_volume": [],
    }
    start_ms = _timestamp_ms(month)
    end_ms = _timestamp_ms(_next_month(month))
    previous: int | None = None
    with zipfile.ZipFile(path) as archive:
        members = [member for member in archive.namelist() if not member.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected exactly one CSV member")
        with archive.open(members[0]) as raw:
            rows = csv.reader(line.decode("utf-8") for line in raw)
            for line_number, row in enumerate(rows, start=1):
                if line_number == 1 and row and not row[0].strip().isdigit():
                    continue
                if len(row) != 12:
                    raise ValueError(f"{path}:{line_number}: expected 12 columns, got {len(row)}")
                try:
                    timestamp = int(row[0])
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number}: open time must be integer milliseconds"
                    ) from error
                if not start_ms <= timestamp < end_ms or timestamp % _INTERVAL_MS:
                    raise ValueError(
                        f"{path}:{line_number}: timestamp is outside month or 15m grid"
                    )
                if previous is not None and timestamp <= previous:
                    raise ValueError(
                        f"{path}:{line_number}: timestamps must be strictly increasing"
                    )
                previous = timestamp
                opening = _parse_float(path, line_number, "open", row[1])
                high = _parse_float(path, line_number, "high", row[2])
                low = _parse_float(path, line_number, "low", row[3])
                close = _parse_float(path, line_number, "close", row[4])
                base_volume = _parse_float(path, line_number, "volume", row[5])
                quote_volume = _parse_float(path, line_number, "quote volume", row[7])
                trades_float = _parse_float(path, line_number, "number of trades", row[8])
                taker_base = _parse_float(path, line_number, "taker buy base volume", row[9])
                taker_quote = _parse_float(path, line_number, "taker buy quote volume", row[10])
                if min(opening, high, low, close) <= 0:
                    raise ValueError(f"{path}:{line_number}: OHLC values must be positive")
                if low > min(opening, close) or high < max(opening, close) or low > high:
                    raise ValueError(f"{path}:{line_number}: OHLC range is inconsistent")
                if (
                    base_volume < 0
                    or quote_volume < 0
                    or trades_float < 0
                    or taker_base < 0
                    or taker_quote < 0
                ):
                    raise ValueError(
                        f"{path}:{line_number}: volumes and trades must be non-negative"
                    )
                if not trades_float.is_integer():
                    raise ValueError(f"{path}:{line_number}: number of trades must be integral")
                if taker_quote > quote_volume + max(1e-12, quote_volume * 1e-9):
                    raise ValueError(
                        f"{path}:{line_number}: taker quote volume exceeds quote volume"
                    )
                if taker_base > base_volume + max(1e-12, base_volume * 1e-9):
                    raise ValueError(f"{path}:{line_number}: taker base volume exceeds base volume")
                columns["timestamps"].append(timestamp)
                columns["opening"].append(opening)
                columns["high"].append(high)
                columns["low"].append(low)
                columns["close"].append(close)
                columns["quote_volume"].append(quote_volume)
                columns["trades"].append(int(trades_float))
                columns["taker_buy_quote_volume"].append(taker_quote)
    if not columns["timestamps"]:
        raise ValueError(f"{path}: archive contains no kline rows")
    return _Klines(
        timestamps=np.asarray(columns["timestamps"], dtype=np.int64),
        opening=np.asarray(columns["opening"], dtype=np.float64),
        high=np.asarray(columns["high"], dtype=np.float64),
        low=np.asarray(columns["low"], dtype=np.float64),
        close=np.asarray(columns["close"], dtype=np.float64),
        quote_volume=np.asarray(columns["quote_volume"], dtype=np.float64),
        trades=np.asarray(columns["trades"], dtype=np.float64),
        taker_buy_quote_volume=np.asarray(columns["taker_buy_quote_volume"], dtype=np.float64),
    )


def _concatenate(parts: list[_Klines]) -> _Klines:
    fields = _Klines.__dataclass_fields__
    values = {name: np.concatenate([getattr(part, name) for part in parts]) for name in fields}
    timestamps = values["timestamps"]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("monthly archives overlap or are out of order")
    return _Klines(**values)


def _feature_rows(klines: _Klines) -> dict[int, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    for index in range(1, klines.timestamps.size):
        if int(klines.timestamps[index] - klines.timestamps[index - 1]) != _INTERVAL_MS:
            continue
        quote = float(klines.quote_volume[index])
        taker = float(klines.taker_buy_quote_volume[index])
        if quote == 0:
            if taker != 0:
                raise ValueError("non-zero taker volume with zero quote volume")
            imbalance = 0.0
        else:
            imbalance = 2.0 * taker / quote - 1.0
        feature = np.asarray(
            [
                math.log(float(klines.close[index] / klines.close[index - 1])),
                math.log(float(klines.high[index] / klines.low[index])),
                math.log(float(klines.close[index] / klines.opening[index])),
                math.log1p(quote),
                math.log1p(float(klines.trades[index])),
                imbalance,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(feature).all():
            raise ValueError(
                f"non-finite engineered feature at timestamp {klines.timestamps[index]}"
            )
        rows[int(klines.timestamps[index])] = feature
    return rows


def binance_spec(config: BinanceDataConfig) -> dict[str, Any]:
    values = asdict(config)
    for key in (
        "raw_cache_dir",
        "processed_root",
        "fingerprint",
        "download_workers",
        "download_retries",
    ):
        values.pop(key)
    values["feature_names"] = list(_FEATURE_NAMES)
    values["schema_version"] = 2
    return values


def _spec_hash(config: BinanceDataConfig) -> str:
    canonical = json.dumps(binance_spec(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _build_windows(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    start: str,
    end: str,
    steps: int,
    stride: int,
) -> tuple[np.ndarray, int]:
    start_ms = _timestamp_ms(date.fromisoformat(start))
    end_ms = _timestamp_ms(date.fromisoformat(end))
    selected = np.flatnonzero((timestamps >= start_ms) & (timestamps < end_ms))
    split_times = timestamps[selected]
    split_values = values[selected]
    windows: list[np.ndarray] = []
    rejected = 0
    for index in range(0, max(split_times.size - steps + 1, 0), stride):
        candidate_times = split_times[index : index + steps]
        if int(candidate_times[-1] - candidate_times[0]) != (steps - 1) * _INTERVAL_MS:
            rejected += 1
            continue
        windows.append(split_values[index : index + steps])
    if not windows:
        raise ValueError(f"split [{start}, {end}) produced no contiguous windows")
    return np.stack(windows).astype(np.float32, copy=False), rejected


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _publish_dataset(
    config: BinanceDataConfig,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> PreparationResult:
    manifest["array_hashes"] = {name: _array_hash(array) for name, array in arrays.items()}
    fingerprint = _manifest_fingerprint(manifest)
    manifest["fingerprint"] = fingerprint
    root = Path(config.processed_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / fingerprint
    if destination.is_dir():
        _load_directory(destination, expected_fingerprint=fingerprint)
        return PreparationResult(
            fingerprint,
            destination,
            {
                name.removesuffix("_sequences"): int(array.shape[0])
                for name, array in arrays.items()
            },
            True,
        )
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        np.savez_compressed(
            staging / "dataset.npz",
            train_sequences=arrays["train_sequences"],
            validation_sequences=arrays["validation_sequences"],
            test_sequences=arrays["test_sequences"],
        )
        with (staging / "manifest.json").open("w") as output:
            json.dump(manifest, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return PreparationResult(
        fingerprint,
        destination,
        {name.removesuffix("_sequences"): int(array.shape[0]) for name, array in arrays.items()},
        False,
    )


def prepare_binance_data(
    config: BinanceDataConfig, *, offline: bool = False, force: bool = False
) -> PreparationResult:
    """Download official klines and atomically publish normalized fixed windows."""
    archives = download_archives(config, offline=offline, force=force)
    grouped: dict[str, list[tuple[Path, str, str]]] = {symbol: [] for symbol in config.symbols}
    for path, checksum, url, symbol in archives:
        grouped[symbol].append((path, checksum, url))
    feature_maps: dict[str, dict[int, np.ndarray]] = {}
    sources: list[dict[str, str]] = []
    for symbol in config.symbols:
        parts: list[_Klines] = []
        for month, (path, checksum, url) in zip(
            _month_starts(config.start, config.end), grouped[symbol], strict=True
        ):
            parts.append(_parse_archive(path, month))
            sources.append({"symbol": symbol, "url": url, "sha256": checksum})
        feature_maps[symbol] = _feature_rows(_concatenate(parts))

    common = sorted(set.intersection(*(set(rows) for rows in feature_maps.values())))
    if not common:
        raise ValueError("symbols have no common valid timestamps")
    timestamps = np.asarray(common, dtype=np.int64)
    values = np.stack(
        [
            np.concatenate([feature_maps[symbol][timestamp] for symbol in config.symbols])
            for timestamp in common
        ]
    )
    train_start = _timestamp_ms(date.fromisoformat(config.train_start))
    train_end = _timestamp_ms(date.fromisoformat(config.train_end))
    train_rows = values[(timestamps >= train_start) & (timestamps < train_end)]
    if train_rows.size == 0:
        raise ValueError("training interval has no aligned rows")
    mean = train_rows.mean(axis=0)
    std = np.maximum(train_rows.std(axis=0), 1e-8)
    normalized = (values - mean) / std
    steps = config.context_steps + config.target_steps
    train, train_rejected = _build_windows(
        timestamps,
        normalized,
        start=config.train_start,
        end=config.train_end,
        steps=steps,
        stride=config.train_stride,
    )
    validation, validation_rejected = _build_windows(
        timestamps,
        normalized,
        start=config.validation_start,
        end=config.validation_end,
        steps=steps,
        stride=config.evaluation_stride,
    )
    test, test_rejected = _build_windows(
        timestamps,
        normalized,
        start=config.test_start,
        end=config.test_end,
        steps=steps,
        stride=config.evaluation_stride,
    )
    arrays = {
        "train_sequences": train,
        "validation_sequences": validation,
        "test_sequences": test,
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "spec": binance_spec(config),
        "spec_hash": _spec_hash(config),
        "sources": sorted(sources, key=lambda value: (value["symbol"], value["url"])),
        "features": [
            f"{symbol}.{feature}" for symbol in config.symbols for feature in _FEATURE_NAMES
        ],
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train"},
        "counts": {
            "aligned_rows": len(common),
            "timestamp_gaps": int(np.count_nonzero(np.diff(timestamps) != _INTERVAL_MS)),
            "rejected_windows": {
                "train": train_rejected,
                "validation": validation_rejected,
                "test": test_rejected,
            },
            "samples": {
                name.removesuffix("_sequences"): int(array.shape[0])
                for name, array in arrays.items()
            },
        },
    }
    return _publish_dataset(config, arrays, manifest)


def _load_directory(
    directory: Path, *, expected_fingerprint: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    dataset_path = directory / "dataset.npz"
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise ValueError(f"processed dataset is incomplete: {directory}")
    manifest = json.loads(manifest_path.read_text())
    fingerprint = manifest.get("fingerprint")
    base = dict(manifest)
    base.pop("fingerprint", None)
    if not isinstance(fingerprint, str) or _manifest_fingerprint(base) != fingerprint:
        raise ValueError(f"processed manifest fingerprint mismatch: {directory}")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError(f"processed dataset fingerprint mismatch: {directory}")
    with np.load(dataset_path, allow_pickle=False) as stored:
        expected_names = {"train_sequences", "validation_sequences", "test_sequences"}
        if set(stored.files) != expected_names:
            raise ValueError(f"processed dataset has unexpected arrays: {directory}")
        arrays = {
            name: np.array(stored[name], dtype=np.float32, copy=True) for name in expected_names
        }
    for name, array in arrays.items():
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError(f"processed array {name} is invalid")
        if _array_hash(array) != manifest["array_hashes"].get(name):
            raise ValueError(f"processed array {name} failed content hash")
    return arrays, manifest


def _candidate_directories(config: BinanceDataConfig) -> list[Path]:
    root = Path(config.processed_root).expanduser().resolve()
    if config.fingerprint is not None:
        return [root / config.fingerprint]
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    expected_spec = _spec_hash(config)
    for path in sorted(root.iterdir()):
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("spec_hash") == expected_spec:
            candidates.append(path)
    return candidates


def load_prepared_binance(config: BinanceDataConfig) -> DatasetBundle:
    validate_data_config(config)
    candidates = _candidate_directories(config)
    if not candidates:
        raise FileNotFoundError(
            "prepared Binance dataset not found; run `uv run jepa prepare-data --config <path>`"
        )
    if len(candidates) > 1:
        fingerprints = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"multiple prepared Binance fingerprints match this config ({fingerprints}); "
            "set data.fingerprint explicitly"
        )
    directory = candidates[0]
    arrays, manifest = _load_directory(directory, expected_fingerprint=config.fingerprint)
    context_length = config.context_steps
    return DatasetBundle(
        train=WindowDataset(
            torch.from_numpy(arrays["train_sequences"]),
            context_length=context_length,
            split="train",
        ),
        validation=WindowDataset(
            torch.from_numpy(arrays["validation_sequences"]),
            context_length=context_length,
            split="validation",
        ),
        test=WindowDataset(
            torch.from_numpy(arrays["test_sequences"]),
            context_length=context_length,
            split="test",
        ),
        fingerprint=manifest["fingerprint"],
        metadata=manifest,
        system=None,
    )
