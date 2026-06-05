"""Persist and load OHLCV and feature DataFrames as Parquet files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from data.paths import (
    FEATURES_DIR,
    RAW_DIR,
    features_parquet_path,
    raw_parquet_path,
)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
ZERO_VOLUME_ALLOWED_SYMBOLS = {"^VIX", "VIX"}
BENCHMARK_RAW_SYMBOLS = {"SPY", "^VIX", "VIX"}
MIN_BENCHMARK_REPLACEMENT_RATIO = 0.5


def _ensure_parent(path) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)


def save_raw(df: pd.DataFrame, symbol: str) -> Path:
    """Write normalized OHLCV data to ``data/raw/{symbol}.parquet``."""
    from data.paths import raw_parquet_path as _raw_path

    symbol_upper = symbol.strip().upper()
    allow_zero_volume = symbol_upper in ZERO_VOLUME_ALLOWED_SYMBOLS
    out = _normalize_ohlcv(
        df,
        allow_zero_volume=allow_zero_volume,
    )
    _require_non_empty_raw(out, symbol_upper)
    path = _raw_path(symbol_upper)
    _ensure_parent(path)
    if symbol_upper in BENCHMARK_RAW_SYMBOLS and path.exists():
        out = _prepare_benchmark_replacement(
            existing=load_raw(symbol_upper),
            incoming=out,
            symbol=symbol_upper,
            allow_zero_volume=allow_zero_volume,
        )
    _atomic_write_raw(out, path, symbol_upper, allow_zero_volume=allow_zero_volume)
    return path


def raw_exists(symbol: str) -> bool:
    """Return True when ``data/raw/{symbol}.parquet`` exists."""
    return raw_parquet_path(symbol.strip().upper()).exists()


def merge_ohlcv(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    *,
    allow_zero_volume: bool = False,
) -> pd.DataFrame:
    """Append new daily bars and dedupe by date (last wins)."""
    if existing.empty:
        return _normalize_ohlcv(new_rows, allow_zero_volume=allow_zero_volume)
    combined = pd.concat(
        [existing, _normalize_ohlcv(new_rows, allow_zero_volume=allow_zero_volume)]
    )
    return _normalize_ohlcv(combined, allow_zero_volume=allow_zero_volume)


def _require_non_empty_raw(df: pd.DataFrame, symbol: str) -> None:
    if df.empty:
        raise ValueError(f"Refusing to write empty raw OHLCV for {symbol}")


def _latest_index(df: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(df.index.max()).normalize()


def _prepare_benchmark_replacement(
    *,
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    symbol: str,
    allow_zero_volume: bool,
) -> pd.DataFrame:
    _require_non_empty_raw(incoming, symbol)
    if existing.empty:
        return incoming

    incoming_latest = _latest_index(incoming)
    existing_latest = _latest_index(existing)
    if incoming_latest < existing_latest:
        raise ValueError(
            f"Refusing to replace {symbol} benchmark raw data with older data: "
            f"{incoming_latest.date()} < {existing_latest.date()}"
        )

    materially_short = len(incoming) < int(len(existing) * MIN_BENCHMARK_REPLACEMENT_RATIO)
    if materially_short and incoming_latest <= existing_latest:
        raise ValueError(
            f"Refusing to replace {symbol} benchmark raw data with truncated data: "
            f"{len(incoming)} rows < {len(existing)} existing rows"
        )

    merged = merge_ohlcv(
        existing,
        incoming,
        allow_zero_volume=allow_zero_volume,
    )
    _require_non_empty_raw(merged, symbol)
    return merged


def _atomic_write_raw(
    df: pd.DataFrame,
    path: Path,
    symbol: str,
    *,
    allow_zero_volume: bool,
) -> None:
    _require_non_empty_raw(df, symbol)
    tmp_name: str | None = None
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_name = tmp.name
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, index=True)
        validated = _normalize_ohlcv(
            pd.read_parquet(tmp_path),
            allow_zero_volume=allow_zero_volume,
        )
        _require_non_empty_raw(validated, symbol)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_raw(symbol: str) -> pd.DataFrame:
    """Load OHLCV data from ``data/raw/{symbol}.parquet``."""
    symbol_upper = symbol.strip().upper()
    path = raw_parquet_path(symbol_upper)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found for {symbol}: {path}")
    df = pd.read_parquet(path)
    return _normalize_ohlcv(
        df,
        allow_zero_volume=symbol_upper in ZERO_VOLUME_ALLOWED_SYMBOLS,
    )


def save_features(df: pd.DataFrame, symbol: str):
    """Write feature matrix to ``data/features/{symbol}.parquet``."""
    from data.paths import features_parquet_path as _features_path

    symbol_upper = symbol.strip().upper()
    path = _features_path(symbol_upper)
    _ensure_parent(path)
    df.to_parquet(path, index=True)
    return path


def load_features(symbol: str) -> pd.DataFrame:
    """Load features from ``data/features/{symbol}.parquet``."""
    path = features_parquet_path(symbol.strip().upper())
    if not path.exists():
        raise FileNotFoundError(f"Features not found for {symbol}: {path}")
    return pd.read_parquet(path)


def _normalize_ohlcv(df: pd.DataFrame, *, allow_zero_volume: bool = False) -> pd.DataFrame:
    """Ensure lowercase OHLCV columns and a DatetimeIndex named ``date``."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out = out.loc[:, list(OHLCV_COLUMNS)]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=list(OHLCV_COLUMNS))
    price_columns = ["open", "high", "low", "close"]
    out = out[(out.loc[:, price_columns] > 0).all(axis=1)]
    if allow_zero_volume:
        out = out[out["volume"] >= 0]
    else:
        out = out[out["volume"] > 0]
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def append_raw(new_rows: pd.DataFrame, symbol: str) -> Path:
    """Merge ``new_rows`` into existing raw Parquet or create a new file."""
    symbol_upper = symbol.strip().upper()
    allow_zero_volume = symbol_upper in ZERO_VOLUME_ALLOWED_SYMBOLS
    if raw_exists(symbol_upper):
        merged = merge_ohlcv(
            load_raw(symbol_upper),
            new_rows,
            allow_zero_volume=allow_zero_volume,
        )
    else:
        merged = _normalize_ohlcv(new_rows, allow_zero_volume=allow_zero_volume)
    return save_raw(merged, symbol_upper)


def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
