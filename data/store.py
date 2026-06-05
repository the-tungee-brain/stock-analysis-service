"""Persist and load OHLCV and feature DataFrames as Parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.paths import (
    FEATURES_DIR,
    RAW_DIR,
    features_parquet_path,
    raw_parquet_path,
)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _ensure_parent(path) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)


def save_raw(df: pd.DataFrame, symbol: str) -> Path:
    """Write normalized OHLCV data to ``data/raw/{symbol}.parquet``."""
    from data.paths import raw_parquet_path as _raw_path

    symbol_upper = symbol.strip().upper()
    out = _normalize_ohlcv(df)
    path = _raw_path(symbol_upper)
    _ensure_parent(path)
    out.to_parquet(path, index=True)
    return path


def raw_exists(symbol: str) -> bool:
    """Return True when ``data/raw/{symbol}.parquet`` exists."""
    return raw_parquet_path(symbol.strip().upper()).exists()


def merge_ohlcv(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new daily bars and dedupe by date (last wins)."""
    if existing.empty:
        return _normalize_ohlcv(new_rows)
    combined = pd.concat([existing, _normalize_ohlcv(new_rows)])
    return _normalize_ohlcv(combined)


def load_raw(symbol: str) -> pd.DataFrame:
    """Load OHLCV data from ``data/raw/{symbol}.parquet``."""
    path = raw_parquet_path(symbol.strip().upper())
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found for {symbol}: {path}")
    df = pd.read_parquet(path)
    return _normalize_ohlcv(df)


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


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
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
    out = out[(out.loc[:, list(OHLCV_COLUMNS)] > 0).all(axis=1)]
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def append_raw(new_rows: pd.DataFrame, symbol: str) -> Path:
    """Merge ``new_rows`` into existing raw Parquet or create a new file."""
    symbol_upper = symbol.strip().upper()
    if raw_exists(symbol_upper):
        merged = merge_ohlcv(load_raw(symbol_upper), new_rows)
    else:
        merged = _normalize_ohlcv(new_rows)
    return save_raw(merged, symbol_upper)


def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
