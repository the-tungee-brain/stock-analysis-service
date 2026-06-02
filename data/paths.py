"""Filesystem paths for raw and feature Parquet files."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"


def raw_parquet_path(symbol: str) -> Path:
    return RAW_DIR / f"{symbol.strip().upper()}.parquet"


def features_parquet_path(symbol: str) -> Path:
    return FEATURES_DIR / f"{symbol.strip().upper()}.parquet"
