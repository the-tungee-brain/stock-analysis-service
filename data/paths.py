"""Filesystem paths for raw and feature Parquet files."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RANKING_DIR = PROJECT_ROOT / "data" / "ranking"
RANKING_FEATURES_DIR = RANKING_DIR / "features"
RANKING_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "ranking_model"
DEFAULT_RANKING_DB_PATH = RANKING_DIR / "ranking_pipeline.db"


def raw_parquet_path(symbol: str) -> Path:
    return RAW_DIR / f"{symbol.strip().upper()}.parquet"


def features_parquet_path(symbol: str) -> Path:
    return FEATURES_DIR / f"{symbol.strip().upper()}.parquet"


def ranking_features_parquet_path(symbol: str) -> Path:
    return RANKING_FEATURES_DIR / f"{symbol.strip().upper()}.parquet"


def active_universe_pointer_path() -> Path:
    return RANKING_DIR / "active_universe.json"
