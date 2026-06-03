"""Parquet persistence for ranking feature matrices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.paths import RANKING_FEATURES_DIR, ranking_features_parquet_path
from ranking_pipeline.datetime_utils import to_naive_utc_index


def ranking_features_exists(symbol: str) -> bool:
    return ranking_features_parquet_path(symbol).exists()


def load_ranking_features(symbol: str) -> pd.DataFrame:
    path = ranking_features_parquet_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Ranking features not found: {path}")
    df = pd.read_parquet(path)
    df.index = to_naive_utc_index(df.index)
    df.index.name = "date"
    return df.sort_index()


def save_ranking_features(df: pd.DataFrame, symbol: str) -> Path:
    symbol_upper = symbol.strip().upper()
    path = ranking_features_parquet_path(symbol_upper)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = to_naive_utc_index(out.index)
    out.index.name = "date"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_parquet(path, index=True)
    return path


def merge_ranking_features(existing: pd.DataFrame, updated: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, updated])
    combined.index = to_naive_utc_index(combined.index)
    combined = combined.sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def ensure_ranking_features_dir() -> None:
    RANKING_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
