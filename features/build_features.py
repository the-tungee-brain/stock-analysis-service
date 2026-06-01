"""Combine price, indicator, and pattern features into a modeling matrix."""

from __future__ import annotations

import pandas as pd

from data.loader import load_symbol
from data.store import save_features
from features.indicators import compute_indicators
from features.patterns import compute_patterns

# Longest lookback in the feature set (SMA 200).
FEATURE_WARMUP_DAYS = 200


def compute_price_features(
    df: pd.DataFrame,
    indicators: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Price-based returns, volatility, and position vs moving averages."""
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["vol_20d"] = out["ret_1d"].rolling(20).std()

    ind = indicators if indicators is not None else compute_indicators(df)
    out["close_vs_sma20"] = close / ind["sma_20"] - 1.0
    out["close_vs_sma200"] = close / ind["sma_200"] - 1.0
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full daily feature matrix indexed by date."""
    indicators = compute_indicators(df)
    price = compute_price_features(df, indicators=indicators)
    patterns = compute_patterns(df)

    features = pd.concat([price, indicators, patterns], axis=1)
    features.index.name = "date"
    return features.sort_index()


def build_and_save_features(symbol: str) -> pd.DataFrame:
    """Load raw OHLCV, compute features, and write ``data/features/{symbol}.parquet``."""
    raw = load_symbol(symbol)
    features = build_features(raw)
    save_features(features, symbol)
    return features


def features_ready_slice(features: pd.DataFrame) -> pd.DataFrame:
    """Return rows after the indicator warm-up window."""
    if len(features) <= FEATURE_WARMUP_DAYS:
        return features.iloc[0:0]
    return features.iloc[FEATURE_WARMUP_DAYS:]
