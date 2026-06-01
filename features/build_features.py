"""Combine price, indicator, and pattern features into a modeling matrix."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import pandas as pd

from data.loader import load_symbol
from data.store import save_features
from data.symbols import get_symbols
from features.indicators import compute_indicators
from features.patterns import compute_patterns

MOMENTUM_HORIZONS: tuple[int, ...] = (21, 63, 126, 252)
# Longest lookback in the per-symbol feature set (252d momentum).
FEATURE_WARMUP_DAYS = 252


def compute_momentum_features(close: pd.Series) -> pd.DataFrame:
    """Multi-horizon price momentum."""
    out = pd.DataFrame(index=close.index)
    for horizon in MOMENTUM_HORIZONS:
        out[f"ret_{horizon}d"] = close.pct_change(horizon)
    return out


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume level, z-score, and short-term trend metrics."""
    volume = df["volume"].astype("float64")
    out = pd.DataFrame(index=df.index)
    vol_ma20 = volume.rolling(20).mean()
    vol_std20 = volume.rolling(20).std()
    vol_ma5 = volume.rolling(5).mean()

    out["vol_ratio_20d"] = volume / vol_ma20.replace(0, np.nan)
    safe_std = vol_std20.replace(0, np.nan)
    out["vol_zscore_20d"] = (volume - vol_ma20) / safe_std
    out["vol_trend_5d_20d"] = vol_ma5 / vol_ma20.replace(0, np.nan)
    out["vol_chg_5d"] = volume.pct_change(5)
    return out


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
    out = pd.concat([out, compute_momentum_features(close)], axis=1)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full daily feature matrix indexed by date."""
    indicators = compute_indicators(df)
    price = compute_price_features(df, indicators=indicators)
    volume = compute_volume_features(df)
    patterns = compute_patterns(df)

    features = pd.concat([price, volume, indicators, patterns], axis=1)
    features.index.name = "date"
    return features.sort_index()


def build_and_save_features(symbol: str) -> pd.DataFrame:
    """Load raw OHLCV, compute features, and write ``data/features/{symbol}.parquet``."""
    raw = load_symbol(symbol)
    features = build_features(raw)
    save_features(features, symbol)
    return features


def build_and_save_all(symbols: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    """Build and persist feature Parquets for each symbol."""
    tickers = list(symbols) if symbols else get_symbols()
    out: dict[str, pd.DataFrame] = {}
    for symbol in tickers:
        features = build_and_save_features(symbol)
        out[symbol.strip().upper()] = features
        print(f"Saved features for {symbol.strip().upper()}: {len(features)} rows")
    return out


def features_ready_slice(features: pd.DataFrame) -> pd.DataFrame:
    """Return rows after the indicator warm-up window."""
    if len(features) <= FEATURE_WARMUP_DAYS:
        return features.iloc[0:0]
    return features.iloc[FEATURE_WARMUP_DAYS:]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build feature Parquet files from raw OHLCV.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to build (default: data.symbols.DEFAULT_SYMBOLS)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    build_and_save_all(args.symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
