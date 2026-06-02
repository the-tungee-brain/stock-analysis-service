"""Feature row helpers for signal change detection."""

from __future__ import annotations

import pandas as pd

from data.benchmarks import BENCHMARK_SYMBOL, VIX_SYMBOL, ensure_benchmark_ohlcv
from data.loader import load_symbol
from features.build_features import build_features
from features.build_features import features_ready_slice
from features.market_context import attach_market_context
from models.prediction_service import ensure_raw_ohlcv


def build_feature_history(symbol: str, *, lookback: int = 2) -> pd.DataFrame:
    """Return the last ``lookback`` feature-ready rows for a symbol."""
    symbol_upper = symbol.strip().upper()
    raw = ensure_raw_ohlcv(symbol_upper)
    ensure_benchmark_ohlcv()
    features = build_features(raw)
    spy_close = load_symbol(BENCHMARK_SYMBOL)["close"]
    vix_close = load_symbol(VIX_SYMBOL)["close"]
    features = attach_market_context(
        features,
        stock_close=raw["close"],
        spy_close=spy_close,
        vix_close=vix_close,
    )
    ready = features_ready_slice(features)
    if ready.empty:
        raise ValueError(f"Insufficient history to build features for {symbol_upper}")
    return ready.iloc[-lookback:]
