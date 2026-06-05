"""Trend and relative-strength context for pattern intelligence."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.benchmarks import BENCHMARK_SYMBOL, VIX_SYMBOL, ensure_benchmark_ohlcv, load_benchmark_ohlcv
from data.loader import load_symbol
from features.build_features import build_features
from features.market_context import attach_market_context


@dataclass(frozen=True)
class TrendContext:
    as_of_date: str
    close: float
    sma_50: float | None
    sma_200: float | None
    above_sma_50: bool | None
    above_sma_200: bool | None
    rs_vs_spy_21d: float | None
    rs_vs_spy_63d: float | None
    rs_vs_spy_126d: float | None
    vol_ratio_20d: float | None
    vol_zscore_20d: float | None

    @property
    def trend_bias(self) -> str:
        if self.above_sma_50 is None or self.above_sma_200 is None:
            return "unknown"
        if self.above_sma_50 and self.above_sma_200:
            return "uptrend"
        if not self.above_sma_50 and not self.above_sma_200:
            return "downtrend"
        return "mixed"


def build_trend_context(symbol: str, raw: pd.DataFrame | None = None) -> TrendContext:
    """Build latest-bar trend context for a symbol."""
    symbol_upper = symbol.strip().upper()
    ohlcv = raw if raw is not None else load_symbol(symbol_upper)
    ensure_benchmark_ohlcv()

    features = build_features(ohlcv)
    spy_close = load_benchmark_ohlcv(BENCHMARK_SYMBOL)["close"]
    vix_close = load_benchmark_ohlcv(VIX_SYMBOL)["close"]
    features = attach_market_context(
        features,
        stock_close=ohlcv["close"],
        spy_close=spy_close,
        vix_close=vix_close,
    )

    latest = features.iloc[-1]
    as_of = pd.Timestamp(features.index[-1]).strftime("%Y-%m-%d")
    close = float(ohlcv["close"].iloc[-1])

    sma_50 = _optional_float(latest, "sma_50")
    sma_200 = _optional_float(latest, "sma_200")

    return TrendContext(
        as_of_date=as_of,
        close=close,
        sma_50=sma_50,
        sma_200=sma_200,
        above_sma_50=(close > sma_50) if sma_50 is not None else None,
        above_sma_200=(close > sma_200) if sma_200 is not None else None,
        rs_vs_spy_21d=_optional_float(latest, "rs_vs_spy_21d"),
        rs_vs_spy_63d=_optional_float(latest, "rs_vs_spy_63d"),
        rs_vs_spy_126d=_optional_float(latest, "rs_vs_spy_126d"),
        vol_ratio_20d=_optional_float(latest, "vol_ratio_20d"),
        vol_zscore_20d=_optional_float(latest, "vol_zscore_20d"),
    )


def _optional_float(row: pd.Series, key: str) -> float | None:
    if key not in row.index:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    return float(value)
