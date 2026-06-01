"""Benchmark series (SPY, VIX) used for excess-return labels and market features."""

from __future__ import annotations

import pandas as pd

from data.download import DEFAULT_YEARS, download_and_store_symbol
from data.loader import load_symbol

BENCHMARK_SYMBOL = "SPY"
VIX_SYMBOL = "^VIX"
BENCHMARK_SYMBOLS: tuple[str, ...] = (BENCHMARK_SYMBOL, VIX_SYMBOL)


def ensure_benchmark_ohlcv(*, years: int = DEFAULT_YEARS) -> None:
    """Download and store benchmark OHLCV when missing locally."""
    for symbol in BENCHMARK_SYMBOLS:
        try:
            load_symbol(symbol)
        except FileNotFoundError:
            download_and_store_symbol(symbol, years=years)


def load_benchmark_close(symbol: str) -> pd.Series:
    """Return the daily close series for a benchmark symbol."""
    return load_symbol(symbol.strip().upper())["close"]
