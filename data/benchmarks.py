"""Benchmark series (SPY, VIX) used for excess-return labels and market features."""

from __future__ import annotations

import pandas as pd

from data.download import DEFAULT_YEARS, download_and_store_symbol
from data.loader import load_symbol
from data.paths import raw_parquet_path

BENCHMARK_SYMBOL = "SPY"
VIX_SYMBOL = "^VIX"
BENCHMARK_SYMBOLS: tuple[str, ...] = (BENCHMARK_SYMBOL, VIX_SYMBOL)
VIX_SYMBOL_ALIASES: tuple[str, ...] = (VIX_SYMBOL, "VIX")


def benchmark_symbol_candidates(symbol: str) -> tuple[str, ...]:
    """Return local raw-data names accepted for a benchmark symbol."""
    symbol_upper = symbol.strip().upper()
    if symbol_upper in VIX_SYMBOL_ALIASES:
        return VIX_SYMBOL_ALIASES
    return (symbol_upper,)


def benchmark_raw_paths(symbol: str) -> tuple[str, ...]:
    """Return raw Parquet paths checked for a benchmark symbol."""
    return tuple(
        str(raw_parquet_path(candidate))
        for candidate in benchmark_symbol_candidates(symbol)
    )


def load_benchmark_ohlcv(symbol: str) -> pd.DataFrame:
    """Load benchmark OHLCV, accepting known local naming aliases."""
    errors: list[Exception] = []
    for candidate in benchmark_symbol_candidates(symbol):
        try:
            frame = load_symbol(candidate)
        except FileNotFoundError as exc:
            errors.append(exc)
            continue
        if not frame.empty:
            return frame
        errors.append(ValueError(f"Raw data is empty for {candidate}"))

    symbol_upper = symbol.strip().upper()
    paths = ", ".join(benchmark_raw_paths(symbol_upper))
    detail = f"Expected one of: {paths}" if paths else "No benchmark path candidates"
    if errors:
        raise FileNotFoundError(
            f"Benchmark OHLCV missing or empty for {symbol_upper}. {detail}"
        ) from errors[-1]
    raise FileNotFoundError(f"Benchmark OHLCV missing or empty for {symbol_upper}. {detail}")


def ensure_benchmark_ohlcv(*, years: int = DEFAULT_YEARS) -> None:
    """Download and store benchmark OHLCV when missing locally."""
    for symbol in BENCHMARK_SYMBOLS:
        try:
            load_benchmark_ohlcv(symbol)
        except (FileNotFoundError, ValueError):
            download_and_store_symbol(symbol, years=years)


def load_benchmark_close(symbol: str) -> pd.Series:
    """Return the daily close series for a benchmark symbol."""
    return load_benchmark_ohlcv(symbol)["close"]
