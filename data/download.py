"""Download daily OHLCV history via yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from app.adapters.market.yfinance_bootstrap import configure_yfinance, yfinance_fetch_lock
from data.store import _normalize_ohlcv, save_raw
from data.symbols import get_symbols

DEFAULT_YEARS = 15


def download_symbol(symbol: str, *, years: int = DEFAULT_YEARS) -> pd.DataFrame:
    """Download ``years`` of daily OHLCV for one symbol."""
    symbol_upper = symbol.strip().upper()

    configure_yfinance()
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25))

    with yfinance_fetch_lock():
        raw = yf.download(
            symbol_upper,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {symbol_upper}")

    return _normalize_ohlcv(raw)


def download_and_store_symbol(symbol: str, *, years: int = DEFAULT_YEARS):
    """Download OHLCV and persist to ``data/raw/{symbol}.parquet``."""
    df = download_symbol(symbol, years=years)
    return save_raw(df, symbol), df


def download_and_store_all(
    symbols: list[str] | None = None,
    *,
    years: int = DEFAULT_YEARS,
) -> dict[str, pd.DataFrame]:
    """Download and store OHLCV for every symbol in the universe."""
    tickers = symbols or get_symbols()
    out: dict[str, pd.DataFrame] = {}
    for symbol in tickers:
        _, df = download_and_store_symbol(symbol, years=years)
        out[symbol.strip().upper()] = df
    return out
