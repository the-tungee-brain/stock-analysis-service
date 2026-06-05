"""Download daily OHLCV history via yfinance."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
import yfinance as yf

from app.adapters.market.yfinance_bootstrap import configure_yfinance, yfinance_fetch_lock
from data.store import _normalize_ohlcv, append_raw, load_raw, raw_exists, save_raw
from data.symbols import get_symbols

logger = logging.getLogger(__name__)

DEFAULT_YEARS = 15
MAX_DOWNLOAD_RETRIES = 5
RETRY_BASE_DELAY_SEC = 3.0
ZERO_VOLUME_ALLOWED_DOWNLOAD_SYMBOLS = {"^VIX", "VIX"}


def _normalize_downloaded_ohlcv(raw: pd.DataFrame, symbol_upper: str) -> pd.DataFrame:
    return _normalize_ohlcv(
        raw,
        allow_zero_volume=symbol_upper in ZERO_VOLUME_ALLOWED_DOWNLOAD_SYMBOLS,
    )


def _fetch_yahoo_ohlcv(symbol_upper: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """Try yf.download, then Ticker.history (Yahoo often rate-limits after bulk screens)."""
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    configure_yfinance()

    with yfinance_fetch_lock():
        raw = yf.download(
            symbol_upper,
            start=start_s,
            end=end_s,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
    )
    if raw is not None and not raw.empty:
        return _normalize_downloaded_ohlcv(raw, symbol_upper)

    with yfinance_fetch_lock():
        hist = yf.Ticker(symbol_upper).history(start=start_s, end=end_s, auto_adjust=True)
    if hist is not None and not hist.empty:
        return _normalize_downloaded_ohlcv(hist, symbol_upper)
    return None


def download_symbol(symbol: str, *, years: int = DEFAULT_YEARS) -> pd.DataFrame:
    """Download ``years`` of daily OHLCV for one symbol."""
    symbol_upper = symbol.strip().upper()
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25))

    last_err: BaseException | None = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            frame = _fetch_yahoo_ohlcv(symbol_upper, start, end)
            if frame is not None and not frame.empty:
                return frame
        except Exception as exc:
            last_err = exc
            logger.warning(
                "Yahoo OHLCV attempt %d/%d failed for %s: %s",
                attempt,
                MAX_DOWNLOAD_RETRIES,
                symbol_upper,
                exc,
            )
        if attempt < MAX_DOWNLOAD_RETRIES:
            delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            logger.info("Retrying %s in %.0fs", symbol_upper, delay)
            time.sleep(delay)

    raise ValueError(f"No data returned for {symbol_upper}") from last_err


def download_symbol_incremental(
    symbol: str,
    *,
    years: int = DEFAULT_YEARS,
    overlap_days: int = 3,
) -> pd.DataFrame:
    """Fetch only bars after the last stored date (full history if missing)."""
    symbol_upper = symbol.strip().upper()
    if not raw_exists(symbol_upper):
        return download_symbol(symbol_upper, years=years)

    existing = load_raw(symbol_upper)
    last_date = pd.Timestamp(existing.index.max()).normalize()
    end = datetime.now()
    if last_date.date() >= end.date():
        return existing

    start = last_date - timedelta(days=overlap_days)
    configure_yfinance()
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
        return existing
    append_raw(raw, symbol_upper)
    return load_raw(symbol_upper)


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download daily OHLCV Parquet files via yfinance.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to download (default: data.symbols.DEFAULT_SYMBOLS)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help="Years of history to download (default: 15)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbols = args.symbols or get_symbols()
    results = download_and_store_all(symbols, years=args.years)
    for symbol, frame in results.items():
        print(f"Saved {symbol}: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
