"""OHLCV loading and date filtering for research."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd

from trade_planner.types import OHLCVBar


def find_bar_index(bars: Sequence[OHLCVBar], trading_date: date) -> int | None:
    for idx, bar in enumerate(bars):
        if bar.trading_date == trading_date:
            return idx
    return None


def filter_bars_through(
    bars: Sequence[OHLCVBar], *, end_date: date
) -> tuple[OHLCVBar, ...]:
    return tuple(bar for bar in bars if bar.trading_date <= end_date)


def filter_bars_in_range(
    bars: Sequence[OHLCVBar],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[OHLCVBar, ...]:
    out: list[OHLCVBar] = []
    for bar in bars:
        if start_date is not None and bar.trading_date < start_date:
            continue
        if end_date is not None and bar.trading_date > end_date:
            continue
        out.append(bar)
    return tuple(out)


def align_benchmark_to_stock(
    stock_bars: Sequence[OHLCVBar],
    benchmark_bars: Sequence[OHLCVBar],
) -> tuple[OHLCVBar, ...]:
    """Reindex benchmark closes to stock trading dates (forward-fill missing)."""
    by_date = {bar.trading_date: bar for bar in benchmark_bars}
    if not by_date:
        return ()
    dates_sorted = sorted(by_date)
    aligned: list[OHLCVBar] = []
    last: OHLCVBar | None = None
    for bar in stock_bars:
        if bar.trading_date in by_date:
            last = by_date[bar.trading_date]
        elif last is None:
            # find latest benchmark date <= stock date
            prior_dates = [d for d in dates_sorted if d <= bar.trading_date]
            if not prior_dates:
                continue
            last = by_date[prior_dates[-1]]
        aligned.append(
            OHLCVBar(
                trading_date=bar.trading_date,
                open=last.open,
                high=last.high,
                low=last.low,
                close=last.close,
                volume=last.volume,
            )
        )
    return tuple(aligned)


def align_stock_and_benchmark(
    stock_bars: Sequence[OHLCVBar],
    benchmark_bars: Sequence[OHLCVBar],
) -> tuple[tuple[OHLCVBar, ...], tuple[OHLCVBar, ...]]:
    """Return stock and benchmark bars aligned 1:1 by stock trading date.

    Benchmark bars are forward-filled to stock dates. Stock bars before the
    first available benchmark bar are dropped, avoiding look-ahead benchmark
    backfills and ensuring StockData receives equal-length sequences.
    """
    by_date = {bar.trading_date: bar for bar in benchmark_bars}
    if not by_date:
        return (), ()
    dates_sorted = sorted(by_date)
    first_benchmark_date = dates_sorted[0]

    aligned_stock: list[OHLCVBar] = []
    aligned_benchmark: list[OHLCVBar] = []
    last: OHLCVBar | None = None
    for stock_bar in stock_bars:
        if stock_bar.trading_date < first_benchmark_date:
            continue
        if stock_bar.trading_date in by_date:
            last = by_date[stock_bar.trading_date]
        elif last is None:
            prior_dates = [d for d in dates_sorted if d <= stock_bar.trading_date]
            if not prior_dates:
                continue
            last = by_date[prior_dates[-1]]

        aligned_stock.append(stock_bar)
        aligned_benchmark.append(
            OHLCVBar(
                trading_date=stock_bar.trading_date,
                open=last.open,
                high=last.high,
                low=last.low,
                close=last.close,
                volume=last.volume,
            )
        )

    return tuple(aligned_stock), tuple(aligned_benchmark)


def ohlcv_bars_from_dataframe(df: pd.DataFrame) -> tuple[OHLCVBar, ...]:
    """Convert a normalized OHLCV DataFrame (DatetimeIndex) to bars."""
    if df.empty:
        return ()
    work = df.sort_index()
    bars: list[OHLCVBar] = []
    for ts, row in work.iterrows():
        trading_date = ts.date() if hasattr(ts, "date") else ts
        bars.append(
            OHLCVBar(
                trading_date=trading_date,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return tuple(bars)
