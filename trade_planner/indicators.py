"""Technical indicators — pure functions over bar sequences."""

from __future__ import annotations

import math
from typing import Sequence

from trade_planner.types import OHLCVBar


def closes(bars: Sequence[OHLCVBar]) -> list[float]:
    return [bar.close for bar in bars]


def volumes(bars: Sequence[OHLCVBar]) -> list[float]:
    return [bar.volume for bar in bars]


def simple_moving_average(values: Sequence[float], period: int) -> float | None:
    if period < 1 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def highest_high(bars: Sequence[OHLCVBar], period: int) -> float | None:
    if period < 1 or len(bars) < period:
        return None
    return max(bar.high for bar in bars[-period:])


def lowest_low(bars: Sequence[OHLCVBar], period: int) -> float | None:
    if period < 1 or len(bars) < period:
        return None
    return min(bar.low for bar in bars[-period:])


def average_true_range(bars: Sequence[OHLCVBar], period: int) -> float | None:
    if period < 1 or len(bars) < period + 1:
        return None
    true_ranges: list[float] = []
    for idx in range(1, len(bars)):
        current = bars[idx]
        prior = bars[idx - 1]
        tr = max(
            current.high - current.low,
            abs(current.high - prior.close),
            abs(current.low - prior.close),
        )
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return None
    return sum(true_ranges[-period:]) / period


def trend_strength_pct(bars: Sequence[OHLCVBar], lookback: int) -> float | None:
    """Signed return over lookback ending at last bar."""
    if lookback < 2 or len(bars) < lookback:
        return None
    start_close = bars[-lookback].close
    end_close = bars[-1].close
    if start_close <= 0:
        return None
    return (end_close - start_close) / start_close


def relative_strength(
    symbol_bars: Sequence[OHLCVBar],
    benchmark_bars: Sequence[OHLCVBar],
    lookback: int,
) -> float | None:
    sym = trend_strength_pct(symbol_bars, lookback)
    bench = trend_strength_pct(benchmark_bars, lookback)
    if sym is None or bench is None:
        return None
    return sym - bench


def volume_ratio(bars: Sequence[OHLCVBar], avg_days: int) -> float | None:
    """Alias for today's volume vs trailing average (excluding today)."""
    return volume_expansion_ratio(bars, avg_days)


def volume_expansion_ratio(bars: Sequence[OHLCVBar], avg_days: int) -> float | None:
    if avg_days < 1 or len(bars) < avg_days + 1:
        return None
    vols = volumes(bars)
    avg_vol = sum(vols[-(avg_days + 1) : -1]) / avg_days
    if avg_vol <= 0:
        return None
    return vols[-1] / avg_vol


def close_within_pct_of_period_high(
    bars: Sequence[OHLCVBar],
    *,
    high_lookback_days: int,
    max_distance_pct: float,
) -> bool:
    """True when last close is within ``max_distance_pct`` of the period high."""
    if high_lookback_days < 1 or len(bars) < high_lookback_days:
        return False
    period_high = highest_high(bars[-high_lookback_days:], high_lookback_days)
    if period_high is None or period_high <= 0:
        return False
    close = bars[-1].close
    distance = (period_high - close) / period_high
    return distance <= max_distance_pct


def prior_lowest_low(bars: Sequence[OHLCVBar], days: int) -> float | None:
    """Lowest low of the ``days`` bars immediately before the final bar."""
    if days < 1 or len(bars) < days + 1:
        return None
    return min(bar.low for bar in bars[-(days + 1) : -1])


def percentile_rank(value: float, series: Sequence[float]) -> float | None:
    """Percentile 0–100 of ``value`` within ``series`` (inclusive rank)."""
    if not series:
        return None
    below_or_equal = sum(1 for item in series if item <= value)
    return (below_or_equal / len(series)) * 100.0


def relative_strength_percentile(
    stock_bars: Sequence[OHLCVBar],
    benchmark_bars: Sequence[OHLCVBar],
    *,
    rs_lookback: int,
    percentile_window: int,
) -> float | None:
    """
    Percentile (0–100) of the latest stock-vs-benchmark RS vs trailing RS readings.
    Bars must be aligned and include the evaluation bar as the last element.
    """
    if rs_lookback < 2 or percentile_window < 2:
        return None
    if len(stock_bars) != len(benchmark_bars):
        return None
    if len(stock_bars) < rs_lookback + percentile_window - 1:
        return None

    rs_values: list[float] = []
    for end in range(rs_lookback, len(stock_bars) + 1):
        stock_slice = stock_bars[end - rs_lookback : end]
        bench_slice = benchmark_bars[end - rs_lookback : end]
        rs = relative_strength(stock_slice, bench_slice, rs_lookback)
        if rs is None:
            return None
        rs_values.append(rs)

    if len(rs_values) < percentile_window:
        return None

    window = rs_values[-percentile_window:]
    return percentile_rank(window[-1], window)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_score(value: float, *, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0) * 100.0


def max_drawdown_pct(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
    return max_dd
