"""Market regime classification from benchmark price action."""

from __future__ import annotations

from typing import Sequence

from trade_planner.indicators import simple_moving_average, trend_strength_pct
from trade_planner.research.models import MarketRegime
from trade_planner.types import OHLCVBar

# Benchmark SMA windows for regime (SPY-style index).
_REGIME_SMA_FAST = 50
_REGIME_SMA_SLOW = 200
_REGIME_TREND_LOOKBACK = 20
_RISK_OFF_TREND_THRESHOLD = -0.03


def classify_market_regime(
    benchmark_bars: Sequence[OHLCVBar],
    *,
    index: int,
) -> MarketRegime:
    """
    Classify regime at ``index`` using only benchmark history up to that bar.

    RISK_ON: close > SMA50 > SMA200 and 20-day trend non-negative.
    RISK_OFF: close below SMA200 or 20-day trend below -3%.
    NEUTRAL: otherwise.
    """
    if index < 0 or index >= len(benchmark_bars):
        return MarketRegime.NEUTRAL

    window = benchmark_bars[: index + 1]
    closes = [bar.close for bar in window]
    close = closes[-1]
    sma_fast = simple_moving_average(closes, _REGIME_SMA_FAST)
    sma_slow = simple_moving_average(closes, _REGIME_SMA_SLOW)
    trend = trend_strength_pct(window, _REGIME_TREND_LOOKBACK)

    if sma_slow is not None and close < sma_slow:
        return MarketRegime.RISK_OFF
    if trend is not None and trend < _RISK_OFF_TREND_THRESHOLD:
        return MarketRegime.RISK_OFF
    if (
        sma_fast is not None
        and sma_slow is not None
        and close > sma_fast > sma_slow
        and (trend is None or trend >= 0.0)
    ):
        return MarketRegime.RISK_ON
    return MarketRegime.NEUTRAL
