"""Momentum breakout feature snapshots at signal time."""

from __future__ import annotations

from typing import Sequence

from trade_planner.config import MomentumBreakoutConfig
from trade_planner.indicators import (
    close_within_pct_of_period_high,
    highest_high,
    relative_strength_percentile,
    simple_moving_average,
    volume_ratio,
)
from trade_planner.research.regime import classify_market_regime
from trade_planner.research.models import FeatureSnapshot, MarketRegime
from trade_planner.types import OHLCVBar


def _distance_to_period_high(bars: Sequence[OHLCVBar], lookback: int) -> float | None:
    if lookback < 1 or len(bars) < lookback:
        return None
    period_high = highest_high(bars[-lookback:], lookback)
    if period_high is None or period_high <= 0:
        return None
    close = bars[-1].close
    return (period_high - close) / period_high


def capture_momentum_feature_snapshot(
    stock_bars: Sequence[OHLCVBar],
    benchmark_bars: Sequence[OHLCVBar] | None,
    *,
    signal_index: int,
    config: MomentumBreakoutConfig,
) -> FeatureSnapshot:
    end = signal_index + 1
    warmup = (
        max(
            config.sma_slow_days,
            config.high_lookback_days,
            config.volume_avg_days + 1,
            config.rs_lookback_days + config.rs_percentile_window,
        )
        + 1
    )
    start = max(0, end - warmup)
    window = stock_bars[start:end]
    bench_window = None
    if benchmark_bars is not None and len(benchmark_bars) >= end:
        bench_window = benchmark_bars[start:end]

    price_series = [bar.close for bar in window]
    sma_fast = simple_moving_average(price_series, config.sma_fast_days)
    sma_slow = simple_moving_average(price_series, config.sma_slow_days)
    close = window[-1].close

    close_vs_sma50 = None
    close_vs_sma200 = None
    if sma_fast and sma_fast > 0:
        close_vs_sma50 = (close / sma_fast) - 1.0
    if sma_slow and sma_slow > 0:
        close_vs_sma200 = (close / sma_slow) - 1.0

    rs_pct = None
    if bench_window is not None and len(bench_window) == len(window):
        rs_pct = relative_strength_percentile(
            window,
            bench_window,
            rs_lookback=config.rs_lookback_days,
            percentile_window=config.rs_percentile_window,
        )

    vol = volume_ratio(window, config.volume_avg_days)
    dist_high = _distance_to_period_high(window, config.high_lookback_days)
    if dist_high is None and close_within_pct_of_period_high(
        window,
        high_lookback_days=config.high_lookback_days,
        max_distance_pct=config.high_proximity_pct,
    ):
        dist_high = 0.0

    regime = MarketRegime.NEUTRAL
    if bench_window is not None:
        regime = classify_market_regime(bench_window, index=len(bench_window) - 1)

    return FeatureSnapshot(
        rs_percentile=round(rs_pct, 2) if rs_pct is not None else None,
        volume_ratio=round(vol, 4) if vol is not None else None,
        close_vs_sma50=round(close_vs_sma50, 6) if close_vs_sma50 is not None else None,
        close_vs_sma200=round(close_vs_sma200, 6) if close_vs_sma200 is not None else None,
        distance_to_20d_high=round(dist_high, 6) if dist_high is not None else None,
        market_regime=regime,
    )
