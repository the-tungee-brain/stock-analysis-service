"""Momentum breakout — trend + RS + volume + proximity to highs, buy-stop entry."""

from __future__ import annotations

from dataclasses import replace

from trade_planner.config import MomentumBreakoutConfig, TradePlannerConfig
from trade_planner.indicators import (
    close_within_pct_of_period_high,
    prior_lowest_low,
    relative_strength_percentile,
    simple_moving_average,
    volume_ratio,
)
from trade_planner.models import TradeDirection, TradePlan, utc_now
from trade_planner.setups.base import BaseSetup, long_target_from_rr
from trade_planner.types import StockData


class MomentumBreakoutSetup(BaseSetup):
    name = "Momentum Breakout"
    direction: TradeDirection = "LONG"

    def __init__(self, config: MomentumBreakoutConfig | None = None) -> None:
        self._config = config or TradePlannerConfig().momentum

    def required_warmup_bars(self) -> int:
        cfg = self._config
        return (
            max(
                cfg.sma_slow_days,
                cfg.high_lookback_days,
                cfg.volume_avg_days + 1,
                cfg.stop_lookback_days + 1,
                cfg.rs_lookback_days + cfg.rs_percentile_window,
            )
            + 1
        )

    def _evaluation_window(self, stock_data: StockData) -> tuple[StockData, ...] | None:
        end = stock_data.index + 1
        start = max(0, end - self.required_warmup_bars())
        stock_slice = stock_data.bars[start:end]
        if len(stock_slice) < self.required_warmup_bars():
            return None
        return stock_slice

    def _benchmark_window(self, stock_data: StockData) -> tuple | None:
        if stock_data.benchmark_bars is None:
            return None
        end = stock_data.index + 1
        start = max(0, end - self.required_warmup_bars())
        return stock_data.benchmark_bars[start:end]

    def is_valid(self, stock_data: StockData) -> bool:
        cfg = self._config
        window = self._evaluation_window(stock_data)
        if window is None:
            return False

        price_series = [bar.close for bar in window]
        sma_fast = simple_moving_average(price_series, cfg.sma_fast_days)
        sma_slow = simple_moving_average(price_series, cfg.sma_slow_days)
        if sma_fast is None or sma_slow is None:
            return False

        current_close = window[-1].close
        if current_close <= sma_fast or sma_fast <= sma_slow:
            return False

        if not close_within_pct_of_period_high(
            window,
            high_lookback_days=cfg.high_lookback_days,
            max_distance_pct=cfg.high_proximity_pct,
        ):
            return False

        vol = volume_ratio(window, cfg.volume_avg_days)
        if vol is None or vol < cfg.volume_ratio_min:
            return False

        bench_window = self._benchmark_window(stock_data)
        if cfg.require_benchmark:
            if bench_window is None:
                return False
            rs_pct = relative_strength_percentile(
                window,
                bench_window,
                rs_lookback=cfg.rs_lookback_days,
                percentile_window=cfg.rs_percentile_window,
            )
            if rs_pct is None or rs_pct < cfg.rs_percentile_min:
                return False
        elif bench_window is not None:
            rs_pct = relative_strength_percentile(
                window,
                bench_window,
                rs_lookback=cfg.rs_lookback_days,
                percentile_window=cfg.rs_percentile_window,
            )
            if rs_pct is not None and rs_pct < cfg.rs_percentile_min:
                return False

        return True

    def generate_entry(self, stock_data: StockData) -> float | None:
        if not self.is_valid(stock_data):
            return None
        return round(stock_data.current.high + self._config.entry_buffer, 4)

    def generate_stop(self, stock_data: StockData, entry_price: float) -> float | None:
        _ = entry_price
        window = self._evaluation_window(stock_data)
        if window is None:
            return None
        stop = prior_lowest_low(window, self._config.stop_lookback_days)
        if stop is None or stop <= 0:
            return None
        return round(stop, 4)

    def generate_target(
        self,
        stock_data: StockData,
        entry_price: float,
        stop_price: float,
    ) -> float | None:
        _ = stock_data
        target = long_target_from_rr(
            entry_price,
            stop_price,
            self._config.target_risk_reward,
        )
        if target is None:
            return None
        return round(target, 4)

    def confidence_score(self, stock_data: StockData) -> float:
        if not self.is_valid(stock_data):
            return 0.0
        cfg = self._config
        window = self._evaluation_window(stock_data)
        if window is None:
            return 0.0

        vol = volume_ratio(window, cfg.volume_avg_days) or 1.0
        vol_score = min(40.0, (vol / cfg.volume_ratio_min) * 40.0)

        bench = self._benchmark_window(stock_data)
        rs_score = 30.0
        if bench is not None:
            rs_pct = relative_strength_percentile(
                window,
                bench,
                rs_lookback=cfg.rs_lookback_days,
                percentile_window=cfg.rs_percentile_window,
            )
            if rs_pct is not None:
                rs_score = min(40.0, (rs_pct / 100.0) * 40.0)

        price_series = [bar.close for bar in window]
        sma_fast = simple_moving_average(price_series, cfg.sma_fast_days) or 0.0
        sma_slow = simple_moving_average(price_series, cfg.sma_slow_days) or 1.0
        trend_score = min(20.0, ((sma_fast - sma_slow) / sma_slow) * 500.0) if sma_slow else 0.0

        return round(min(100.0, vol_score + rs_score + trend_score), 2)

    def build_plan(self, stock_data: StockData) -> TradePlan | None:
        plan = super().build_plan(stock_data)
        if plan is None:
            return None
        return replace(plan, entry_is_stop=True)
