"""Trend continuation when fast MA stays above slow MA with supportive volume."""

from __future__ import annotations

from trade_planner.config import TradePlannerConfig, TrendContinuationConfig
from trade_planner.indicators import (
    closes,
    simple_moving_average,
    trend_strength_pct,
    volume_expansion_ratio,
)
from trade_planner.models import TradeDirection
from trade_planner.setups.base import BaseSetup, long_stop_from_atr, long_target_from_rr
from trade_planner.types import StockData


class TrendContinuationSetup(BaseSetup):
    name = "Trend Continuation"
    direction: TradeDirection = "LONG"

    def __init__(self, config: TrendContinuationConfig | None = None) -> None:
        self._config = config or TradePlannerConfig().trend_continuation

    def required_warmup_bars(self) -> int:
        return max(
            self._config.slow_ma_days,
            self._config.volume_avg_days,
            self._config.atr_period + 2,
        ) + 1

    def is_valid(self, stock_data: StockData) -> bool:
        cfg = self._config
        window = stock_data.slice_end(cfg.slow_ma_days + 1)
        if len(window) < self.required_warmup_bars():
            return False

        price_series = closes(window)
        fast_ma = simple_moving_average(price_series, cfg.fast_ma_days)
        slow_ma = simple_moving_average(price_series, cfg.slow_ma_days)
        vol_ratio = volume_expansion_ratio(window, cfg.volume_avg_days)
        if fast_ma is None or slow_ma is None or vol_ratio is None:
            return False

        spread = (fast_ma - slow_ma) / slow_ma if slow_ma > 0 else 0.0
        current = stock_data.current
        return (
            spread >= cfg.min_ma_spread_pct
            and current.close > fast_ma
            and current.close >= current.open
            and vol_ratio >= cfg.volume_expansion_ratio
        )

    def generate_entry(self, stock_data: StockData) -> float | None:
        if not self.is_valid(stock_data):
            return None
        return stock_data.current.close

    def generate_stop(self, stock_data: StockData, entry_price: float) -> float | None:
        cfg = self._config
        return long_stop_from_atr(
            stock_data,
            entry_price,
            atr_period=cfg.atr_period,
            atr_multiple=cfg.stop_atr_multiple,
        )

    def generate_target(
        self,
        stock_data: StockData,
        entry_price: float,
        stop_price: float,
    ) -> float | None:
        _ = stock_data
        return long_target_from_rr(
            entry_price,
            stop_price,
            self._config.target_risk_reward,
        )

    def confidence_score(self, stock_data: StockData) -> float:
        cfg = self._config
        window = stock_data.slice_end(cfg.slow_ma_days + 1)
        trend = trend_strength_pct(window, cfg.slow_ma_days) or 0.0
        vol_ratio = volume_expansion_ratio(window, cfg.volume_avg_days) or 1.0
        trend_score = min(60.0, max(0.0, trend * 400.0))
        vol_score = min(40.0, (vol_ratio / cfg.volume_expansion_ratio) * 40.0)
        return min(100.0, trend_score + vol_score)
