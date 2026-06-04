"""Pullback to short-term MA within an established uptrend."""

from __future__ import annotations

from trade_planner.config import PullbackConfig, TradePlannerConfig
from trade_planner.indicators import closes, simple_moving_average, trend_strength_pct
from trade_planner.models import TradeDirection
from trade_planner.setups.base import BaseSetup, long_stop_from_atr, long_target_from_rr
from trade_planner.types import StockData


class PullbackSetup(BaseSetup):
    name = "Pullback"
    direction: TradeDirection = "LONG"

    def __init__(self, config: PullbackConfig | None = None) -> None:
        self._config = config or TradePlannerConfig().pullback

    def required_warmup_bars(self) -> int:
        return max(
            self._config.trend_ma_days,
            self._config.pullback_ma_days,
            self._config.atr_period + 2,
        ) + 1

    def is_valid(self, stock_data: StockData) -> bool:
        cfg = self._config
        window = stock_data.slice_end(cfg.trend_ma_days + 1)
        if len(window) < self.required_warmup_bars():
            return False

        price_series = closes(window)
        slow_ma = simple_moving_average(price_series, cfg.trend_ma_days)
        fast_ma = simple_moving_average(price_series, cfg.pullback_ma_days)
        trend = trend_strength_pct(window, cfg.trend_ma_days)
        if slow_ma is None or fast_ma is None or trend is None:
            return False

        current_close = stock_data.current.close
        if trend <= 0 or current_close <= slow_ma:
            return False

        pullback_depth = (fast_ma - current_close) / fast_ma
        return cfg.min_pullback_pct <= pullback_depth <= cfg.max_pullback_pct

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
        window = stock_data.slice_end(cfg.trend_ma_days + 1)
        trend = trend_strength_pct(window, cfg.trend_ma_days) or 0.0
        return min(100.0, max(0.0, trend * 500.0))
