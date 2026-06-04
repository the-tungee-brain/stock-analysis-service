"""Shared helpers and base class for setup implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trade_planner.indicators import average_true_range
from trade_planner.models import TradeDirection, TradePlan, utc_now
from trade_planner.types import StockData


def long_stop_from_atr(
    stock_data: StockData,
    entry_price: float,
    *,
    atr_period: int,
    atr_multiple: float,
) -> float | None:
    window = stock_data.slice_end(atr_period + 2)
    atr = average_true_range(window, atr_period)
    if atr is None:
        return None
    return entry_price - atr_multiple * atr


def long_target_from_rr(
    entry_price: float,
    stop_price: float,
    risk_reward: float,
) -> float | None:
    risk = entry_price - stop_price
    if risk <= 0:
        return None
    return entry_price + risk_reward * risk


class BaseSetup(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def direction(self) -> TradeDirection: ...

    @abstractmethod
    def required_warmup_bars(self) -> int: ...

    @abstractmethod
    def is_valid(self, stock_data: StockData) -> bool: ...

    @abstractmethod
    def generate_entry(self, stock_data: StockData) -> float | None: ...

    @abstractmethod
    def generate_stop(self, stock_data: StockData, entry_price: float) -> float | None: ...

    @abstractmethod
    def generate_target(
        self,
        stock_data: StockData,
        entry_price: float,
        stop_price: float,
    ) -> float | None: ...

    @abstractmethod
    def confidence_score(self, stock_data: StockData) -> float: ...

    def build_plan(self, stock_data: StockData) -> TradePlan | None:
        if not self.is_valid(stock_data):
            return None
        entry = self.generate_entry(stock_data)
        if entry is None or entry <= 0:
            return None
        stop = self.generate_stop(stock_data, entry)
        if stop is None:
            return None
        target = self.generate_target(stock_data, entry, stop)
        if target is None:
            return None
        rr = TradePlan.calculate_risk_reward(
            direction=self.direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
        )
        return TradePlan(
            symbol=stock_data.symbol,
            setup_name=self.name,
            direction=self.direction,
            entry_price=round(entry, 4),
            stop_price=round(stop, 4),
            target_price=round(target, 4),
            risk_reward=round(rr, 4),
            confidence_score=round(self.confidence_score(stock_data), 2),
            generated_at=utc_now(),
            entry_is_stop=False,
        )
