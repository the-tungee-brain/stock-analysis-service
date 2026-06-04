"""Setup protocol — all trade setups implement this contract."""

from __future__ import annotations

from typing import Protocol

from trade_planner.models import TradeDirection, TradePlan
from trade_planner.types import StockData


class Setup(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def direction(self) -> TradeDirection: ...

    def required_warmup_bars(self) -> int: ...

    def is_valid(self, stock_data: StockData) -> bool: ...

    def generate_entry(self, stock_data: StockData) -> float | None: ...

    def generate_stop(self, stock_data: StockData, entry_price: float) -> float | None: ...

    def generate_target(
        self,
        stock_data: StockData,
        entry_price: float,
        stop_price: float,
    ) -> float | None: ...

    def confidence_score(self, stock_data: StockData) -> float: ...

    def build_plan(self, stock_data: StockData) -> TradePlan | None: ...
