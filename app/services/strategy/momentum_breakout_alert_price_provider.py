"""Latest price lookup for Momentum Breakout alert monitoring."""

from __future__ import annotations

import logging
from typing import Protocol

from app.builders.finnhub_builder import FinnhubBuilder

logger = logging.getLogger(__name__)


class MomentumBreakoutPriceProvider(Protocol):
    def get_latest_price(self, symbol: str) -> float | None: ...


class FinnhubMomentumBreakoutPriceProvider:
    def __init__(self, finnhub_builder: FinnhubBuilder) -> None:
        self._builder = finnhub_builder

    def get_latest_price(self, symbol: str) -> float | None:
        quote = self._builder.get_quote(symbol.strip().upper())
        if quote is None:
            return None
        price = quote.c
        if price is None or price <= 0:
            return None
        return float(price)
