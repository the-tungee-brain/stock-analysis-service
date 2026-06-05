from __future__ import annotations

import logging

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.core.latency_observability import record_dependency_latency
from data.loader import load_symbol

logger = logging.getLogger(__name__)


class ResearchPriceHistoryService:
    def __init__(self, *, yahoo_fallback: YFinanceAdapter):
        self.yahoo_fallback = yahoo_fallback

    def get_daily_closes_1y(self, symbol: str) -> pd.Series:
        symbol_upper = symbol.strip().upper()
        local = self._local_daily_closes_1y(symbol_upper)
        if local is not None:
            record_dependency_latency(
                "research_price_history",
                0.0,
                cache_status="local_ohlcv",
            )
            logger.info(
                "Research price history source=local_ohlcv symbol=%s",
                symbol_upper,
            )
            return local

        record_dependency_latency(
            "research_price_history",
            0.0,
            cache_status="yahoo_fallback",
        )
        logger.info(
            "Research price history source=yahoo_fallback symbol=%s",
            symbol_upper,
        )
        fallback = self.yahoo_fallback.get_daily_closes_1y(symbol=symbol_upper)
        if fallback.empty:
            record_dependency_latency(
                "research_price_history",
                0.0,
                cache_status="unavailable",
            )
            logger.info(
                "Research price history source=unavailable symbol=%s",
                symbol_upper,
            )
        return fallback

    def _local_daily_closes_1y(self, symbol_upper: str) -> pd.Series | None:
        try:
            frame = load_symbol(symbol_upper)
        except FileNotFoundError:
            return None
        except Exception:
            logger.warning(
                "Local OHLCV unavailable for research performance: %s",
                symbol_upper,
                exc_info=True,
            )
            return None

        if frame.empty or "close" not in frame.columns:
            return None

        closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
        closes = closes[closes > 0]
        if len(closes) < 2:
            return None

        closes.index = pd.to_datetime(closes.index)
        closes = closes.sort_index()
        cutoff = closes.index[-1] - pd.Timedelta(days=365)
        recent = closes[closes.index >= cutoff]
        if len(recent) < 2:
            return None
        return recent
