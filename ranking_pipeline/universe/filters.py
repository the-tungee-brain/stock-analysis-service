"""Liquidity screening for universe construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ranking_pipeline.config import LiquidityFilters


@dataclass(frozen=True)
class UniverseMemberMetrics:
    symbol: str
    last_close: float
    market_cap: float | None
    avg_dollar_volume_20d: float
    passed: bool


def compute_adv_dollars(ohlcv: pd.DataFrame, lookback: int = 20) -> float:
    tail = ohlcv.tail(lookback)
    if tail.empty:
        return 0.0
    return float((tail["close"] * tail["volume"]).mean())


def screen_symbol_ohlcv(
    symbol: str,
    ohlcv: pd.DataFrame,
    *,
    market_cap: float | None,
    filters: LiquidityFilters,
) -> UniverseMemberMetrics:
    """Apply price, ADV, and market-cap rules to one symbol."""
    if ohlcv.empty:
        return UniverseMemberMetrics(symbol, 0.0, market_cap, 0.0, False)

    last_close = float(ohlcv["close"].iloc[-1])
    adv = compute_adv_dollars(ohlcv, lookback=20)
    passed = (
        last_close > filters.min_price
        and adv >= filters.min_avg_dollar_volume_20d
        and market_cap is not None
        and market_cap >= filters.min_market_cap
    )
    return UniverseMemberMetrics(
        symbol=symbol.strip().upper(),
        last_close=last_close,
        market_cap=market_cap,
        avg_dollar_volume_20d=adv,
        passed=passed,
    )
