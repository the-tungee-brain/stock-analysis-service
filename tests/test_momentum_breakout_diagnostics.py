"""Momentum Breakout setup diagnostics."""

from __future__ import annotations

from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from trade_planner.setups.momentum_breakout_diagnostics import (
    RULE_PRICE_BELOW_SMA50,
    diagnose_momentum_breakout_setup,
)
from trade_planner.types import OHLCVBar, StockData
from tests.test_momentum_breakout_setup import _aligned_trend_series, _test_config


def test_diagnose_valid_setup_has_no_failures() -> None:
    stock, bench = _aligned_trend_series(120)
    setup = MomentumBreakoutSetup(_test_config())
    data = StockData.from_bars("NVDA", stock, benchmark_bars=bench)
    diag = diagnose_momentum_breakout_setup(data, setup)
    assert diag.setup_valid is True
    assert diag.failed_setup_rules == []


def test_diagnose_reports_price_below_sma50() -> None:
    stock, bench = _aligned_trend_series(120)
    # Crush last close below moving averages
    last_idx = len(stock) - 1
    stock = list(stock)
    stock[last_idx] = OHLCVBar(
        trading_date=stock[last_idx].trading_date,
        open=50.0,
        high=52.0,
        low=48.0,
        close=50.0,
        volume=stock[last_idx].volume,
    )
    setup = MomentumBreakoutSetup(_test_config())
    data = StockData.from_bars("WEAK", tuple(stock), benchmark_bars=bench)
    diag = diagnose_momentum_breakout_setup(data, setup)
    assert diag.setup_valid is False
    assert RULE_PRICE_BELOW_SMA50 in diag.failed_rule_ids
