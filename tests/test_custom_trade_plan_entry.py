"""Custom trade plan entry trigger derivation."""

from __future__ import annotations

from datetime import date, timedelta

from trade_planner.types import OHLCVBar

from app.services.strategy.custom_trade_plan_entry import (
    derive_custom_long_entry,
    entry_distance_warning,
)


def _bar(d: date, close: float, high: float | None = None) -> OHLCVBar:
    h = high if high is not None else close + 1.0
    return OHLCVBar(
        trading_date=d,
        open=close,
        high=h,
        low=close - 1.0,
        close=close,
        volume=1_000_000.0,
    )


def test_breakout_above_20d_high_when_price_below_high() -> None:
    start = date(2026, 1, 2)
    bars = [_bar(start + timedelta(days=i), close=400.0, high=405.0) for i in range(19)]
    bars.append(_bar(start + timedelta(days=19), close=450.0, high=466.0))
    bars.append(_bar(start + timedelta(days=20), close=428.0, high=432.0))

    info = derive_custom_long_entry(bars, symbol="MSFT")
    assert info.entry_method == "BREAKOUT_ABOVE_20D_HIGH"
    assert info.current_price == 428.0
    assert info.period_high_20d == 466.0
    assert info.entry_price == 466.01
    assert info.distance_to_entry_pct > 8.0
    assert info.show_inactive_until_trigger is True
    assert info.plan_active_at_current_price is False
    assert "20-day high" in info.entry_explanation


def test_plan_active_when_close_near_high() -> None:
    start = date(2026, 3, 1)
    bars = [
        _bar(start + timedelta(days=i), close=100.0 + i * 0.5, high=101.0 + i * 0.5)
        for i in range(21)
    ]
    info = derive_custom_long_entry(bars, symbol="TEST")
    assert info.distance_to_entry_pct <= 2.0
    assert info.plan_active_at_current_price is True


def test_entry_distance_warning_over_five_pct() -> None:
    assert entry_distance_warning(8.96) is not None
    assert entry_distance_warning(3.0) is None
