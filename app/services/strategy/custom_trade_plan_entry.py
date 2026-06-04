"""Custom trade plan entry trigger derivation (educational, not live orders)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trade_planner.indicators import highest_high
from trade_planner.types import OHLCVBar

CustomEntryMethod = Literal[
    "CURRENT_CLOSE",
    "AT_20D_HIGH",
    "BREAKOUT_ABOVE_20D_HIGH",
]

ENTRY_BUFFER_USD = 0.01
HIGH_LOOKBACK_DAYS = 20
ENTRY_WARNING_DISTANCE_PCT = 5.0
ENTRY_INACTIVE_DISPLAY_PCT = 2.0


@dataclass(frozen=True, slots=True)
class CustomEntryDerivation:
    current_price: float
    period_high_20d: float
    entry_price: float
    entry_method: CustomEntryMethod
    distance_to_entry_pct: float
    entry_explanation: str
    latest_bar_date: str
    plan_active_at_current_price: bool

    @property
    def show_inactive_until_trigger(self) -> bool:
        return self.distance_to_entry_pct > ENTRY_INACTIVE_DISPLAY_PCT


def derive_custom_long_entry(
    stock_bars: tuple[OHLCVBar, ...] | list[OHLCVBar],
    *,
    symbol: str,
    high_lookback_days: int = HIGH_LOOKBACK_DAYS,
    entry_buffer: float = ENTRY_BUFFER_USD,
) -> CustomEntryDerivation:
    """
    Long entry trigger for custom educational plans.

    Uses the latest daily close as ``current_price`` and sets the trigger to
    ``max(close, 20-day high) + buffer``. When price is below the 20-day high,
    the plan is inactive until that breakout level is reached (buy-stop style).
    """
    if not stock_bars:
        raise ValueError("No price bars available")

    last = stock_bars[-1]
    current_price = round(last.close, 4)
    period_high = highest_high(stock_bars, high_lookback_days)
    if period_high is None or period_high <= 0:
        period_high = last.high
    period_high = round(period_high, 4)

    trigger_base = max(current_price, period_high)
    entry_price = round(trigger_base + entry_buffer, 4)

    if entry_price <= current_price * 1.0001:
        if current_price >= period_high * 0.999:
            entry_method: CustomEntryMethod = "AT_20D_HIGH"
            entry_explanation = (
                f"Price is at or near the recent {high_lookback_days}-day high. "
                f"The educational plan uses the latest close (${current_price:,.2f}) "
                f"as the reference entry level."
            )
        else:
            entry_method = "CURRENT_CLOSE"
            entry_explanation = (
                f"The educational plan uses the latest closing price "
                f"(${current_price:,.2f}) as the entry reference."
            )
    else:
        entry_method = "BREAKOUT_ABOVE_20D_HIGH"
        entry_explanation = (
            f"Plan activates only if price breaks above the recent "
            f"{high_lookback_days}-day high (${period_high:,.2f}), plus a "
            f"${entry_buffer:.2f} buffer."
        )

    distance_to_entry_pct = (
        round((entry_price - current_price) / current_price * 100.0, 2)
        if current_price > 0
        else 0.0
    )

    return CustomEntryDerivation(
        current_price=current_price,
        period_high_20d=period_high,
        entry_price=entry_price,
        entry_method=entry_method,
        distance_to_entry_pct=max(0.0, distance_to_entry_pct),
        entry_explanation=entry_explanation,
        latest_bar_date=last.trading_date.isoformat(),
        plan_active_at_current_price=distance_to_entry_pct
        <= ENTRY_INACTIVE_DISPLAY_PCT,
    )


def entry_distance_warning(distance_to_entry_pct: float) -> str | None:
    if distance_to_entry_pct > ENTRY_WARNING_DISTANCE_PCT:
        return (
            "Entry trigger is significantly above the current price and may "
            "take time to activate."
        )
    return None
