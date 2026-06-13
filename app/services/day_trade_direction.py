from __future__ import annotations

from app.models.day_trade_backtest_models import DayTradeDirectionMode

DAY_TRADE_DIRECTION_MODES: tuple[DayTradeDirectionMode, ...] = (
    "long_only",
    "short_only",
    "long_and_short",
)


def day_trade_direction_allowed(
    direction: str,
    direction_mode: DayTradeDirectionMode,
) -> bool:
    if direction_mode == "long_only":
        return direction == "long"
    if direction_mode == "short_only":
        return direction == "short"
    return True


def parse_day_trade_direction_mode(value: str) -> DayTradeDirectionMode | None:
    normalized = value.strip().lower()
    if normalized in DAY_TRADE_DIRECTION_MODES:
        return normalized  # type: ignore[return-value]
    return None
