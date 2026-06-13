from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DayTradeBacktestOutcome = Literal[
    "win",
    "loss",
    "breakeven",
    "invalidated",
    "no_trade",
]

DayTradeSetupDirection = Literal["long", "short", "none"]

_MODEL_CONFIG = ConfigDict(populate_by_name=True)


class DayTradeBacktestRow(BaseModel):
    model_config = _MODEL_CONFIG

    date: date
    symbol: str
    setup_direction: DayTradeSetupDirection = Field(
        serialization_alias="setup_direction"
    )
    opening_range_high: float | None = Field(
        default=None,
        serialization_alias="opening_range_high",
    )
    opening_range_low: float | None = Field(
        default=None,
        serialization_alias="opening_range_low",
    )
    long_trigger: float | None = Field(default=None, serialization_alias="long_trigger")
    short_trigger: float | None = Field(
        default=None,
        serialization_alias="short_trigger",
    )
    vwap_at_entry: float | None = Field(
        default=None,
        serialization_alias="vwap_at_entry",
    )
    entry_time: datetime | None = Field(default=None, serialization_alias="entry_time")
    entry_price: float | None = Field(
        default=None,
        serialization_alias="entry_price",
    )
    stop_price: float | None = Field(default=None, serialization_alias="stop_price")
    target_1: float | None = Field(default=None, serialization_alias="target_1")
    target_2: float | None = Field(default=None, serialization_alias="target_2")
    exit_time: datetime | None = Field(default=None, serialization_alias="exit_time")
    exit_price: float | None = Field(default=None, serialization_alias="exit_price")
    outcome: DayTradeBacktestOutcome
    r_multiple: float = Field(serialization_alias="r_multiple")
    dollar_pl: float = Field(serialization_alias="dollar_pl")
    max_favorable_excursion: float = Field(
        serialization_alias="max_favorable_excursion"
    )
    max_adverse_excursion: float = Field(serialization_alias="max_adverse_excursion")


class DayTradeBacktestSummary(BaseModel):
    model_config = _MODEL_CONFIG

    total_trading_days_tested: int = Field(
        serialization_alias="total_trading_days_tested"
    )
    total_trades: int = Field(serialization_alias="total_trades")
    win_rate: float = Field(serialization_alias="win_rate")
    average_r: float = Field(serialization_alias="average_r")
    total_r: float = Field(serialization_alias="total_r")
    profit_factor: float | None = Field(serialization_alias="profit_factor")
    max_drawdown: float = Field(serialization_alias="max_drawdown")
    average_win: float = Field(serialization_alias="average_win")
    average_loss: float = Field(serialization_alias="average_loss")
    best_day: float = Field(serialization_alias="best_day")
    worst_day: float = Field(serialization_alias="worst_day")


class DayTradeBacktestResponse(BaseModel):
    model_config = _MODEL_CONFIG

    symbol: str
    start: date
    end: date
    risk_per_trade: float = Field(serialization_alias="risk_per_trade")
    rows: list[DayTradeBacktestRow]
    summary: DayTradeBacktestSummary
