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
DayTradeExitReason = Literal[
    "stop_hit",
    "target_1_hit",
    "target_2_hit",
    "invalidated",
    "close_exit",
    "no_trade",
]

DayTradeBacktestComparisonScenario = Literal[
    "baseline",
    "close-confirmed breakout",
    "VWAP-aligned",
    "OR-width-filtered",
    "all filters combined",
]

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
    or_width: float | None = Field(default=None, serialization_alias="or_width")
    stop_distance: float | None = Field(
        default=None,
        serialization_alias="stop_distance",
    )
    target_distance: float | None = Field(
        default=None,
        serialization_alias="target_distance",
    )
    exit_time: datetime | None = Field(default=None, serialization_alias="exit_time")
    exit_price: float | None = Field(default=None, serialization_alias="exit_price")
    exit_reason: DayTradeExitReason = Field(serialization_alias="exit_reason")
    stop_reason: str | None = Field(default=None, serialization_alias="stop_reason")
    target_reason: str | None = Field(default=None, serialization_alias="target_reason")
    hold_minutes: float | None = Field(default=None, serialization_alias="hold_minutes")
    entry_candle_closed_inside_or: bool = Field(
        default=False,
        serialization_alias="entry_candle_closed_inside_or",
    )
    outcome: DayTradeBacktestOutcome
    r_achieved: float = Field(serialization_alias="r_achieved")
    r_multiple: float = Field(serialization_alias="r_multiple")
    dollar_pl: float = Field(serialization_alias="dollar_pl")
    mfe: float = Field(serialization_alias="mfe")
    mae: float = Field(serialization_alias="mae")
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
    stop_hit_pct: float = Field(serialization_alias="stop_hit_pct")
    target_1_hit_pct: float = Field(serialization_alias="target_1_hit_pct")
    target_2_hit_pct: float = Field(serialization_alias="target_2_hit_pct")
    close_exit_pct: float = Field(serialization_alias="close_exit_pct")
    invalidation_pct: float = Field(serialization_alias="invalidation_pct")
    same_candle_invalidation_count: int = Field(
        serialization_alias="same_candle_invalidation_count"
    )
    average_stop_distance: float = Field(serialization_alias="average_stop_distance")
    average_or_width: float = Field(serialization_alias="average_or_width")
    average_hold_minutes: float = Field(serialization_alias="average_hold_minutes")


class DayTradeBacktestEntryFilters(BaseModel):
    model_config = _MODEL_CONFIG

    require_close_confirmed_breakout: bool = Field(
        default=False,
        serialization_alias="require_close_confirmed_breakout",
    )
    require_vwap_alignment: bool = Field(
        default=False,
        serialization_alias="require_vwap_alignment",
    )
    min_or_width_pct: float | None = Field(
        default=None,
        serialization_alias="min_or_width_pct",
    )
    max_or_width_pct: float | None = Field(
        default=None,
        serialization_alias="max_or_width_pct",
    )
    no_trade_after_noon: bool = Field(
        default=False,
        serialization_alias="no_trade_after_noon",
    )


class DayTradeBacktestComparisonRow(BaseModel):
    model_config = _MODEL_CONFIG

    scenario: DayTradeBacktestComparisonScenario
    total_trades: int = Field(serialization_alias="total_trades")
    win_rate: float = Field(serialization_alias="win_rate")
    average_r: float = Field(serialization_alias="average_r")
    total_r: float = Field(serialization_alias="total_r")
    profit_factor: float | None = Field(serialization_alias="profit_factor")
    max_drawdown: float = Field(serialization_alias="max_drawdown")
    target_1_hit_pct: float = Field(serialization_alias="target_1_hit_pct")
    invalidation_pct: float = Field(serialization_alias="invalidation_pct")


class DayTradeBacktestSymbolAggregateRow(BaseModel):
    model_config = _MODEL_CONFIG

    symbol: str
    total_trades: int = Field(serialization_alias="total_trades")
    win_rate: float = Field(serialization_alias="win_rate")
    average_r: float = Field(serialization_alias="average_r")
    total_r: float = Field(serialization_alias="total_r")
    profit_factor: float | None = Field(serialization_alias="profit_factor")
    max_drawdown: float = Field(serialization_alias="max_drawdown")
    target_1_hit_pct: float = Field(serialization_alias="target_1_hit_pct")
    invalidation_pct: float = Field(serialization_alias="invalidation_pct")
    stop_hit_pct: float = Field(serialization_alias="stop_hit_pct")


class DayTradeBacktestPortfolioSummary(BaseModel):
    model_config = _MODEL_CONFIG

    total_trades: int = Field(serialization_alias="total_trades")
    total_r: float = Field(serialization_alias="total_r")
    average_r: float = Field(serialization_alias="average_r")
    profit_factor: float | None = Field(serialization_alias="profit_factor")
    max_drawdown: float = Field(serialization_alias="max_drawdown")
    best_symbol: str | None = Field(serialization_alias="best_symbol")
    worst_symbol: str | None = Field(serialization_alias="worst_symbol")


class DayTradeBacktestMultiSymbolReport(BaseModel):
    model_config = _MODEL_CONFIG

    candidate_scenario: DayTradeBacktestComparisonScenario = Field(
        serialization_alias="candidate_scenario"
    )
    entry_filters: DayTradeBacktestEntryFilters = Field(
        serialization_alias="entry_filters"
    )
    symbols: list[str]
    aggregate_comparison: list[DayTradeBacktestSymbolAggregateRow] = Field(
        serialization_alias="aggregate_comparison"
    )
    portfolio_summary: DayTradeBacktestPortfolioSummary = Field(
        serialization_alias="portfolio_summary"
    )
    baseline_comparison: list[DayTradeBacktestSymbolAggregateRow] = Field(
        serialization_alias="baseline_comparison"
    )


class DayTradeBacktestResponse(BaseModel):
    model_config = _MODEL_CONFIG

    symbol: str
    start: date
    end: date
    available_start_date: date = Field(serialization_alias="available_start_date")
    available_end_date: date = Field(serialization_alias="available_end_date")
    provider_limit_reason: str = Field(serialization_alias="provider_limit_reason")
    risk_per_trade: float = Field(serialization_alias="risk_per_trade")
    invalidation_confirmation_closes: int = Field(
        serialization_alias="invalidation_confirmation_closes"
    )
    entry_filters: DayTradeBacktestEntryFilters = Field(
        serialization_alias="entry_filters"
    )
    rows: list[DayTradeBacktestRow]
    summary: DayTradeBacktestSummary
    top_winners: list[DayTradeBacktestRow] = Field(serialization_alias="top_winners")
    top_losers: list[DayTradeBacktestRow] = Field(serialization_alias="top_losers")
    comparison_table: list[DayTradeBacktestComparisonRow] = Field(
        serialization_alias="comparison_table"
    )
