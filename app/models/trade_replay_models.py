from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.day_trade_backtest_models import DayTradeDirectionMode

TradeReplayWorkflow = Literal["day_trade", "swing_trade"]
TradeReplaySeverity = Literal["info", "important", "warning"]
TradeReplayActionability = Literal["active", "missed", "invalidated"]
TradeReplaySource = Literal["realtime", "delayed", "historical"]
MissedMovesRange = Literal["today", "last_5_trading_days"]
MissedMovesSort = Literal["most_recent", "biggest_move", "highest_setup_quality"]
MissedMoveOutcome = Literal["target_hit", "extended", "invalidated", "stopped"]

_MODEL_CONFIG = ConfigDict(populate_by_name=True)


class TradeReplayEvent(BaseModel):
    model_config = _MODEL_CONFIG

    id: str
    plan_id: str | None = Field(default=None, serialization_alias="plan_id")
    symbol: str
    event_date: date = Field(serialization_alias="event_date")
    workflow: TradeReplayWorkflow
    event_type: str = Field(serialization_alias="event_type")
    event_time: datetime = Field(serialization_alias="event_time")
    level_price: float | None = Field(default=None, serialization_alias="level_price")
    observed_price: float | None = Field(
        default=None,
        serialization_alias="observed_price",
    )
    message: str
    severity: TradeReplaySeverity
    actionability: TradeReplayActionability
    source: TradeReplaySource
    source_freshness_label: str | None = Field(
        default=None,
        serialization_alias="source_freshness_label",
    )
    dedupe_key: str
    created_at: datetime | None = Field(default=None, serialization_alias="created_at")


class TradeReplayResponse(BaseModel):
    model_config = _MODEL_CONFIG

    symbol: str
    date: date
    workflow: TradeReplayWorkflow
    direction_mode: DayTradeDirectionMode | None = Field(
        default=None,
        serialization_alias="direction_mode",
    )
    source: TradeReplaySource = "delayed"
    source_freshness_label: str | None = Field(
        default=None,
        serialization_alias="source_freshness_label",
    )
    events: list[TradeReplayEvent] = Field(default_factory=list)


class MissedMoveSummaryRow(BaseModel):
    model_config = _MODEL_CONFIG

    id: str
    date: date
    symbol: str
    workflow: TradeReplayWorkflow
    setup_type: str = Field(serialization_alias="setup_type")
    direction: str | None = None
    trigger_time: datetime | None = Field(
        default=None,
        serialization_alias="trigger_time",
    )
    trigger_price: float | None = Field(default=None, serialization_alias="trigger_price")
    outcome: MissedMoveOutcome
    max_move_after_trigger_pct: float | None = Field(
        default=None,
        serialization_alias="max_move_after_trigger_pct",
    )
    setup_quality_score: float | None = Field(
        default=None,
        serialization_alias="setup_quality_score",
    )
    entry: float | None = None
    stop: float | None = None
    target_1: float | None = Field(default=None, serialization_alias="target_1")
    target_2: float | None = Field(default=None, serialization_alias="target_2")
    open_range_high: float | None = Field(
        default=None,
        serialization_alias="open_range_high",
    )
    open_range_low: float | None = Field(
        default=None,
        serialization_alias="open_range_low",
    )
    vwap: float | None = None
    event_count: int | None = Field(default=None, serialization_alias="event_count")
    source: TradeReplaySource = "historical"
    source_freshness_label: str | None = Field(
        default=None,
        serialization_alias="source_freshness_label",
    )


class MissedMovesSummaryResponse(BaseModel):
    model_config = _MODEL_CONFIG

    range: MissedMovesRange
    sort: MissedMovesSort
    source: TradeReplaySource = "historical"
    source_freshness_label: str | None = Field(
        default=None,
        serialization_alias="source_freshness_label",
    )
    rows: list[MissedMoveSummaryRow] = Field(default_factory=list)


class TradeReplayRefreshRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12)
    workflow: TradeReplayWorkflow
    date: date
    direction_mode: DayTradeDirectionMode | None = Field(
        default=None,
        serialization_alias="direction_mode",
    )


class TradeReplayRefreshResponse(BaseModel):
    model_config = _MODEL_CONFIG

    success: bool
    symbol: str
    date: date
    workflow: TradeReplayWorkflow
    direction_mode: DayTradeDirectionMode | None = Field(
        default=None,
        serialization_alias="direction_mode",
    )
    plan_id: str | None = Field(default=None, serialization_alias="plan_id")
    plan_created: bool = Field(default=False, serialization_alias="plan_created")
    events_created: int = Field(default=0, serialization_alias="events_created")
    source: TradeReplaySource = "delayed"
    source_freshness_label: str | None = Field(
        default=None,
        serialization_alias="source_freshness_label",
    )
