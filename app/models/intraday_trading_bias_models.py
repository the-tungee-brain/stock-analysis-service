from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

IntradayBiasLabel = Literal["Bullish", "Neutral", "Bearish"]
IntradayConfidenceLabel = Literal["High", "Medium", "Low"]
IntradaySetupType = Literal[
    "GapAndGo",
    "OpeningRangeBreakout",
    "VWAPReclaim",
    "GapFade",
    "TrendDay",
    "RangeDay",
    "FailedBreakout",
    "None",
]
IntradayAction = Literal[
    "Watch",
    "Avoid",
    "ConfirmBreakout",
    "WaitForPullback",
    "RiskOff",
]
IntradayAlignmentState = Literal["aligned", "mixed", "against"]
IntradayVwapState = Literal["above", "below", "reclaiming", "rejecting"]
IntradayVolumeState = Literal["confirmed", "neutral", "weak"]
IntradayCatalystState = Literal["positive", "neutral", "negative", "none"]

_MODEL_CONFIG = ConfigDict(populate_by_name=True)


class IntradayTradingBiasLevels(BaseModel):
    model_config = _MODEL_CONFIG

    premarket_high: float | None = Field(
        default=None,
        serialization_alias="premarketHigh",
    )
    premarket_low: float | None = Field(
        default=None,
        serialization_alias="premarketLow",
    )
    open_range_high: float | None = Field(
        default=None,
        serialization_alias="openRangeHigh",
    )
    open_range_low: float | None = Field(
        default=None,
        serialization_alias="openRangeLow",
    )
    vwap: float | None = None
    support: float | None = None
    resistance: float | None = None
    invalidation: float | None = None


class IntradayTradingBiasAlignment(BaseModel):
    market: IntradayAlignmentState
    intraday_trend: IntradayAlignmentState = Field(
        serialization_alias="intradayTrend",
    )
    vwap: IntradayVwapState
    volume: IntradayVolumeState
    catalyst: IntradayCatalystState


class IntradayTradingBiasResponse(BaseModel):
    """Delayed intraday trading bias built from 5-minute yfinance bars."""

    model_config = _MODEL_CONFIG

    bias: IntradayBiasLabel
    confidence: IntradayConfidenceLabel
    horizon: Literal["Intraday"] = "Intraday"
    setup_type: IntradaySetupType = Field(serialization_alias="setupType")
    action: IntradayAction
    levels: IntradayTradingBiasLevels
    alignment: IntradayTradingBiasAlignment
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list, serialization_alias="dataGaps")
    last_updated: datetime | None = Field(
        default=None,
        serialization_alias="lastUpdated",
    )
    staleness_seconds: int | None = Field(
        default=None,
        serialization_alias="stalenessSeconds",
    )
    provider: Literal["yfinance"] = "yfinance"
    is_realtime: Literal[False] = Field(default=False, serialization_alias="isRealtime")
