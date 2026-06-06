from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BiasLabel = Literal["Bullish", "Neutral", "Bearish"]
ConfidenceLabel = Literal["High", "Medium", "Low"]
TradingBiasAction = Literal[
    "Watch",
    "Avoid",
    "Confirm breakout",
    "Pullback setup",
    "Risk-off",
]
AlignmentState = Literal["aligned", "mixed", "against"]
VolumeAlignment = Literal["confirmed", "neutral", "warning"]
CatalystAlignment = Literal["positive", "neutral", "negative", "none"]

_TRADING_BIAS_MODEL_CONFIG = ConfigDict(populate_by_name=True)


class TradingBiasLevels(BaseModel):
    model_config = _TRADING_BIAS_MODEL_CONFIG

    support: float | None = None
    resistance: float | None = None
    breakout_level: float | None = Field(
        default=None,
        serialization_alias="breakoutLevel",
    )
    stop_invalid_level: float | None = Field(
        default=None,
        serialization_alias="stopInvalidLevel",
    )


class TradingBiasAlignment(BaseModel):
    model_config = _TRADING_BIAS_MODEL_CONFIG

    market_regime: AlignmentState = Field(serialization_alias="marketRegime")
    relative_strength: AlignmentState = Field(serialization_alias="relativeStrength")
    pattern_trend: AlignmentState = Field(serialization_alias="patternTrend")
    volume: VolumeAlignment
    catalyst: CatalystAlignment


class TradingBiasResponse(BaseModel):
    """Short-term daily trading bias synthesized from existing research signals."""

    model_config = _TRADING_BIAS_MODEL_CONFIG

    symbol: str
    bias: BiasLabel
    confidence: ConfidenceLabel
    horizon: Literal["1-5 sessions"] = "1-5 sessions"
    action: TradingBiasAction
    bullish_factors: list[str] = Field(
        default_factory=list,
        serialization_alias="bullishFactors",
    )
    bearish_factors: list[str] = Field(
        default_factory=list,
        serialization_alias="bearishFactors",
    )
    invalidation: str | None = None
    levels: TradingBiasLevels
    alignment: TradingBiasAlignment
    data_gaps: list[str] = Field(default_factory=list, serialization_alias="dataGaps")
