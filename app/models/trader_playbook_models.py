from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PlaybookDirection = Literal["Bullish", "Neutral", "Bearish"]
PlaybookConfidence = Literal["High", "Medium", "Low"]
PlaybookBestSetup = Literal[
    "BreakoutContinuation",
    "PullbackToSupport",
    "FailedBreakout",
    "RangeDay",
    "TrendContinuation",
    "None",
]
PlaybookStatus = Literal["Valid", "Waiting", "Invalid", "NoSetup"]
RiskRewardLabel = Literal["favorable", "mixed", "poor", "unavailable"]
AlignmentState = Literal["aligned", "mixed", "against"]
AlignmentWithUnavailable = Literal["aligned", "mixed", "against", "unavailable"]
ExecutionReadinessAlignment = Literal["ready", "watch", "avoid"]
CatalystAlignment = Literal["positive", "neutral", "negative", "none"]

_MODEL_CONFIG = ConfigDict(populate_by_name=True)


class TraderPlaybookConditions(BaseModel):
    model_config = _MODEL_CONFIG

    valid_if: list[str] = Field(default_factory=list, serialization_alias="validIf")
    invalid_if: list[str] = Field(default_factory=list, serialization_alias="invalidIf")


class TraderPlaybookLevels(BaseModel):
    model_config = _MODEL_CONFIG

    entry: float | None = None
    stop: float | None = None
    target1: float | None = None
    target2: float | None = None
    support: float | None = None
    resistance: float | None = None
    breakout_level: float | None = Field(
        default=None,
        serialization_alias="breakoutLevel",
    )


class TraderPlaybookRisk(BaseModel):
    model_config = _MODEL_CONFIG

    risk_per_share: float | None = Field(
        default=None,
        serialization_alias="riskPerShare",
    )
    reward_to_target1: float | None = Field(
        default=None,
        serialization_alias="rewardToTarget1",
    )
    reward_to_target2: float | None = Field(
        default=None,
        serialization_alias="rewardToTarget2",
    )
    r_multiple_target1: float | None = Field(
        default=None,
        serialization_alias="rMultipleTarget1",
    )
    r_multiple_target2: float | None = Field(
        default=None,
        serialization_alias="rMultipleTarget2",
    )
    risk_reward_label: RiskRewardLabel = Field(
        default="unavailable",
        serialization_alias="riskRewardLabel",
    )


class TraderPlaybookAlignment(BaseModel):
    model_config = _MODEL_CONFIG

    daily_bias: AlignmentState = Field(serialization_alias="dailyBias")
    execution_readiness: ExecutionReadinessAlignment = Field(
        serialization_alias="executionReadiness",
    )
    market_regime: AlignmentWithUnavailable = Field(
        serialization_alias="marketRegime",
    )
    relative_strength: AlignmentWithUnavailable = Field(
        serialization_alias="relativeStrength",
    )
    price_structure: AlignmentWithUnavailable = Field(
        serialization_alias="priceStructure",
    )
    catalyst: CatalystAlignment


class TraderPlaybookResponse(BaseModel):
    """Daily condition-based trader playbook. Educational, not intraday."""

    model_config = _MODEL_CONFIG

    direction: PlaybookDirection
    confidence: PlaybookConfidence
    horizon: Literal["1-5 sessions"] = "1-5 sessions"
    data_mode: Literal["daily"] = Field(default="daily", serialization_alias="dataMode")
    best_setup: PlaybookBestSetup = Field(serialization_alias="bestSetup")
    status: PlaybookStatus
    conditions: TraderPlaybookConditions
    levels: TraderPlaybookLevels
    risk: TraderPlaybookRisk
    alignment: TraderPlaybookAlignment
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list, serialization_alias="dataGaps")
