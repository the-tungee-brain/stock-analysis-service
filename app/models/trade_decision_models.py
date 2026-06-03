from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TradeEnvironment = Literal["FAVORABLE", "NEUTRAL", "AVOID"]
OpportunityGrade = Literal["A", "B", "C", "D"]
SetupGrade = Literal["A", "B", "C", "D"]
BreakoutGrade = Literal["A", "B", "C", "D", "F"]
PatternReliability = Literal["low", "medium", "high"]
TradeVerdict = Literal[
    "HIGH_CONVICTION_TRADE",
    "MEDIUM_SETUP",
    "WATCHLIST",
    "NO_TRADE",
]
ActionHint = Literal["Buy", "Wait", "Avoid"]
TrendStage = Literal["early", "mid", "late", "unknown"]


class TradeDecisionStageRegime(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    regime_id: str | None = Field(default=None, alias="regimeId")
    trade_environment: TradeEnvironment = Field(alias="tradeEnvironment")


class TradeDecisionStageOpportunity(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    opportunity_grade: OpportunityGrade = Field(alias="opportunityGrade")
    rs_percentile: float | None = Field(default=None, alias="rsPercentile")
    trend_stage: TrendStage = Field(default="unknown", alias="trendStage")


class TradeDecisionStageSetup(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    setup_grade: SetupGrade = Field(alias="setupGrade")
    breakout_quality_score: int = Field(alias="breakoutQualityScore")
    breakout_grade: BreakoutGrade = Field(alias="breakoutGrade")
    support_resistance_confidence: int = Field(alias="supportResistanceConfidence")
    pattern_reliability: PatternReliability = Field(alias="patternReliability")


class TradeDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    as_of_date: str | None = Field(default=None, alias="asOfDate")
    regime: TradeDecisionStageRegime
    opportunity: TradeDecisionStageOpportunity
    setup: TradeDecisionStageSetup
    verdict: TradeVerdict
    action_hint: ActionHint = Field(alias="actionHint")
    explanation: list[str] = Field(default_factory=list, max_length=5)
