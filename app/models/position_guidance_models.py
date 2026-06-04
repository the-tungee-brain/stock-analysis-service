from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SymbolThesis = Literal["BULLISH", "NEUTRAL", "BEARISH"]
PositionKind = Literal[
    "EQUITY_LONG",
    "LONG_CALL",
    "LONG_PUT",
    "SHORT_CALL",
    "SHORT_PUT",
]
EquityVerdict = Literal["HOLD", "TRIM", "REVIEW_SELL", "EXIT"]
LongOptionVerdict = Literal["HOLD", "REVIEW_CLOSE", "CLOSE"]
ShortOptionVerdict = Literal["HOLD", "ROLL", "CLOSE", "REVIEW_ASSIGNMENT_RISK"]
PositionVerdict = EquityVerdict | LongOptionVerdict | ShortOptionVerdict
GuidanceConfidence = Literal["low", "medium", "high"]

_DISCLAIMER = (
    "Decision support only — not investment advice or a trade recommendation."
)


class GuidanceDriverModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    code: str
    label: str
    points: float
    detail: str | None = None


class ScoringContributorModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bucket: str
    points: float
    label: str
    driver_code: str | None = Field(default=None, alias="driverCode")


class SymbolThesisBlock(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    thesis: SymbolThesis
    summary: str
    trade_quality_score: int | None = Field(default=None, alias="tradeQualityScore")
    regime_id: str | None = Field(default=None, alias="regimeId")


class PositionGuidanceItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    position_key: str = Field(alias="positionKey")
    position_kind: PositionKind = Field(alias="positionKind")
    display_label: str = Field(alias="displayLabel")
    instrument_symbol: str = Field(alias="instrumentSymbol")
    underlying_symbol: str = Field(alias="underlyingSymbol")
    put_call: str | None = Field(default=None, alias="putCall")
    strike: float | None = None
    expiration: str | None = None
    quantity: float
    market_value: float = Field(alias="marketValue")
    open_profit_loss_pct: float | None = Field(
        default=None, alias="openProfitLossPct"
    )
    verdict: PositionVerdict
    confidence: GuidanceConfidence
    urgency: int = Field(ge=0, le=100)
    relative_risk_rank: int = Field(default=0, ge=0, le=100, alias="relativeRiskRank")
    cross_leg_sanity: bool = Field(default=False, alias="crossLegSanity")
    justification: str
    primary_driver: GuidanceDriverModel = Field(alias="primaryDriver")
    secondary_driver: GuidanceDriverModel | None = Field(
        default=None, alias="secondaryDriver"
    )
    tertiary_driver: GuidanceDriverModel | None = Field(
        default=None, alias="tertiaryDriver"
    )
    primary_reason: str = Field(alias="primaryReason")
    supporting_factors: list[str] = Field(
        default_factory=list, alias="supportingFactors"
    )
    risk_factors: list[str] = Field(default_factory=list, alias="riskFactors")
    scoring_contributors: list[ScoringContributorModel] = Field(
        default_factory=list, alias="scoringContributors"
    )


class SymbolPositionGuidanceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    as_of_date: str | None = Field(default=None, alias="asOfDate")
    has_positions: bool = Field(default=False, alias="hasPositions")
    thesis: SymbolThesisBlock | None = None
    positions: list[PositionGuidanceItem] = Field(default_factory=list)
    synthesis_narrative: str = Field(default="", alias="synthesisNarrative")
    analysis_prompt: str = Field(default="", alias="analysisPrompt")
    disclaimer: str = Field(default=_DISCLAIMER)
    data_gaps: list[str] = Field(default_factory=list, alias="dataGaps")
    scoring_trace: str = Field(default="", alias="scoringTrace")


class PortfolioExitAttentionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    position_key: str = Field(alias="positionKey")
    symbol: str
    position_kind: PositionKind = Field(alias="positionKind")
    display_label: str = Field(alias="displayLabel")
    verdict: PositionVerdict
    confidence: GuidanceConfidence
    urgency: int = Field(ge=0, le=100)
    relative_risk_rank: int = Field(default=0, ge=0, le=100, alias="relativeRiskRank")
    primary_reason: str = Field(alias="primaryReason")


class PortfolioExitAttentionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: list[PortfolioExitAttentionItem] = Field(default_factory=list)
