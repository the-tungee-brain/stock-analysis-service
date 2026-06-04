from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ExitVerdict = Literal["HOLD", "TRIM", "REVIEW_SELL", "EXIT"]
ExitConfidence = Literal["low", "medium", "high"]

_DISCLAIMER = (
    "Decision support only — not investment advice or a trade recommendation."
)


class EquityExitGuidanceContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    regime_id: str | None = Field(default=None, alias="regimeId")
    trade_quality_score: int | None = Field(default=None, alias="tradeQualityScore")
    position_weight_pct: float | None = Field(default=None, alias="positionWeightPct")
    open_profit_loss_pct: float | None = Field(default=None, alias="openProfitLossPct")
    ranking_rank: int | None = Field(default=None, alias="rankingRank")


class EquityExitGuidance(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    as_of_date: str | None = Field(default=None, alias="asOfDate")
    eligible: bool = True
    verdict: ExitVerdict | None = None
    confidence: ExitConfidence | None = None
    exit_urgency: int | None = Field(default=None, alias="exitUrgency", ge=0, le=100)
    primary_reason: str | None = Field(default=None, alias="primaryReason")
    supporting_factors: list[str] = Field(default_factory=list, alias="supportingFactors")
    risk_factors: list[str] = Field(default_factory=list, alias="riskFactors")
    would_improve: list[str] = Field(default_factory=list, alias="wouldImprove")
    would_worsen: list[str] = Field(default_factory=list, alias="wouldWorsen")
    disclaimer: str = Field(default=_DISCLAIMER)
    data_gaps: list[str] = Field(default_factory=list, alias="dataGaps")
    context: EquityExitGuidanceContext | None = None


class PortfolioExitAttentionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    verdict: ExitVerdict
    confidence: ExitConfidence
    exit_urgency: int = Field(alias="exitUrgency", ge=0, le=100)
    primary_reason: str = Field(alias="primaryReason")


class PortfolioExitAttentionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: list[PortfolioExitAttentionItem] = Field(default_factory=list)
