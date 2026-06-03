from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TradeEnvironment = Literal["FAVORABLE", "NEUTRAL", "AVOID"]
TradeVerdict = Literal["HIGH_CONVICTION_TRADE", "WATCHLIST", "NO_TRADE"]
TradeAction = Literal["ENTER", "WAIT_FOR_SETUP", "AVOID"]


class TradeDecisionRegime(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    regime_id: str | None = Field(default=None, alias="regimeId")
    trade_environment: TradeEnvironment = Field(alias="tradeEnvironment")


class TradeDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    as_of_date: str | None = Field(default=None, alias="asOfDate")
    regime: TradeDecisionRegime
    trade_quality_score: int = Field(alias="tradeQualityScore", ge=0, le=100)
    verdict: TradeVerdict
    action: TradeAction
    explanation: list[str] = Field(default_factory=list, max_length=5)
