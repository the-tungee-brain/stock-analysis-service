from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SetupStageId = Literal[
    "BASE_BUILDING",
    "TIGHTENING",
    "BREAKOUT_WATCH",
    "BREAKOUT_TRIGGERED",
    "EXTENDED",
]


class EmergingLeaderItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rank: int
    symbol: str
    setup_quality_score: int = Field(alias="setupQualityScore", ge=0, le=100)
    setup_stage: SetupStageId = Field(alias="setupStage")
    setup_stage_label: str = Field(alias="setupStageLabel")
    why_it_ranks: str = Field(alias="whyItRanks")
    positive_factors: list[str] = Field(alias="positiveFactors")
    missing_factors: list[str] = Field(alias="missingFactors")
    next_confirmation: list[str] = Field(alias="nextConfirmation")


class EmergingLeadersResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    as_of_date: str | None = Field(default=None, alias="asOfDate")
    timestamp: str
    universe_scanned: int = Field(alias="universeScanned")
    symbols_with_data: int = Field(alias="symbolsWithData")
    evaluations_computed: int = Field(alias="evaluationsComputed")
    excluded_top_movers: int = Field(alias="excludedTopMovers")
    items: list[EmergingLeaderItem]
