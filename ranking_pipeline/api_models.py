"""API response models for stock rankings."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    group: str
    weighted_contribution: float


class RankedStock(BaseModel):
    symbol: str
    rank: int
    final_score: float
    composite_score: float | None = None
    probability_outperform_spy: float | None = Field(
        None,
        description="ML probability of outperforming SPY over 5 sessions",
    )
    expected_excess_return: float | None = None
    contributions: list[FeatureContribution] = Field(default_factory=list)


class RankingRunMeta(BaseModel):
    run_id: str
    as_of_date: str
    model_backend: str
    universe_snapshot_id: str | None = None
    symbol_count: int | None = None


class TopRankingsResponse(BaseModel):
    run: RankingRunMeta
    stocks: list[RankedStock]
