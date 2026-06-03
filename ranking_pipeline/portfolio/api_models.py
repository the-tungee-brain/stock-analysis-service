"""API models for constructed portfolio (separate from ranking API)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PortfolioHolding(BaseModel):
    symbol: str
    weight: float
    final_score: float | None = None
    probability_outperform_spy: float | None = None
    expected_excess_return: float | None = None


class PortfolioContributor(BaseModel):
    symbol: str
    weight: float
    expected_excess_return: float
    contribution: float


class PortfolioRiskSummary(BaseModel):
    expected_return_5d: float
    expected_excess_5d: float
    portfolio_volatility: float | None = None
    turnover: float | None = None
    concentration_hhi: float | None = None


class LatestPortfolioResponse(BaseModel):
    portfolio_id: str
    ranking_run_id: str
    as_of_date: str
    sizing_mode: str
    holdings: list[PortfolioHolding]
    risk: PortfolioRiskSummary
    top_contributors: list[PortfolioContributor] = Field(default_factory=list)
