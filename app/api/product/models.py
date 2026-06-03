"""Stable versioned API contracts for Web/iOS clients (v1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

API_VERSION = "v1"


class ApiEnvelope(BaseModel):
    api_version: str = Field(default=API_VERSION, description="Contract version")


# --- Rankings ---


class RankingItemV1(BaseModel):
    symbol: str
    rank: int
    final_score: float
    ml_probability: float | None = None
    expected_excess_return: float | None = None


class RankingsTopResponseV1(ApiEnvelope):
    timestamp: str
    run_id: str
    as_of_date: str
    regime_id: str | None = None
    items: list[RankingItemV1]


# --- Portfolio ---


class HoldingScoreContributionV1(BaseModel):
    symbol: str
    weight: float
    score_contribution: float
    final_score: float | None = None
    expected_excess_return: float | None = None


class PortfolioMetricsV1(BaseModel):
    expected_return_5d: float
    expected_excess_5d: float
    volatility: float | None = None
    beta_vs_spy: float | None = None
    correlation_risk_score: float | None = None
    sector_breakdown: dict[str, float] = Field(default_factory=dict)
    turnover_estimate: float | None = None
    concentration_hhi: float | None = None


class RiskLayerV1(BaseModel):
    portfolio_beta: float | None = None
    portfolio_volatility: float | None = None
    target_volatility: float | None = None
    correlation_risk_score: float | None = None
    sector_breakdown: dict[str, float] = Field(default_factory=dict)
    vol_scale_factor: float | None = None


class PortfolioLatestResponseV1(ApiEnvelope):
    timestamp: str
    portfolio_id: str
    ranking_run_id: str
    as_of_date: str
    sizing_mode: str
    holdings: list[HoldingScoreContributionV1]
    metrics: PortfolioMetricsV1
    risk_layer: RiskLayerV1 | None = None
    top_contributors: list[dict[str, float]] = Field(default_factory=list)


# --- Health ---


class SystemHealthResponseV1(ApiEnvelope):
    last_pipeline_run_time: str | None = None
    universe_size: int | None = None
    last_successful_ranking_run: str | None = None
    last_successful_portfolio_run: str | None = None
    system_status: str = Field(description="ok | degraded | failing")
    last_ranking_run_at: str | None = None
    last_portfolio_run_at: str | None = None
    regime_id: str | None = None
