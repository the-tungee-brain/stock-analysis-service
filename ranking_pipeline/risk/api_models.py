"""API extensions for portfolio risk layer (additive fields)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ranking_pipeline.portfolio.api_models import (
    LatestPortfolioResponse,
    PortfolioContributor,
    PortfolioHolding,
    PortfolioRiskSummary,
)


class PortfolioRiskLayerSummary(BaseModel):
    portfolio_beta: float | None = None
    portfolio_volatility: float | None = None
    target_volatility: float | None = None
    correlation_risk_score: float | None = None
    sector_breakdown: dict[str, float] = Field(default_factory=dict)
    vol_scale_factor: float | None = None


class LatestPortfolioEnrichedResponse(LatestPortfolioResponse):
    """Backward-compatible extension of latest portfolio response."""

    risk_layer: PortfolioRiskLayerSummary | None = None
