"""Latest portfolio API with optional risk layer fields."""

from __future__ import annotations

import json

from app.core.latency_observability import observe_dependency
from app.services.portfolio_construction_service import get_latest_portfolio
from ranking_pipeline.portfolio.api_models import LatestPortfolioResponse
from ranking_pipeline.risk.api_models import (
    LatestPortfolioEnrichedResponse,
    PortfolioRiskLayerSummary,
)
from ranking_pipeline.portfolio.persistence import open_portfolio_store


def get_latest_portfolio_enriched() -> LatestPortfolioEnrichedResponse:
    base: LatestPortfolioResponse = get_latest_portfolio()
    risk_layer = _load_risk_layer(base.portfolio_id)
    return LatestPortfolioEnrichedResponse(
        **base.model_dump(),
        risk_layer=risk_layer,
    )


def _load_risk_layer(portfolio_id: str) -> PortfolioRiskLayerSummary | None:
    store = open_portfolio_store()
    with observe_dependency("sqlite"):
        data = store.get_portfolio(portfolio_id)
    if not data:
        return None
    metrics_raw = data.get("metrics") or {}
    if not metrics_raw.get("metrics_json"):
        return None
    payload = json.loads(metrics_raw["metrics_json"])
    layer = payload.get("risk_layer")
    if not layer:
        return None
    return PortfolioRiskLayerSummary(
        portfolio_beta=layer.get("portfolio_beta"),
        portfolio_volatility=layer.get("portfolio_volatility"),
        target_volatility=layer.get("target_volatility"),
        correlation_risk_score=layer.get("correlation_risk_score"),
        sector_breakdown=layer.get("sector_breakdown") or {},
        vol_scale_factor=layer.get("vol_scale_factor"),
    )
