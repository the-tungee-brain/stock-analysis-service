"""Latest constructed portfolio from ranking pipeline output."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.portfolio_enriched_service import get_latest_portfolio_enriched
from ranking_pipeline.risk.api_models import LatestPortfolioEnrichedResponse

router = APIRouter(prefix="/portfolio", tags=["Portfolio Construction"])


@router.get("/latest", response_model=LatestPortfolioEnrichedResponse)
async def get_latest_constructed_portfolio() -> LatestPortfolioEnrichedResponse:
    """
    Return weights and risk metrics from the latest portfolio snapshot.

    Downstream of nightly ranking; does not recompute features or scores.
    Includes optional ``risk_layer`` when built via portfolio+risk pipeline.
    """
    try:
        return get_latest_portfolio_enriched()
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
