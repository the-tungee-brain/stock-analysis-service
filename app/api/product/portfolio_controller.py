"""Product-facing portfolio API (precomputed snapshots only)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.product.models import PortfolioLatestResponseV1
from app.services.product_api_service import get_portfolio_latest_v1

router = APIRouter(prefix="/portfolio", tags=["Product — Portfolio"])


@router.get("/latest", response_model=PortfolioLatestResponseV1)
async def portfolio_latest() -> PortfolioLatestResponseV1:
    """
    Latest portfolio weights, metrics, and optional risk layer.

    No real-time optimization; returns last successful construction run.
    """
    try:
        return get_portfolio_latest_v1()
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
