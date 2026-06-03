"""Product-facing rankings API (precomputed snapshots only)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.product.models import RankingsTopResponseV1
from app.services.product_api_service import get_rankings_top_v1

router = APIRouter(prefix="/rankings", tags=["Product — Rankings"])


@router.get("/top", response_model=RankingsTopResponseV1)
async def rankings_top(
    limit: int = Query(20, ge=1, le=100),
    run_id: str | None = Query(None, description="Optional historical run id"),
) -> RankingsTopResponseV1:
    """
    Top ranked US equities for 5-session excess return vs SPY.

    Serves the latest successful batch run when ``run_id`` is omitted (failure fallback).
    """
    try:
        return get_rankings_top_v1(limit=limit, run_id=run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
