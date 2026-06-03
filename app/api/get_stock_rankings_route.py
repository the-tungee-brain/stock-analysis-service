"""Precomputed 5-day stock ranking API (backend batch output)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.ranking_service import get_top_rankings
from ranking_pipeline.api_models import TopRankingsResponse

router = APIRouter(prefix="/rankings", tags=["Stock Rankings"])


@router.get("/top", response_model=TopRankingsResponse)
async def get_top_ranked_stocks(
    limit: int = Query(20, ge=1, le=100),
    run_id: str | None = Query(None, description="Specific ranking run id"),
) -> TopRankingsResponse:
    """
    Return top ranked US equities for 5-session excess return vs SPY.

    Scores are precomputed nightly; this endpoint does not run feature engineering.
    """
    try:
        return get_top_rankings(limit=limit, run_id=run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
