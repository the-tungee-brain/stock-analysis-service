"""Research decision layer API routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.auth.dependencies import get_current_user_id
from app.core.plan_features import PRO_FEATURE_PATTERN_TREND, require_paid_feature
from app.models.research_decision_models import (
    ModelDiagnostics,
    PortfolioRankingDashboard,
    ResearchDecision,
)
from app.services.research_decision_service import (
    build_model_diagnostics_payload,
    build_portfolio_ranking_payload,
    build_research_decision_payload,
)
from models.prediction_service import LoadedModel

router = APIRouter(prefix="/research", tags=["Research Decision"])


def _get_loaded_model(app) -> LoadedModel | None:  # noqa: ANN001
    return getattr(app.state, "pattern_loaded_model", None)


@router.get(
    "/decision",
    response_model=ResearchDecision,
    response_model_by_alias=True,
)
async def get_research_decision(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=12),
    user_id: str = Depends(get_current_user_id),
) -> ResearchDecision:
    require_paid_feature(user_id, PRO_FEATURE_PATTERN_TREND)
    loaded = _get_loaded_model(request.app)
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="Pattern model is not deployed.",
        )

    payload = await asyncio.to_thread(
        build_research_decision_payload,
        symbol.strip().upper(),
        loaded,
    )
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Research decision unavailable for {symbol.strip().upper()}",
        )
    return payload


@router.get(
    "/portfolio-ranking",
    response_model=PortfolioRankingDashboard,
    response_model_by_alias=True,
)
async def get_portfolio_ranking(
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> PortfolioRankingDashboard:
    require_paid_feature(user_id, PRO_FEATURE_PATTERN_TREND)
    loaded = _get_loaded_model(request.app)
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="Pattern model is not deployed.",
        )

    payload = await asyncio.to_thread(build_portfolio_ranking_payload, loaded)
    if payload is None:
        raise HTTPException(status_code=503, detail="Portfolio ranking unavailable.")
    return payload


@router.get(
    "/model-diagnostics",
    response_model=ModelDiagnostics,
    response_model_by_alias=True,
)
async def get_model_diagnostics(
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> ModelDiagnostics:
    require_paid_feature(user_id, PRO_FEATURE_PATTERN_TREND)
    loaded = _get_loaded_model(request.app)
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="Pattern model is not deployed.",
        )

    payload = await asyncio.to_thread(build_model_diagnostics_payload, loaded)
    if payload is None:
        raise HTTPException(status_code=503, detail="Model diagnostics unavailable.")
    return payload
