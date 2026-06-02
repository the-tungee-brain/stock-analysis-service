"""Productization API routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user_id
from app.core.plan_features import PRO_FEATURE_PATTERN_TREND, require_paid_feature
from app.models.productization_models import (
    EnhancedModelHealth,
    PortfolioCopilot,
    PredictionLedgerSummary,
)
from app.services.productization_service import (
    build_enhanced_model_health,
    build_portfolio_copilot_payload,
    build_prediction_ledger_payload,
)
from models.prediction_service import LoadedModel

router = APIRouter(prefix="/research", tags=["Productization"])


def _get_loaded_model(app) -> LoadedModel | None:  # noqa: ANN001
    return getattr(app.state, "pattern_loaded_model", None)


class PortfolioCopilotRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list, min_length=1, max_length=50)


@router.get(
    "/prediction-ledger",
    response_model=PredictionLedgerSummary,
    response_model_by_alias=True,
)
async def get_prediction_ledger(
    request: Request,
    symbol: str | None = Query(default=None, max_length=12),
    days: int = Query(default=30, ge=7, le=90),
    user_id: str = Depends(get_current_user_id),
) -> PredictionLedgerSummary:
    require_paid_feature(user_id, PRO_FEATURE_PATTERN_TREND)
    loaded = _get_loaded_model(request.app)
    if loaded is None:
        raise HTTPException(status_code=503, detail="Pattern model is not deployed.")

    return await asyncio.to_thread(
        build_prediction_ledger_payload,
        loaded,
        symbol=symbol.strip().upper() if symbol else None,
        days=days,
    )


@router.post(
    "/portfolio-copilot",
    response_model=PortfolioCopilot,
    response_model_by_alias=True,
)
async def post_portfolio_copilot(
    request: Request,
    body: PortfolioCopilotRequest,
    user_id: str = Depends(get_current_user_id),
) -> PortfolioCopilot:
    require_paid_feature(user_id, PRO_FEATURE_PATTERN_TREND)
    loaded = _get_loaded_model(request.app)
    if loaded is None:
        raise HTTPException(status_code=503, detail="Pattern model is not deployed.")

    return await asyncio.to_thread(
        build_portfolio_copilot_payload,
        body.symbols,
        loaded,
    )


@router.get(
    "/model-health",
    response_model=EnhancedModelHealth,
    response_model_by_alias=True,
)
async def get_model_health(
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> EnhancedModelHealth:
    require_paid_feature(user_id, PRO_FEATURE_PATTERN_TREND)
    loaded = _get_loaded_model(request.app)
    if loaded is None:
        raise HTTPException(status_code=503, detail="Pattern model is not deployed.")

    return await asyncio.to_thread(build_enhanced_model_health, loaded)
