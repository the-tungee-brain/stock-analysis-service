"""Pattern prediction routes on the main Tomcrest API."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.auth.dependencies import get_current_user_id
from app.core.plan_features import PRO_FEATURE_PATTERN_TREND, require_paid_feature
from models.prediction_service import (
    LoadedModel,
    health_payload,
    load_deployed_model,
    predict_for_symbol,
)

router = APIRouter(prefix="/pattern", tags=["Pattern Prediction"])


def _get_loaded_model(app) -> LoadedModel | None:  # noqa: ANN001
    return getattr(app.state, "pattern_loaded_model", None)


@router.get("/health")
async def pattern_health(
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    require_paid_feature(user_id, PRO_FEATURE_PATTERN_TREND)
    loaded = _get_loaded_model(request.app)
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="Pattern model is not deployed. Run models.train_and_save first.",
        )
    return health_payload(loaded)


@router.get("/predict")
async def pattern_predict(
    request: Request,
    symbol: str = Query(..., min_length=1),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    require_paid_feature(user_id, PRO_FEATURE_PATTERN_TREND)
    loaded = _get_loaded_model(request.app)
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="Pattern model is not deployed. Run models.train_and_save first.",
        )

    try:
        return await asyncio.to_thread(predict_for_symbol, symbol, loaded)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
