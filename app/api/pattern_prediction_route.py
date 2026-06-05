"""Pattern prediction routes on the main Tomcrest API."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.auth.dependencies import get_current_user_id
from app.core.plan_features import PRO_FEATURE_PATTERN_TREND, require_paid_feature
from app.services.pattern_forecast_service import pattern_forecast_to_api_dict
from app.services.pattern_analysis_service import PatternAnalysisService
from models.prediction_service import (
    LoadedModel,
    health_payload,
)

router = APIRouter(prefix="/pattern", tags=["Pattern Prediction"])


def _get_loaded_model(app) -> LoadedModel | None:  # noqa: ANN001
    return getattr(app.state, "pattern_loaded_model", None)


def _get_pattern_analysis_service(app) -> PatternAnalysisService:  # noqa: ANN001
    service = getattr(app.state, "pattern_analysis_service", None)
    if service is not None:
        return service

    cache = None
    redis_client = getattr(app.state, "redis_client", None)
    if redis_client is not None:
        from app.adapters.cache.pattern_analysis_cache import PatternAnalysisCache

        cache = PatternAnalysisCache(redis_client=redis_client)
    service = PatternAnalysisService(cache=cache)
    app.state.pattern_analysis_service = service
    return service


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
        service = _get_pattern_analysis_service(request.app)
        payload = await asyncio.to_thread(
            service.get_or_build_prediction_payload,
            symbol,
            loaded,
        )
        return {"symbol": payload["symbol"], **pattern_forecast_to_api_dict(payload)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/intelligence")
async def pattern_intelligence(
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

    from app.services.pattern_intelligence_service import (
        pattern_intelligence_from_dict,
        pattern_intelligence_to_api_dict,
    )

    try:
        service = _get_pattern_analysis_service(request.app)
        snapshot = await asyncio.to_thread(service.get_or_build, symbol, loaded)
        payload = pattern_intelligence_from_dict(snapshot.pattern_intelligence)
    except (FileNotFoundError, ValueError, OSError):
        payload = None

    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pattern intelligence unavailable for {symbol.strip().upper()}",
        )
    return pattern_intelligence_to_api_dict(payload)
