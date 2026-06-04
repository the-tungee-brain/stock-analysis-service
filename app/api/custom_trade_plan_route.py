from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.api.momentum_breakout_feature_guards import require_mb_alerts_enabled
from app.dependencies.service_dependencies import get_custom_trade_plan_service
from app.models.custom_trade_plan_models import (
    CustomTradePlanRequest,
    CustomTradePlanResponse,
)
from app.services.strategy.custom_trade_plan_service import CustomTradePlanService

router = APIRouter(dependencies=[Depends(require_mb_alerts_enabled)])


@router.post(
    "/strategy/custom-trade-plan",
    response_model=CustomTradePlanResponse,
    response_model_by_alias=True,
)
async def post_custom_trade_plan(
    body: CustomTradePlanRequest,
    service: CustomTradePlanService = Depends(get_custom_trade_plan_service),
) -> CustomTradePlanResponse:
    try:
        return await asyncio.to_thread(service.generate, body)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"OHLCV data not available for {body.symbol.strip().upper()}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
