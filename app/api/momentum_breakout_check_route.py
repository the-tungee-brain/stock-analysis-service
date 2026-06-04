from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.api.momentum_breakout_feature_guards import require_mb_alerts_enabled
from app.dependencies.service_dependencies import get_momentum_breakout_check_service
from app.models.momentum_breakout_check_models import MomentumBreakoutCheckResponse
from app.services.strategy.momentum_breakout_check_service import (
    MomentumBreakoutCheckService,
)

router = APIRouter(dependencies=[Depends(require_mb_alerts_enabled)])


@router.get(
    "/strategy/momentum-breakout/check/{symbol}",
    response_model=MomentumBreakoutCheckResponse,
    response_model_by_alias=True,
)
async def check_momentum_breakout_symbol(
    symbol: str,
    service: MomentumBreakoutCheckService = Depends(get_momentum_breakout_check_service),
) -> MomentumBreakoutCheckResponse:
    try:
        return await asyncio.to_thread(service.check, symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
