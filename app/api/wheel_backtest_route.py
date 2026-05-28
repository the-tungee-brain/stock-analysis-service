from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies.service_dependencies import get_wheel_backtest_service
from app.models.wheel_backtest_models import WheelBacktestResponse
from app.services.strategy.wheel_backtest_service import WheelBacktestService

router = APIRouter()


@router.get(
    "/strategy/wheel-backtest",
    response_model=WheelBacktestResponse,
    response_model_by_alias=True,
)
async def get_wheel_backtest(
    symbol: str = Query(..., min_length=1, max_length=12),
    years: int = Query(5, ge=5, le=15, description="5, 10, or 15"),
    target_delta_min: float = Query(0.20, ge=0.05, le=0.50, alias="targetDeltaMin"),
    target_delta_max: float = Query(0.30, ge=0.05, le=0.50, alias="targetDeltaMax"),
    dte_days: int = Query(30, ge=5, le=60, alias="dteDays"),
    contracts: int = Query(1, ge=1, le=20),
    maintain_one_lot: bool = Query(
        False,
        alias="maintainOneLot",
        description="Inject cash to keep selling one CSP when collateral rises",
    ),
    service: WheelBacktestService = Depends(get_wheel_backtest_service),
) -> WheelBacktestResponse:
    if years not in (5, 10, 15):
        raise HTTPException(status_code=400, detail="years must be 5, 10, or 15")

    try:
        return await asyncio.to_thread(
            service.run_backtest,
            symbol,
            lookback_years=years,
            target_delta_min=target_delta_min,
            target_delta_max=target_delta_max,
            dte_days=dte_days,
            contracts=contracts,
            maintain_one_lot=maintain_one_lot,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
