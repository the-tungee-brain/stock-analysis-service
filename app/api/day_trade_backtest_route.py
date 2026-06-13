from __future__ import annotations

import asyncio
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.dependencies.adapter_dependencies import get_yfinance_adapter
from app.models.day_trade_backtest_models import DayTradeBacktestResponse
from app.services.day_trade_backtest_service import (
    DayTradeBacktestDataError,
    DayTradeBacktestService,
)

router = APIRouter()


@router.get(
    "/research/day-trade/backtest",
    response_model=DayTradeBacktestResponse,
    response_model_by_alias=True,
)
async def get_day_trade_backtest(
    symbol: str = Query(..., min_length=1, max_length=12),
    start: date = Query(...),
    end: date = Query(...),
    risk_per_trade: float = Query(100.0, ge=1),
    yfinance_adapter: YFinanceAdapter = Depends(get_yfinance_adapter),
) -> DayTradeBacktestResponse:
    service = DayTradeBacktestService(yfinance_adapter)
    try:
        return await asyncio.to_thread(
            service.run_backtest,
            symbol=symbol,
            start=start,
            end=end,
            risk_per_trade=risk_per_trade,
        )
    except DayTradeBacktestDataError as exc:
        raise HTTPException(status_code=400, detail=exc.to_detail()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
