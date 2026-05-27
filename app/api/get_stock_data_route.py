import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.dependencies.adapter_dependencies import get_yfinance_adapter

router = APIRouter()


@router.get("/get-stock-data")
async def get_stock_data(
    symbol: str,
    period: str = Query("1mo", description="1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query(
        "1d", description="1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo"
    ),
    yfinance_adapter: YFinanceAdapter = Depends(get_yfinance_adapter),
):
    symbol_upper = symbol.strip().upper()

    try:
        return await asyncio.to_thread(
            yfinance_adapter.get_stock_chart_payload,
            symbol_upper,
            period=period,
            interval=interval,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
