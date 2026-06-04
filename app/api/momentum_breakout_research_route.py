from __future__ import annotations

import asyncio
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.dependencies.service_dependencies import get_momentum_breakout_research_service
from app.models.momentum_breakout_research_models import (
    MomentumBreakoutResearchDashboardResponse,
)
from app.services.strategy.momentum_breakout_research_service import (
    MomentumBreakoutResearchService,
    dashboard_to_response,
)

router = APIRouter()


@router.get(
    "/research/momentum-breakout/dashboard",
    response_model=MomentumBreakoutResearchDashboardResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_research_dashboard(
    symbols: str = Query(
        ...,
        min_length=1,
        description="Comma-separated symbols, e.g. AAPL,MSFT,NVDA",
    ),
    start_date: date = Query(date(2000, 1, 1), alias="startDate"),
    end_date: date = Query(date(2024, 12, 31), alias="endDate"),
    service: MomentumBreakoutResearchService = Depends(
        get_momentum_breakout_research_service
    ),
) -> MomentumBreakoutResearchDashboardResponse:
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="startDate must be <= endDate")

    symbol_list = [part.strip() for part in symbols.split(",") if part.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="At least one symbol is required")

    try:
        dashboard = await asyncio.to_thread(
            service.run_research,
            symbol_list,
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return dashboard_to_response(dashboard)


@router.get("/research/momentum-breakout/export")
async def export_momentum_breakout_research_csv(
    export_type: str = Query(
        "trades",
        alias="exportType",
        description="trades | yearly | regime | walk_forward | bundle",
    ),
    symbols: str = Query(
        ...,
        min_length=1,
        description="Comma-separated symbols (required to run research before export)",
    ),
    start_date: date = Query(date(2000, 1, 1), alias="startDate"),
    end_date: date = Query(date(2024, 12, 31), alias="endDate"),
    service: MomentumBreakoutResearchService = Depends(
        get_momentum_breakout_research_service
    ),
) -> PlainTextResponse:
    symbol_list = [part.strip() for part in symbols.split(",") if part.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="At least one symbol is required")

    try:
        await asyncio.to_thread(
            service.run_research,
            symbol_list,
            start_date=start_date,
            end_date=end_date,
        )
        filename, content = await asyncio.to_thread(service.export_csv, export_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    media_type = "text/csv" if filename.endswith(".csv") else "text/plain"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return PlainTextResponse(content=content, media_type=media_type, headers=headers)
