from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.momentum_breakout_feature_guards import require_mb_alerts_enabled
from app.dependencies.service_dependencies import get_momentum_breakout_scanner_service
from app.models.momentum_breakout_scan_models import MomentumBreakoutScanResponse
from app.services.strategy.momentum_breakout_scanner_service import (
    MomentumBreakoutScannerService,
)

router = APIRouter(dependencies=[Depends(require_mb_alerts_enabled)])


@router.get(
    "/strategy/momentum-breakout/scan",
    response_model=MomentumBreakoutScanResponse,
    response_model_by_alias=True,
)
async def scan_momentum_breakout_candidates(
    symbols: str | None = Query(
        default=None,
        description="Optional comma-separated symbols; default is ranking universe",
    ),
    limit: int = Query(50, ge=1, le=200),
    service: MomentumBreakoutScannerService = Depends(get_momentum_breakout_scanner_service),
) -> MomentumBreakoutScanResponse:
    try:
        return await asyncio.to_thread(service.scan, symbols=symbols, limit=limit)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/strategy/momentum-breakout/top-candidates",
    response_model=MomentumBreakoutScanResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_top_candidates(
    service: MomentumBreakoutScannerService = Depends(get_momentum_breakout_scanner_service),
) -> MomentumBreakoutScanResponse:
    try:
        return await asyncio.to_thread(service.top_candidates)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
