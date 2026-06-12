import asyncio
from datetime import date

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_trade_replay_service
from app.models.trade_replay_models import (
    MissedMovesRange,
    MissedMovesSort,
    MissedMovesSummaryResponse,
    TradeReplayRefreshRequest,
    TradeReplayRefreshResponse,
    TradeReplayResponse,
    TradeReplayWorkflow,
)
from app.services.trade_replay_service import TradeReplayService

router = APIRouter()


@router.get(
    "/research/trade-replay/missed-moves",
    response_model=MissedMovesSummaryResponse,
    response_model_by_alias=True,
)
async def get_missed_moves_summary(
    symbol: str = Query(..., min_length=1, max_length=12),
    workflow: TradeReplayWorkflow = Query(...),
    range_: MissedMovesRange = Query(..., alias="range"),
    sort: MissedMovesSort = Query("most_recent"),
    user_id: str = Depends(get_current_user_id),
    service: TradeReplayService = Depends(get_trade_replay_service),
):
    del user_id
    return await asyncio.to_thread(
        service.list_missed_moves,
        symbol=symbol,
        workflow=workflow,
        range_=range_,
        sort=sort,
    )


@router.get(
    "/research/trade-replay",
    response_model=TradeReplayResponse,
    response_model_by_alias=True,
)
async def get_trade_replay(
    symbol: str = Query(..., min_length=1, max_length=12),
    workflow: TradeReplayWorkflow = Query(...),
    date_: date = Query(..., alias="date"),
    missed_move_id: str | None = Query(default=None),
    user_id: str = Depends(get_current_user_id),
    service: TradeReplayService = Depends(get_trade_replay_service),
):
    del user_id
    return await asyncio.to_thread(
        service.get_replay,
        symbol=symbol,
        workflow=workflow,
        event_date=date_,
        missed_move_id=missed_move_id,
    )


@router.post(
    "/research/trade-replay/refresh",
    response_model=TradeReplayRefreshResponse,
    response_model_by_alias=True,
)
async def refresh_trade_replay(
    request: TradeReplayRefreshRequest,
    user_id: str = Depends(get_current_user_id),
    service: TradeReplayService = Depends(get_trade_replay_service),
):
    del user_id
    return await asyncio.to_thread(
        service.refresh,
        symbol=request.symbol,
        workflow=request.workflow,
        event_date=request.date,
    )
