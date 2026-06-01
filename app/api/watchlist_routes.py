from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_watchlist_service
from app.models.watchlist_models import (
    WatchlistWorkspaceResponse,
    WatchlistWorkspaceSyncRequest,
)
from app.services.watchlist_service import WatchlistService

router = APIRouter()


@router.get(
    "/watchlist/workspace",
    response_model=WatchlistWorkspaceResponse,
    response_model_by_alias=True,
)
async def get_watchlist_workspace(
    include_quotes: bool = Query(default=True, alias="includeQuotes"),
    user_id: str = Depends(get_current_user_id),
    watchlist_service: WatchlistService = Depends(get_watchlist_service),
) -> WatchlistWorkspaceResponse:
    return await asyncio.to_thread(
        watchlist_service.get_workspace,
        user_id=user_id,
        include_quotes=include_quotes,
    )


@router.put(
    "/watchlist/workspace",
    response_model=WatchlistWorkspaceResponse,
    response_model_by_alias=True,
)
async def sync_watchlist_workspace(
    payload: WatchlistWorkspaceSyncRequest,
    user_id: str = Depends(get_current_user_id),
    watchlist_service: WatchlistService = Depends(get_watchlist_service),
) -> WatchlistWorkspaceResponse:
    try:
        return await asyncio.to_thread(
            watchlist_service.sync_workspace,
            user_id=user_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
