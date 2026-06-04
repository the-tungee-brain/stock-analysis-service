from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.momentum_breakout_feature_guards import require_mb_alerts_enabled
from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_momentum_breakout_notification_service,
)
from app.models.momentum_breakout_notification_models import (
    MarkNotificationReadResponse,
    MomentumBreakoutNotificationListResponse,
)
from app.notifications.composite_service import CompositeNotificationService
from app.services.strategy.momentum_breakout_notification_app import (
    NOTIFICATION_DISCLAIMER,
    notification_to_dto,
)

router = APIRouter(dependencies=[Depends(require_mb_alerts_enabled)])


@router.get(
    "/strategy/momentum-breakout/notifications",
    response_model=MomentumBreakoutNotificationListResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_notifications(
    unread_only: bool = Query(False, alias="unreadOnly"),
    limit: int = 100,
    user_id: str = Depends(get_current_user_id),
    notification_service: CompositeNotificationService = Depends(
        get_momentum_breakout_notification_service
    ),
) -> MomentumBreakoutNotificationListResponse:
    rows = await asyncio.to_thread(
        notification_service.list_notifications,
        user_id,
        unread_only=unread_only,
        limit=limit,
    )
    return MomentumBreakoutNotificationListResponse(
        disclaimer=NOTIFICATION_DISCLAIMER,
        notifications=[notification_to_dto(row) for row in rows],
    )


@router.post(
    "/strategy/momentum-breakout/notifications/{notification_id}/read",
    response_model=MarkNotificationReadResponse,
    response_model_by_alias=True,
)
async def post_mark_momentum_breakout_notification_read(
    notification_id: str,
    user_id: str = Depends(get_current_user_id),
    notification_service: CompositeNotificationService = Depends(
        get_momentum_breakout_notification_service
    ),
) -> MarkNotificationReadResponse:
    try:
        updated = await asyncio.to_thread(
            notification_service.mark_read,
            user_id,
            notification_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return MarkNotificationReadResponse(
        disclaimer=NOTIFICATION_DISCLAIMER,
        notification=notification_to_dto(updated),
    )
