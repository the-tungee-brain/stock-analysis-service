from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.api.momentum_breakout_feature_guards import (
    require_mb_alert_creation_enabled,
    require_mb_alerts_enabled,
)
from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_momentum_breakout_alert_refresh_service,
    get_momentum_breakout_alert_service,
)
from app.models.momentum_breakout_alert_models import (
    AlertStatusChangeDto,
    MomentumBreakoutAlertListResponse,
    MomentumBreakoutAlertDto,
    MomentumBreakoutAlertRequest,
    MomentumBreakoutAlertResponse,
    MomentumBreakoutAlertRefreshResponse,
    MomentumBreakoutPriceUpdateRequest,
)
from app.services.strategy.momentum_breakout_alert_lifecycle_app import (
    LIFECYCLE_DISCLAIMER,
    record_to_dto,
)
from app.services.strategy.momentum_breakout_alert_refresh_service import (
    MomentumBreakoutAlertRefreshService,
)
from app.services.strategy.momentum_breakout_alert_service import (
    MomentumBreakoutAlertService,
)
from trade_planner.alerts.lifecycle_store import AlertNotCancellableError

router = APIRouter(dependencies=[Depends(require_mb_alerts_enabled)])


@router.post(
    "/strategy/momentum-breakout/trade-plan-alert",
    response_model=MomentumBreakoutAlertResponse,
    response_model_by_alias=True,
)
async def post_momentum_breakout_trade_plan_alert(
    body: MomentumBreakoutAlertRequest,
    user_id: str = Depends(get_current_user_id),
    service: MomentumBreakoutAlertService = Depends(get_momentum_breakout_alert_service),
) -> MomentumBreakoutAlertResponse:
    symbol = body.symbol.strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    if body.persist_alert:
        require_mb_alert_creation_enabled()

    try:
        return await asyncio.to_thread(service.evaluate, body, user_id=user_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"OHLCV data not available for {symbol.upper()}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/strategy/momentum-breakout/alerts/active",
    response_model=MomentumBreakoutAlertListResponse,
    response_model_by_alias=True,
)
async def get_active_momentum_breakout_alerts(
    user_id: str = Depends(get_current_user_id),
    service: MomentumBreakoutAlertService = Depends(get_momentum_breakout_alert_service),
) -> MomentumBreakoutAlertListResponse:
    lifecycle = service.lifecycle_service
    records = await asyncio.to_thread(lifecycle.list_active_alerts, user_id)
    return MomentumBreakoutAlertListResponse(
        disclaimer=LIFECYCLE_DISCLAIMER,
        alerts=[
            record_to_dto(record, lifecycle=lifecycle, include_events=True)
            for record in records
        ],
    )


@router.get(
    "/strategy/momentum-breakout/alerts/history",
    response_model=MomentumBreakoutAlertListResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_alert_history(
    limit: int = 100,
    user_id: str = Depends(get_current_user_id),
    service: MomentumBreakoutAlertService = Depends(get_momentum_breakout_alert_service),
) -> MomentumBreakoutAlertListResponse:
    lifecycle = service.lifecycle_service
    records = await asyncio.to_thread(
        lifecycle.list_alert_history, user_id, limit=limit
    )
    return MomentumBreakoutAlertListResponse(
        disclaimer=LIFECYCLE_DISCLAIMER,
        alerts=[
            record_to_dto(record, lifecycle=lifecycle, include_events=True)
            for record in records
        ],
    )


@router.post(
    "/strategy/momentum-breakout/alerts/{alert_id}/cancel",
    response_model=MomentumBreakoutAlertDto,
    response_model_by_alias=True,
)
async def post_momentum_breakout_alert_cancel(
    alert_id: str,
    user_id: str = Depends(get_current_user_id),
    service: MomentumBreakoutAlertService = Depends(get_momentum_breakout_alert_service),
) -> MomentumBreakoutAlertDto:
    lifecycle = service.lifecycle_service
    try:
        updated = await asyncio.to_thread(
            lifecycle.cancel_alert,
            user_id,
            alert_id,
        )
    except AlertNotCancellableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return record_to_dto(updated, lifecycle=lifecycle, include_events=True)


@router.post(
    "/strategy/momentum-breakout/alerts/{alert_id}/price-update",
    response_model=MomentumBreakoutAlertDto,
    response_model_by_alias=True,
)
async def post_momentum_breakout_alert_price_update(
    alert_id: str,
    body: MomentumBreakoutPriceUpdateRequest,
    user_id: str = Depends(get_current_user_id),
    service: MomentumBreakoutAlertService = Depends(get_momentum_breakout_alert_service),
) -> MomentumBreakoutAlertDto:
    lifecycle = service.lifecycle_service
    try:
        updated = await asyncio.to_thread(
            lifecycle.update_with_latest_price,
            user_id,
            alert_id,
            symbol=body.symbol.strip().upper(),
            price=body.price,
            timestamp=body.timestamp,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return record_to_dto(updated, lifecycle=lifecycle, include_events=True)


@router.post(
    "/strategy/momentum-breakout/alerts/refresh",
    response_model=MomentumBreakoutAlertRefreshResponse,
    response_model_by_alias=True,
)
async def post_momentum_breakout_alerts_refresh(
    user_id: str = Depends(get_current_user_id),
    refresh_service: MomentumBreakoutAlertRefreshService = Depends(
        get_momentum_breakout_alert_refresh_service
    ),
    alert_service: MomentumBreakoutAlertService = Depends(
        get_momentum_breakout_alert_service
    ),
) -> MomentumBreakoutAlertRefreshResponse:
    result = await asyncio.to_thread(
        refresh_service.refresh_user_active_alerts,
        user_id,
        force=True,
    )
    lifecycle = alert_service.lifecycle_service
    active = await asyncio.to_thread(lifecycle.list_active_alerts, user_id)
    return MomentumBreakoutAlertRefreshResponse(
        disclaimer=LIFECYCLE_DISCLAIMER,
        processed=result.processed,
        updated=result.updated,
        skippedMarketHours=result.skipped_market_hours,
        warnings=list(result.warnings),
        changes=[
            AlertStatusChangeDto(
                alertId=change.alert_id,
                symbol=change.symbol,
                priorStatus=change.prior_status,
                newStatus=change.new_status,
            )
            for change in result.changes
        ],
        alerts=[
            record_to_dto(record, lifecycle=lifecycle, include_events=True)
            for record in active
        ],
    )
