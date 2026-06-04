from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.momentum_breakout_feature_guards import require_mb_admin_metrics_access
from app.dependencies.service_dependencies import (
    get_momentum_breakout_admin_metrics_service,
)
from app.models.momentum_breakout_feature_models import (
    MomentumBreakoutAdminMetricsResponse,
    MomentumBreakoutStatusCountsDto,
)
from app.services.strategy.momentum_breakout_admin_metrics_service import (
    MomentumBreakoutAdminMetricsService,
)

router = APIRouter()


def _verify_admin_token(x_mb_admin_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("MB_ADMIN_TOKEN", "").strip()
    if not expected:
        return
    if not x_mb_admin_token or x_mb_admin_token != expected:
        raise HTTPException(status_code=403, detail="Admin metrics require MB_ADMIN_TOKEN.")


@router.get(
    "/strategy/momentum-breakout/admin/metrics",
    response_model=MomentumBreakoutAdminMetricsResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_mb_admin_metrics_access)],
)
async def get_momentum_breakout_admin_metrics(
    _: None = Depends(_verify_admin_token),
    service: MomentumBreakoutAdminMetricsService = Depends(
        get_momentum_breakout_admin_metrics_service
    ),
) -> MomentumBreakoutAdminMetricsResponse:
    snapshot = await asyncio.to_thread(service.snapshot)
    counts = snapshot.status_counts
    return MomentumBreakoutAdminMetricsResponse(
        alertsCreatedToday=snapshot.alerts_created_today,
        activeAlertsCount=snapshot.active_alerts_count,
        statusCounts=MomentumBreakoutStatusCountsDto(
            pendingEntry=counts.pending_entry,
            entryTriggered=counts.entry_triggered,
            open=counts.open,
            targetHit=counts.target_hit,
            stopHit=counts.stop_hit,
            expired=counts.expired,
            cancelled=counts.cancelled,
            completed=counts.completed,
        ),
        notificationsEmittedToday=snapshot.notifications_emitted_today,
        schedulerLastRunAt=snapshot.scheduler_last_run_at,
        schedulerLastError=snapshot.scheduler_last_error,
        paperTradeRowsCount=snapshot.paper_trade_rows_count,
        readinessReady=snapshot.readiness_ready,
        readinessWarnings=list(snapshot.readiness_warnings),
    )
