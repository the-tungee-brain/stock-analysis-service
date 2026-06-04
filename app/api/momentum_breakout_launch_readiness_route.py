from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Depends, Header

from app.api.momentum_breakout_feature_guards import require_mb_alerts_enabled
from app.auth.dependencies import get_current_user_id
from app.core.momentum_breakout_monitoring import log_mb_warning
from app.services.strategy.momentum_breakout_ops_metrics import (
    get_momentum_breakout_ops_metrics,
)
from app.dependencies.service_dependencies import (
    get_momentum_breakout_launch_readiness_service,
)
from app.models.momentum_breakout_launch_readiness_models import (
    HealthCheckDto,
    LaunchReadinessResponse,
)
from app.services.strategy.momentum_breakout_launch_readiness_service import (
    LaunchReadinessReport,
    MomentumBreakoutLaunchReadinessService,
)

router = APIRouter(dependencies=[Depends(require_mb_alerts_enabled)])


def _admin_diagnostics_allowed(x_mb_admin_token: str | None = Header(default=None)) -> bool:
    expected = os.getenv("MB_ADMIN_TOKEN", "").strip()
    if not expected:
        return os.getenv("MB_LAUNCH_READINESS_PUBLIC", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return bool(x_mb_admin_token and x_mb_admin_token == expected)


def _to_response(
    report: LaunchReadinessReport, *, admin: bool
) -> LaunchReadinessResponse:
    return LaunchReadinessResponse(
        adminDiagnostics=admin,
        ready=report.ready,
        alertStoreType=report.alert_store_type,
        paperTradeStoreType=report.paper_trade_store_type,
        schedulerEnabled=report.scheduler_enabled,
        refreshIntervalSec=report.refresh_interval_sec,
        rawRefreshIntervalSec=report.raw_refresh_interval_sec,
        oracleAlertDdlRequired=report.oracle_alert_ddl_required,
        oracleAlertDdlApplied=report.oracle_alert_ddl_applied,
        oraclePaperDdlRequired=report.oracle_paper_ddl_required,
        oraclePaperDdlApplied=report.oracle_paper_ddl_applied,
        ddlDetailAlert=report.ddl_detail_alert,
        ddlDetailPaper=report.ddl_detail_paper,
        activeAlertsCount=report.active_alerts_count,
        paperTradeRowsCount=report.paper_trade_rows_count,
        totalAlertsCount=report.total_alerts_count,
        alertsMissingPaperRows=report.alerts_missing_paper_rows,
        latestLifecycleUpdateAt=report.latest_lifecycle_update_at,
        latestPaperTradeUpdateAt=report.latest_paper_trade_update_at,
        quoteProviderAvailable=report.quote_provider_available,
        healthChecks=[
            HealthCheckDto(name=check.name, ok=check.ok, detail=check.detail)
            for check in report.health_checks
        ],
        warnings=report.warnings if admin else _public_warnings(report),
    )


def _public_warnings(report: LaunchReadinessReport) -> list[str]:
    return [
        "Detailed operational warnings require admin diagnostics access.",
    ]


@router.get(
    "/strategy/momentum-breakout/launch-readiness",
    response_model=LaunchReadinessResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_launch_readiness(
    user_id: str = Depends(get_current_user_id),
    admin: bool = Depends(_admin_diagnostics_allowed),
    service: MomentumBreakoutLaunchReadinessService = Depends(
        get_momentum_breakout_launch_readiness_service
    ),
) -> LaunchReadinessResponse:
    _ = user_id
    report = await asyncio.to_thread(service.evaluate)
    get_momentum_breakout_ops_metrics().record_readiness(
        ready=report.ready,
        warnings=tuple(report.warnings),
    )
    if not report.ready:
        log_mb_warning(
            "launch_readiness_failed",
            warnings=report.warnings,
            alert_store_type=report.alert_store_type,
            paper_trade_store_type=report.paper_trade_store_type,
        )
    return _to_response(report, admin=admin)
