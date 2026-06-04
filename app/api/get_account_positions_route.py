from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_portfolio_analysis_service,
    get_portfolio_memory_service,
    get_portfolio_service,
    get_schwab_auth_service,
    get_transaction_service,
)
from app.broker.option_utils import summarize_assignment_risk_structural
from app.core.latency_observability import set_latency_attribute
from app.models.intelligence_models import PortfolioIntelligence, ProactiveAlert
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.models.recent_order_models import RecentActivitySummary
from app.services.portfolio_memory_service import PortfolioMemoryService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired
from app.services.transaction_service import DEFAULT_DAYS_BACK, TransactionService

logger = logging.getLogger(__name__)

router = APIRouter()


def _persist_portfolio_memory(
    *,
    portfolio_memory_service: PortfolioMemoryService,
    user_id: str,
    account,
    positions,
    portfolio_brief: PortfolioIntelligence | None,
    proactive_alerts: list[ProactiveAlert],
) -> None:
    if portfolio_brief is not None:
        portfolio_memory_service.capture_snapshot(
            user_id=user_id,
            account=account,
            positions=positions,
            portfolio_brief=portfolio_brief,
        )
    portfolio_memory_service.record_alerts(
        user_id=user_id,
        alerts=proactive_alerts,
    )


def _warm_portfolio_brief_cache(
    *,
    portfolio_analysis_service: PortfolioAnalysisService,
    portfolio_memory_service: PortfolioMemoryService,
    user_id: str,
    account,
    positions,
    access_token: str,
    suggested_actions: list,
    assignment_risk_summary: dict[str, object] | None,
) -> None:
    try:
        brief = portfolio_analysis_service.build_portfolio_brief_for_positions_load(
            user_id=user_id,
            account=account,
            positions=positions,
            access_token=access_token,
            suggested_actions=suggested_actions,
            assignment_risk_summary=assignment_risk_summary,
            refresh=False,
        )
        _persist_portfolio_memory(
            portfolio_memory_service=portfolio_memory_service,
            user_id=user_id,
            account=account,
            positions=positions,
            portfolio_brief=brief,
            proactive_alerts=brief.alerts,
        )
    except Exception:
        logger.exception("Background portfolio brief warm failed for user %s", user_id)


def _serialize_positions(positions_by_symbol: dict) -> dict:
    return {
        symbol: [position.model_dump(mode="json") for position in positions]
        for symbol, positions in positions_by_symbol.items()
    }


@router.get("/get-account-positions")
async def get_account_positions(
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    transaction_service: TransactionService = Depends(get_transaction_service),
    portfolio_analysis_service: PortfolioAnalysisService = Depends(
        get_portfolio_analysis_service
    ),
    portfolio_memory_service: PortfolioMemoryService = Depends(
        get_portfolio_memory_service
    ),
    refresh: bool = Query(
        default=False,
        description="Bypass cached order history and fetch fresh data from Schwab",
    ),
):
    try:
        schwab_token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
    except SchwabReauthRequired as exc:
        raise HTTPException(
            status_code=401,
            detail=schwab_auth_service.reauth_http_detail(user_id, exc),
        )

    account_map = portfolio_service.get_enriched_account(
        access_token=schwab_token.access_token
    )
    account = account_map["account"]
    positions = account.securitiesAccount.positions
    set_latency_attribute("account_count", 1)
    set_latency_attribute("symbol_count", len(positions))
    account_number = account.securitiesAccount.accountNumber
    positions_synced_at = datetime.now(timezone.utc).isoformat()

    recent_activity: RecentActivitySummary | None = None
    try:
        recent_activity = transaction_service.build_recent_activity_summary(
            account_number=account_number,
            access_token=schwab_token.access_token,
            user_id=user_id,
            days_back=DEFAULT_DAYS_BACK,
            refresh=refresh,
        )
    except Exception:
        recent_activity = None

    suggested_actions = (
        recent_activity.suggested_actions if recent_activity else []
    )
    assignment_risk_summary = account_map.get("assignmentRiskSummary")

    proactive_alerts: list[ProactiveAlert] = []
    portfolio_brief: PortfolioIntelligence | None = None
    brief_status = "pending"

    if refresh:
        try:
            portfolio_brief = await asyncio.to_thread(
                portfolio_analysis_service.build_portfolio_brief_for_positions_load,
                user_id=user_id,
                account=account,
                positions=positions,
                access_token=schwab_token.access_token,
                suggested_actions=suggested_actions,
                assignment_risk_summary=assignment_risk_summary,
                refresh=True,
            )
            proactive_alerts = portfolio_brief.alerts
            brief_status = "ready"
        except Exception:
            logger.exception("Portfolio brief build failed on refresh for %s", user_id)
            portfolio_brief = None
            brief_status = "pending"
    else:
        portfolio_brief = portfolio_analysis_service.try_get_light_cached_portfolio_brief(
            user_id=user_id,
            account=account,
            positions=positions,
        )
        if portfolio_brief is not None:
            proactive_alerts = portfolio_brief.alerts
            brief_status = "cached"
        else:
            background_tasks.add_task(
                _warm_portfolio_brief_cache,
                portfolio_analysis_service=portfolio_analysis_service,
                portfolio_memory_service=portfolio_memory_service,
                user_id=user_id,
                account=account,
                positions=positions,
                access_token=schwab_token.access_token,
                suggested_actions=suggested_actions,
                assignment_risk_summary=assignment_risk_summary,
            )

    portfolio_brief_payload = (
        portfolio_brief.model_dump(mode="json", by_alias=True)
        if portfolio_brief is not None
        else None
    )
    proactive_alerts_payload = [
        alert.model_dump(mode="json", by_alias=True) for alert in proactive_alerts
    ]

    if portfolio_brief is not None:
        background_tasks.add_task(
            _persist_portfolio_memory,
            portfolio_memory_service=portfolio_memory_service,
            user_id=user_id,
            account=account,
            positions=positions,
            portfolio_brief=portfolio_brief,
            proactive_alerts=proactive_alerts,
        )

    portfolio_metrics = account_map["portfolioMetrics"]

    logger.info(
        "positions load user=%s refresh=%s brief_status=%s",
        user_id,
        refresh,
        brief_status,
    )

    return {
        "schwab_positions": _serialize_positions(account_map["positions"]),
        "account": account_map["account"],
        "cashSecuredPutSummary": account_map["cashSecuredPutSummary"],
        "assignmentRiskSummary": account_map["assignmentRiskSummary"],
        "recentActivity": recent_activity,
        "proactiveAlerts": proactive_alerts_payload,
        "portfolioBrief": portfolio_brief_payload,
        "portfolioMetrics": portfolio_metrics.model_dump(mode="json"),
        "dataFreshness": {
            "positionsSyncedAt": positions_synced_at,
            "positionsSource": "schwab",
            "briefStatus": brief_status,
        },
    }
