import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_morning_brief_delivery_service,
    get_portfolio_analysis_service,
    get_portfolio_memory_service,
    get_portfolio_service,
    get_schwab_auth_service,
    get_transaction_service,
)
from app.broker.option_utils import summarize_assignment_risk_structural
from app.models.portfolio_memory_models import MorningBrief, PortfolioChanges
from app.services.morning_brief_delivery_service import MorningBriefDeliveryService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_memory_service import PortfolioMemoryService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired
from app.services.transaction_service import DEFAULT_DAYS_BACK, TransactionService

router = APIRouter()


@router.get(
    "/portfolio/changes",
    response_model=PortfolioChanges,
    response_model_by_alias=True,
)
async def get_portfolio_changes(
    user_id: str = Depends(get_current_user_id),
    portfolio_memory_service: PortfolioMemoryService = Depends(
        get_portfolio_memory_service
    ),
    compare_days: int = Query(
        default=1,
        ge=1,
        le=30,
        description="Compare current snapshot to the snapshot N days earlier",
    ),
) -> PortfolioChanges:
    return await asyncio.to_thread(
        portfolio_memory_service.get_portfolio_changes,
        user_id=user_id,
        compare_days=compare_days,
    )


@router.get("/portfolio/alerts/history")
async def get_alert_history(
    user_id: str = Depends(get_current_user_id),
    portfolio_memory_service: PortfolioMemoryService = Depends(
        get_portfolio_memory_service
    ),
    days: int = Query(default=30, ge=1, le=90),
):
    items = await asyncio.to_thread(
        portfolio_memory_service.list_alert_history,
        user_id=user_id,
        days=days,
    )
    return {
        "items": [item.model_dump(mode="json", by_alias=True) for item in items],
    }


@router.get("/portfolio/attention")
async def get_attention_queue(
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
    refresh: bool = Query(default=False),
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
    account_number = account.securitiesAccount.accountNumber

    suggested_actions = []
    try:
        recent_activity = transaction_service.build_recent_activity_summary(
            account_number=account_number,
            access_token=schwab_token.access_token,
            user_id=user_id,
            days_back=DEFAULT_DAYS_BACK,
            refresh=refresh,
        )
        if recent_activity:
            suggested_actions = recent_activity.suggested_actions
    except Exception:
        suggested_actions = []

    portfolio_brief = await asyncio.to_thread(
        portfolio_analysis_service.build_portfolio_brief,
        user_id=user_id,
        account=account,
        positions=positions,
        access_token=schwab_token.access_token,
        suggested_actions=suggested_actions,
        assignment_risk_summary=account_map["assignmentRiskSummary"],
    )

    queue = await asyncio.to_thread(
        portfolio_memory_service.build_attention_queue,
        user_id=user_id,
        current_alerts=portfolio_brief.alerts,
    )
    return {
        "items": [item.model_dump(mode="json", by_alias=True) for item in queue],
    }


@router.get(
    "/portfolio/morning-brief",
    response_model=MorningBrief,
    response_model_by_alias=True,
)
async def get_morning_brief(
    user_id: str = Depends(get_current_user_id),
    delivery_service: MorningBriefDeliveryService = Depends(
        get_morning_brief_delivery_service
    ),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    refresh: bool = Query(default=False),
) -> MorningBrief:
    brief = await asyncio.to_thread(
        delivery_service.build_for_user,
        user_id=user_id,
        refresh=refresh,
        persist=True,
    )
    if brief is None:
        raise HTTPException(
            status_code=401,
            detail=schwab_auth_service.reauth_http_detail(
                user_id, "Schwab re-authorization required."
            ),
        )
    return brief


@router.post("/portfolio/alerts/{alert_id}/dismiss")
async def dismiss_alert(
    alert_id: str,
    user_id: str = Depends(get_current_user_id),
    portfolio_memory_service: PortfolioMemoryService = Depends(
        get_portfolio_memory_service
    ),
):
    dismissed = await asyncio.to_thread(
        portfolio_memory_service.dismiss_alert,
        user_id=user_id,
        alert_id=alert_id,
    )
    if not dismissed:
        raise HTTPException(status_code=404, detail="Active alert not found")
    return {"dismissed": True}
