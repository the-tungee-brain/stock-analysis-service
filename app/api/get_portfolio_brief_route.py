import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_portfolio_analysis_service,
    get_portfolio_service,
    get_schwab_auth_service,
    get_transaction_service,
)
from app.models.intelligence_models import PortfolioIntelligence
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired
from app.services.transaction_service import DEFAULT_DAYS_BACK, TransactionService
from app.broker.option_utils import summarize_assignment_risk_structural

router = APIRouter()


@router.get(
    "/portfolio/brief",
    response_model=PortfolioIntelligence,
    response_model_by_alias=True,
)
async def get_portfolio_brief(
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    transaction_service: TransactionService = Depends(get_transaction_service),
    portfolio_analysis_service: PortfolioAnalysisService = Depends(
        get_portfolio_analysis_service
    ),
    refresh: bool = Query(
        default=False,
        description="Bypass cached order history when loading suggested actions",
    ),
) -> PortfolioIntelligence:
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
    assignment_risk_summary = summarize_assignment_risk_structural(positions=positions)

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

    return await asyncio.to_thread(
        portfolio_analysis_service.build_portfolio_brief,
        user_id=user_id,
        account=account,
        positions=positions,
        access_token=schwab_token.access_token,
        suggested_actions=suggested_actions,
        assignment_risk_summary=assignment_risk_summary,
    )
