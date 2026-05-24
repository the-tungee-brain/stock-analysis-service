from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_portfolio_service,
    get_schwab_auth_service,
    get_transaction_service,
    get_portfolio_analysis_service,
)
from app.broker.option_utils import summarize_assignment_risk_structural
from app.models.intelligence_models import PortfolioIntelligence, ProactiveAlert
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.models.recent_order_models import RecentActivitySummary
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired
from app.services.transaction_service import DEFAULT_DAYS_BACK, TransactionService

router = APIRouter()


@router.get("/get-account-positions")
def get_account_positions(
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    transaction_service: TransactionService = Depends(get_transaction_service),
    portfolio_analysis_service: PortfolioAnalysisService = Depends(
        get_portfolio_analysis_service
    ),
    refresh: bool = Query(
        default=False,
        description="Bypass cached order history and fetch fresh data from Schwab",
    ),
):
    try:
        schwab_token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
    except SchwabReauthRequired as exc:
        auth_url = schwab_auth_service.build_authorization_url(state=user_id)
        raise HTTPException(
            status_code=401,
            detail={
                "message": str(exc),
                "reauth_required": True,
                "authorization_url": auth_url,
            },
        )

    account_map = portfolio_service.get_enriched_account(
        access_token=schwab_token.access_token
    )
    account_number = account_map["account"].securitiesAccount.accountNumber

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

    proactive_alerts: list[ProactiveAlert] = []
    portfolio_brief: PortfolioIntelligence | None = None
    try:
        portfolio_brief = portfolio_analysis_service.build_portfolio_brief(
            user_id=user_id,
            account=account_map["account"],
            positions=account_map["account"].securitiesAccount.positions,
            access_token=schwab_token.access_token,
            suggested_actions=(
                recent_activity.suggested_actions if recent_activity else []
            ),
            assignment_risk_summary=account_map["assignmentRiskSummary"],
        )
        proactive_alerts = portfolio_brief.alerts
    except Exception:
        portfolio_brief = None
        proactive_alerts = []

    portfolio_brief_payload = (
        portfolio_brief.model_dump(mode="json", by_alias=True)
        if portfolio_brief is not None
        else None
    )
    proactive_alerts_payload = [
        alert.model_dump(mode="json", by_alias=True) for alert in proactive_alerts
    ]

    return {
        "schwab_positions": account_map["positions"],
        "account": account_map["account"],
        "cashSecuredPutSummary": account_map["cashSecuredPutSummary"],
        "assignmentRiskSummary": account_map["assignmentRiskSummary"],
        "recentActivity": recent_activity,
        "proactiveAlerts": proactive_alerts_payload,
        "portfolioBrief": portfolio_brief_payload,
    }
