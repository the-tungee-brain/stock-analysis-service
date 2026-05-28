from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_portfolio_service,
    get_schwab_auth_service,
    get_transaction_service,
)
from app.models.recent_order_models import RecentOrdersResponse
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired
from app.services.transaction_service import (
    DEFAULT_DAYS_BACK,
    RECENT_ORDERS_PAGE_LIMIT,
    TransactionService,
)

router = APIRouter()


@router.get("/recent-orders", response_model=RecentOrdersResponse)
def get_recent_orders(
    user_id: str = Depends(get_current_user_id),
    symbol: Optional[str] = Query(default=None, description="Filter to one symbol"),
    days_back: int = Query(
        default=DEFAULT_DAYS_BACK,
        ge=1,
        le=60,
        description="How many days of filled orders to include (Schwab max 60)",
    ),
    limit: int = Query(
        default=RECENT_ORDERS_PAGE_LIMIT,
        ge=1,
        le=100,
        description="Maximum orders to return in this page",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of orders to skip (newest-first)",
    ),
    refresh: bool = Query(
        default=False,
        description="Bypass cached order history and fetch fresh data from Schwab",
    ),
    transaction_service: TransactionService = Depends(get_transaction_service),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
):
    try:
        schwab_token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
    except SchwabReauthRequired as exc:
        raise HTTPException(
            status_code=401,
            detail=schwab_auth_service.reauth_http_detail(user_id, exc),
        )

    account = portfolio_service.get_enriched_account(
        access_token=schwab_token.access_token
    )["account"]
    account_number = account.securitiesAccount.accountNumber

    return transaction_service.build_recent_orders_response(
        account_number=account_number,
        access_token=schwab_token.access_token,
        user_id=user_id,
        symbol=symbol,
        days_back=days_back,
        limit=limit,
        offset=offset,
        refresh=refresh,
    )
