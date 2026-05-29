import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_portfolio_news_service,
    get_portfolio_service,
    get_schwab_auth_service,
)
from app.models.portfolio_news_models import PortfolioNewsResponse
from app.services.portfolio_news_service import PortfolioNewsService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired

router = APIRouter()


@router.get(
    "/portfolio/news",
    response_model=PortfolioNewsResponse,
    response_model_by_alias=True,
)
async def get_portfolio_news(
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    portfolio_news_service: PortfolioNewsService = Depends(get_portfolio_news_service),
) -> PortfolioNewsResponse:
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

    return await asyncio.to_thread(
        portfolio_news_service.build_portfolio_news,
        positions=positions,
        account=account,
    )
