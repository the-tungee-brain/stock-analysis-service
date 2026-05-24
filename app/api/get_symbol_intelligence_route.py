import asyncio

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_portfolio_analysis_service,
    get_portfolio_service,
    get_schwab_auth_service,
)
from app.models.intelligence_models import SymbolIntelligence
from app.models.schwab_models import Position, SchwabAccounts
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired

router = APIRouter()


def _positions_for_symbol(positions: list[Position], symbol: str) -> list[Position]:
    symbol_upper = symbol.upper()
    matched: list[Position] = []

    for position in positions:
        instrument = position.instrument
        if instrument.assetType == "OPTION":
            underlying = (instrument.underlyingSymbol or instrument.symbol or "").upper()
            if underlying == symbol_upper:
                matched.append(position)
        elif instrument.symbol.upper() == symbol_upper:
            matched.append(position)

    return matched


@router.get("/research/intelligence", response_model=SymbolIntelligence)
async def get_symbol_intelligence(
    symbol: str = Query(..., min_length=1, max_length=12),
    include_options: bool = Query(
        default=True,
        description="Include Schwab option chain scoring when linked",
    ),
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    portfolio_analysis_service: PortfolioAnalysisService = Depends(
        get_portfolio_analysis_service
    ),
) -> SymbolIntelligence:
    symbol_upper = symbol.strip().upper()
    account: SchwabAccounts | None = None
    positions: list[Position] = []
    access_token: str | None = None

    try:
        schwab_token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
        access_token = schwab_token.access_token
        account_map = portfolio_service.get_enriched_account(
            access_token=access_token
        )
        account = account_map["account"]
        positions = _positions_for_symbol(
            account.securitiesAccount.positions,
            symbol_upper,
        )
    except SchwabReauthRequired:
        account = None
        positions = []
        access_token = None
    except Exception:
        account = None
        positions = []
        access_token = None

    return await asyncio.to_thread(
        portfolio_analysis_service.build_symbol_intelligence,
        user_id=user_id,
        symbol=symbol_upper,
        account=account,
        positions=positions,
        access_token=access_token,
        include_options=include_options and access_token is not None,
    )
