import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_company_research_service,
    get_portfolio_service,
    get_schwab_auth_service,
)
from app.models.position_guidance_models import (
    PortfolioExitAttentionResponse,
    SymbolPositionGuidanceResponse,
)
from app.services.company_research_service import CompanyResearchService
from app.services.portfolio_service import PortfolioService
from app.services.position_guidance_service import (
    build_portfolio_exit_attention,
    build_symbol_position_guidance,
    load_portfolio_for_user,
)
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired

router = APIRouter()


@router.get(
    "/research/position-guidance",
    response_model=SymbolPositionGuidanceResponse,
    response_model_by_alias=True,
)
async def get_position_guidance(
    symbol: str,
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    research_service: CompanyResearchService = Depends(get_company_research_service),
) -> SymbolPositionGuidanceResponse:
    try:
        token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
    except SchwabReauthRequired as exc:
        raise HTTPException(
            status_code=401,
            detail=schwab_auth_service.reauth_http_detail(user_id, exc),
        ) from exc

    enriched = await asyncio.to_thread(
        load_portfolio_for_user,
        user_id=user_id,
        portfolio_service=portfolio_service,
        access_token=token.access_token,
    )
    positions_by_symbol = enriched["positions"]
    account = enriched["account"]
    symbol_upper = symbol.strip().upper()
    positions = positions_by_symbol.get(symbol_upper, [])

    return await asyncio.to_thread(
        build_symbol_position_guidance,
        symbol=symbol_upper,
        positions=positions,
        account=account,
        research_service=research_service,
        all_positions=account.securitiesAccount.positions,
    )


@router.get(
    "/research/equity-exit-guidance",
    response_model=SymbolPositionGuidanceResponse,
    response_model_by_alias=True,
    deprecated=True,
)
async def get_equity_exit_guidance_legacy(
    symbol: str,
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    research_service: CompanyResearchService = Depends(get_company_research_service),
) -> SymbolPositionGuidanceResponse:
    return await get_position_guidance(
        symbol=symbol,
        user_id=user_id,
        portfolio_service=portfolio_service,
        schwab_auth_service=schwab_auth_service,
        research_service=research_service,
    )


@router.get(
    "/portfolio/exit-attention",
    response_model=PortfolioExitAttentionResponse,
    response_model_by_alias=True,
)
async def get_portfolio_exit_attention(
    user_id: str = Depends(get_current_user_id),
    limit: int = Query(default=10, ge=1, le=25),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    research_service: CompanyResearchService = Depends(get_company_research_service),
) -> PortfolioExitAttentionResponse:
    try:
        token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)
    except SchwabReauthRequired as exc:
        raise HTTPException(
            status_code=401,
            detail=schwab_auth_service.reauth_http_detail(user_id, exc),
        ) from exc

    enriched = await asyncio.to_thread(
        load_portfolio_for_user,
        user_id=user_id,
        portfolio_service=portfolio_service,
        access_token=token.access_token,
    )

    return await asyncio.to_thread(
        build_portfolio_exit_attention,
        positions_by_symbol=enriched["positions"],
        account=enriched["account"],
        research_service=research_service,
        limit=limit,
    )
