import asyncio
from typing import Callable, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_portfolio_analysis_service,
    get_portfolio_service,
    get_schwab_auth_service,
)
from app.models.symbol_analysis_precomputed_models import SymbolAnalysisPrecomputed
from app.models.schwab_models import Position, SchwabAccounts
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired

router = APIRouter()

T = TypeVar("T")


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


async def _run_sync(work: Callable[[], T]) -> T:
    return await asyncio.to_thread(work)


@router.get(
    "/research/symbol-analysis-precomputed",
    response_model=SymbolAnalysisPrecomputed | None,
    response_model_by_alias=True,
    summary="Precomputed roll/close/hold outcomes for symbol analyze UI",
    description=(
        "Returns server-computed option decision outcomes (roll, close, hold paths) "
        "for the Compare paths card. Use alongside POST /analyze-positions-by-symbol "
        "with response_format=portfolio_analysis_v1, which wraps the same data in "
        "analysis.precomputed when available."
    ),
)
async def get_symbol_analysis_precomputed(
    symbol: str = Query(..., min_length=1, max_length=12),
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    portfolio_analysis_service: PortfolioAnalysisService = Depends(
        get_portfolio_analysis_service
    ),
) -> SymbolAnalysisPrecomputed | None:
    symbol_upper = symbol.strip().upper()

    def load() -> SymbolAnalysisPrecomputed | None:
        try:
            schwab_token = schwab_auth_service.get_valid_token_by_user_id(
                user_id=user_id
            )
        except SchwabReauthRequired as exc:
            raise HTTPException(
                status_code=401,
                detail=schwab_auth_service.reauth_http_detail(user_id, exc),
            ) from exc

        account_map = portfolio_service.get_enriched_account(
            access_token=schwab_token.access_token
        )
        account: SchwabAccounts = account_map["account"]
        positions = account.securitiesAccount.positions

        return portfolio_analysis_service.build_symbol_analysis_precomputed(
            user_id=user_id,
            symbol=symbol_upper,
            account=account,
            positions=positions,
            access_token=schwab_token.access_token,
        )

    return await _run_sync(load)
