import asyncio
from typing import Callable, TypeVar

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_portfolio_analysis_service,
    get_portfolio_service,
    get_schwab_auth_service,
)
from app.models.intelligence_models import SymbolIntelligence
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.symbol_intelligence_service import fetch_symbol_intelligence

router = APIRouter()

T = TypeVar("T")


async def _run_sync(work: Callable[[], T]) -> T:
    return await asyncio.to_thread(work)


@router.get(
    "/research/intelligence",
    response_model=SymbolIntelligence,
    response_model_by_alias=True,
)
async def get_symbol_intelligence(
    symbol: str = Query(..., min_length=1, max_length=12),
    include_options: bool = Query(
        default=True,
        description="Include Schwab option chain scoring when linked (heavier request)",
    ),
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    portfolio_analysis_service: PortfolioAnalysisService = Depends(
        get_portfolio_analysis_service
    ),
) -> SymbolIntelligence:
    symbol_upper = symbol.strip().upper()

    return await _run_sync(
        lambda: fetch_symbol_intelligence(
            user_id=user_id,
            symbol_upper=symbol_upper,
            include_options=include_options,
            portfolio_service=portfolio_service,
            schwab_auth_service=schwab_auth_service,
            portfolio_analysis_service=portfolio_analysis_service,
        )
    )
