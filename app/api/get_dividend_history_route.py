import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies.service_dependencies import get_dividend_research_service
from app.models.dividend_research_models import DividendHistoryContext
from app.services.dividend_research_service import DividendResearchService

router = APIRouter()


@router.get(
    "/research/dividends",
    response_model=DividendHistoryContext,
    response_model_by_alias=True,
)
async def get_dividend_history(
    symbol: str = Query(..., min_length=1, max_length=12),
    shares: float = Query(
        default=100.0,
        ge=0,
        le=1_000_000,
        description="Share count used for income and snowball scenarios",
    ),
    start_year: int | None = Query(
        default=None,
        ge=1980,
        le=2100,
        description="First calendar year included in the cash-collected scenario",
    ),
    dividend_research_service: DividendResearchService = Depends(
        get_dividend_research_service
    ),
) -> DividendHistoryContext:
    symbol_upper = symbol.strip().upper()
    context = await asyncio.to_thread(
        dividend_research_service.build_history_context,
        symbol_upper,
        shares=shares,
        start_year=start_year,
    )
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dividend history not found for {symbol_upper}",
        )
    return context
