import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies.service_dependencies import get_etf_research_service
from app.models.company_research_models import EtfHoldingsContext
from app.services.etf_research_service import EtfResearchService

router = APIRouter()


@router.get(
    "/research/etf-holdings",
    response_model=EtfHoldingsContext,
    response_model_by_alias=True,
)
async def get_etf_holdings(
    symbol: str = Query(..., min_length=1, max_length=12),
    limit: int = Query(
        default=25,
        ge=1,
        le=100,
        description="Maximum number of holdings to return",
    ),
    etf_research_service: EtfResearchService = Depends(get_etf_research_service),
) -> EtfHoldingsContext:
    symbol_upper = symbol.strip().upper()
    context = await asyncio.to_thread(
        etf_research_service.build_holdings_context,
        symbol_upper,
        holdings_limit=limit,
    )
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"ETF holdings not found for {symbol_upper}",
        )
    return context
