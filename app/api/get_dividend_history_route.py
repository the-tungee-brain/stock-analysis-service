import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user_id
from app.core.llm_model_policy import is_paid_user
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
    user_id: str = Depends(get_current_user_id),
    symbol: str = Query(..., min_length=1, max_length=12),
    shares: float = Query(
        default=100.0,
        ge=0,
        le=1_000_000,
        description="Share count used for income and snowball scenarios",
    ),
    investment_usd: float | None = Query(
        default=None,
        ge=0,
        le=100_000_000,
        description="Optional dollar investment used to derive fractional shares",
    ),
    share_price: float | None = Query(
        default=None,
        gt=0,
        le=1_000_000,
        description="Share price paired with investment_usd to derive fractional shares",
    ),
    reinvest_dividends: bool = Query(
        default=False,
        description="Simulate dividend reinvestment with average annual price growth",
    ),
    price_cagr_pct: float | None = Query(
        default=None,
        ge=-99,
        le=500,
        description="Average annual price growth used for advanced DRIP simulation",
    ),
    project_years: int | None = Query(
        default=None,
        ge=1,
        le=50,
        description="Forward projection horizon in years from the current year",
    ),
    dividend_cagr_pct: float | None = Query(
        default=None,
        ge=-99,
        le=500,
        description="Average annual dividend growth used for forward projection",
    ),
    history_start_year: int | None = Query(
        default=None,
        ge=1980,
        le=2100,
        description="Optional first year for the historical cash-collected backtest",
    ),
    annual_contribution_usd: float = Query(
        default=0.0,
        ge=0,
        le=100_000_000,
        description="New cash invested at the start of each projected year (after year one)",
    ),
    dividend_research_service: DividendResearchService = Depends(
        get_dividend_research_service
    ),
) -> DividendHistoryContext:
    symbol_upper = symbol.strip().upper()
    include_snowball = is_paid_user(user_id)
    context = await asyncio.to_thread(
        dividend_research_service.build_history_context,
        symbol_upper,
        shares=shares,
        investment_usd=investment_usd,
        share_price=share_price,
        reinvest_dividends=reinvest_dividends if include_snowball else False,
        price_cagr_pct=price_cagr_pct if include_snowball else None,
        project_years=project_years if include_snowball else None,
        dividend_cagr_pct=dividend_cagr_pct if include_snowball else None,
        history_start_year=history_start_year if include_snowball else None,
        annual_contribution_usd=annual_contribution_usd if include_snowball else 0.0,
        include_snowball=include_snowball,
    )
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dividend history not found for {symbol_upper}",
        )
    return context
