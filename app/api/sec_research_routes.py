from typing import Literal

from fastapi import APIRouter, Depends, Query

from app.dependencies.service_dependencies import get_sec_research_service
from app.models.sec_research_models import (
    SecFilingsResponse,
    SecFinancialsResponse,
    SecLookupResponse,
    SecRatiosResponse,
)
from app.services.sec_research_service import SecResearchService

router = APIRouter(prefix="/research/sec", tags=["SEC Research"])


@router.get("/lookup", response_model=SecLookupResponse)
def sec_lookup(
    symbol: str,
    sec_research_service: SecResearchService = Depends(get_sec_research_service),
):
    """Resolve a ticker symbol to SEC CIK and company metadata."""
    return sec_research_service.lookup(symbol=symbol)


@router.get("/filings", response_model=SecFilingsResponse)
def sec_filings(
    symbol: str,
    limit: int = Query(default=20, ge=1, le=100),
    sec_research_service: SecResearchService = Depends(get_sec_research_service),
):
    """Recent SEC filings for a symbol (10-K, 10-Q, 8-K, etc.)."""
    return sec_research_service.filings(symbol=symbol, limit=limit)


@router.get("/financials", response_model=SecFinancialsResponse)
def sec_financials(
    symbol: str,
    period: Literal["annual", "quarterly"] = Query(default="annual"),
    limit: int = Query(default=12, ge=1, le=40),
    sec_research_service: SecResearchService = Depends(get_sec_research_service),
):
    """Income statement, balance sheet, and cash flow time series from SEC XBRL."""
    return sec_research_service.financials(
        symbol=symbol, period=period, limit=limit
    )


@router.get("/ratios", response_model=SecRatiosResponse)
def sec_ratios(
    symbol: str,
    period: Literal["annual", "quarterly"] = Query(default="annual"),
    limit: int = Query(default=12, ge=1, le=40),
    sec_research_service: SecResearchService = Depends(get_sec_research_service),
):
    """Derived margins, returns, leverage, FCF, and YoY growth from SEC filings."""
    return sec_research_service.ratios(symbol=symbol, period=period, limit=limit)
