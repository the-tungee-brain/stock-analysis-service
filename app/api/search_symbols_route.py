from fastapi import APIRouter, Depends, Query
from app.services.ticker_service import TickerService
from app.models.ticker_symbol_models import TickerSymbolItem
from app.dependencies.service_dependencies import get_ticker_service
from typing import List

router = APIRouter()


@router.get("/symbols/search", response_model=List[TickerSymbolItem])
def search_symbols(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    ticker_service: TickerService = Depends(get_ticker_service),
):
    return ticker_service.get_symbols_by_keyword(keyword=q, limit=limit)
