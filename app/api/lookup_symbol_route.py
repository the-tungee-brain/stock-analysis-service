from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies.service_dependencies import get_ticker_service
from app.models.ticker_symbol_models import TickerSymbolItem
from app.services.ticker_service import TickerService

router = APIRouter()


@router.get(
    "/symbols/lookup",
    response_model=TickerSymbolItem,
    response_model_by_alias=True,
)
def lookup_symbol(
    symbol: str = Query(..., min_length=1, max_length=16),
    ticker_service: TickerService = Depends(get_ticker_service),
) -> TickerSymbolItem:
    item = ticker_service.get_by_symbol(symbol=symbol)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol {symbol.strip().upper()} not found",
        )
    return item
