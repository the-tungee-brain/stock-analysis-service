from fastapi import APIRouter, Depends
from app.models.company_research_models import PerformanceSnapshot
from app.services.market_service import MarketService
from app.dependencies.service_dependencies import get_market_service

router = APIRouter()


@router.get("/research/performance", response_model=PerformanceSnapshot)
def get_performance_snapshot(
    symbol: str, market_service: MarketService = Depends(get_market_service)
):
    return market_service.get_performance(symbol=symbol)
