from fastapi import APIRouter, Depends
from app.models.company_research_models import PerformanceSnapshot
from app.services.research_symbol_data_service import ResearchSymbolDataService
from app.dependencies.service_dependencies import get_research_symbol_data_service

router = APIRouter()


@router.get("/research/performance", response_model=PerformanceSnapshot)
def get_performance_snapshot(
    symbol: str,
    research_symbol_data_service: ResearchSymbolDataService = Depends(
        get_research_symbol_data_service
    ),
):
    return research_symbol_data_service.get_performance(symbol=symbol)
