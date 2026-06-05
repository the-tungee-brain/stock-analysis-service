from fastapi import APIRouter, Depends
from app.models.company_research_models import ResearchSnapshot
from app.services.research_symbol_data_service import ResearchSymbolDataService
from app.dependencies.service_dependencies import get_research_symbol_data_service

router = APIRouter()


@router.get("/research/snapshot", response_model=ResearchSnapshot)
async def snapshot(
    symbol: str,
    research_symbol_data_service: ResearchSymbolDataService = Depends(
        get_research_symbol_data_service
    ),
):
    return research_symbol_data_service.get_snapshot(symbol=symbol)
