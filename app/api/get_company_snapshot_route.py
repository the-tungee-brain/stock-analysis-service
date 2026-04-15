from fastapi import APIRouter, Depends
from app.models.company_research_models import ResearchSnapshot
from app.services.company_profile_service import CompanyProfileService
from app.dependencies.service_dependencies import get_company_profile_service

router = APIRouter()


@router.get("/research/snapshot", response_model=ResearchSnapshot)
async def snapshot(
    symbol: str,
    company_profile_service: CompanyProfileService = Depends(
        get_company_profile_service
    ),
):
    return company_profile_service.get_snapshot(symbol=symbol)
