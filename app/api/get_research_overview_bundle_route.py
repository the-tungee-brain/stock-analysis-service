from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_research_overview_service
from app.services.research_overview_service import (
    ResearchOverviewBundle,
    ResearchOverviewService,
)

router = APIRouter()


@router.get(
    "/research/overview-bundle",
    response_model=ResearchOverviewBundle,
    response_model_by_alias=True,
)
async def get_research_overview_bundle(
    symbol: str = Query(..., min_length=1, max_length=12),
    holdings_limit: int = Query(default=8, ge=1, le=25),
    user_id: str = Depends(get_current_user_id),
    overview_service: ResearchOverviewService = Depends(get_research_overview_service),
) -> ResearchOverviewBundle:
    return await overview_service.build_bundle_async(
        user_id=user_id,
        symbol=symbol,
        holdings_limit=holdings_limit,
    )
