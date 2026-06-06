import asyncio

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_pattern_analysis_service,
    get_pattern_loaded_model,
    get_research_events_service,
)
from app.models.trader_playbook_models import TraderPlaybookResponse
from app.services.pattern_analysis_service import PatternAnalysisService
from app.services.research_events_service import ResearchEventsService
from app.services.trader_playbook_service import build_trader_playbook

router = APIRouter()


@router.get(
    "/research/trader-playbook",
    response_model=TraderPlaybookResponse,
    response_model_by_alias=True,
)
async def get_trader_playbook(
    symbol: str = Query(..., min_length=1, max_length=12),
    user_id: str = Depends(get_current_user_id),
    pattern_analysis_service: PatternAnalysisService = Depends(
        get_pattern_analysis_service
    ),
    loaded_model=Depends(get_pattern_loaded_model),
    research_events_service: ResearchEventsService = Depends(
        get_research_events_service
    ),
):
    del user_id
    return await asyncio.to_thread(
        build_trader_playbook,
        symbol,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
        research_events_service=research_events_service,
    )
