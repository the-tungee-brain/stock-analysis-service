import asyncio

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_pattern_analysis_service,
    get_pattern_loaded_model,
    get_research_events_service,
)
from app.models.trading_bias_models import TradingBiasResponse
from app.services.pattern_analysis_service import PatternAnalysisService
from app.services.research_events_service import ResearchEventsService
from app.services.trading_bias_service import build_trading_bias

router = APIRouter()


@router.get(
    "/research/trading-bias",
    response_model=TradingBiasResponse,
    response_model_by_alias=True,
)
async def get_trading_bias(
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
        build_trading_bias,
        symbol,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
        research_events_service=research_events_service,
    )
