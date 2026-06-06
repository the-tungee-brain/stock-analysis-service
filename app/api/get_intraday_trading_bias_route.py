import asyncio

from fastapi import APIRouter, Depends, Query

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.auth.dependencies import get_current_user_id
from app.dependencies.adapter_dependencies import get_yfinance_adapter
from app.dependencies.service_dependencies import (
    get_pattern_analysis_service,
    get_pattern_loaded_model,
    get_research_events_service,
)
from app.models.intraday_trading_bias_models import IntradayTradingBiasResponse
from app.services.intraday_trading_bias_service import build_intraday_trading_bias
from app.services.pattern_analysis_service import PatternAnalysisService
from app.services.research_events_service import ResearchEventsService

router = APIRouter()


@router.get(
    "/research/intraday-trading-bias",
    response_model=IntradayTradingBiasResponse,
    response_model_by_alias=True,
)
async def get_intraday_trading_bias(
    symbol: str = Query(..., min_length=1, max_length=12),
    user_id: str = Depends(get_current_user_id),
    yfinance_adapter: YFinanceAdapter = Depends(get_yfinance_adapter),
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
        build_intraday_trading_bias,
        symbol,
        yfinance_adapter=yfinance_adapter,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
        research_events_service=research_events_service,
    )
