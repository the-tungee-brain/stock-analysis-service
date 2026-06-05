import asyncio

from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import (
    get_pattern_analysis_service,
    get_pattern_loaded_model,
)
from app.models.trade_decision_models import TradeDecision
from app.services.pattern_analysis_service import PatternAnalysisService
from app.services.trade_decision_service import build_trade_decision

router = APIRouter()


@router.get(
    "/research/trade-decision",
    response_model=TradeDecision,
    response_model_by_alias=True,
)
async def get_trade_decision(
    symbol: str,
    user_id: str = Depends(get_current_user_id),
    pattern_analysis_service: PatternAnalysisService = Depends(
        get_pattern_analysis_service
    ),
    loaded_model=Depends(get_pattern_loaded_model),
):
    del user_id
    return await asyncio.to_thread(
        build_trade_decision,
        symbol,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
    )
