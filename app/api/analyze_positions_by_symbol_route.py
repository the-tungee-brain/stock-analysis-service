from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse

from app.models.schwab_models import Position, SchwabAccounts
from app.services.llm_service import LLMService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.dependencies.service_dependencies import (
    get_llm_service,
    get_portfolio_analysis_service,
    get_prompt_enrichment_service,
)
from openai.types.shared import ResponsesModel
from app.core.prompts import (
    AnalysisAction,
    SYSTEM_MESSAGE,
    SYSTEM_NATURAL_MESSAGE,
)
from app.auth.dependencies import get_current_user_id
from app.core.llm_config import settings

router = APIRouter()


class AnalyzePositionsBySymbolRequest(BaseModel):
    account: SchwabAccounts
    positions: List[Position]
    session_id: Optional[str] = None
    symbol: Optional[str] = None
    prompt: Optional[str] = None
    action: AnalysisAction = AnalysisAction.FREE_FORM
    model: Optional[ResponsesModel] = "gpt-4.1-mini"


@router.post("/analyze-positions-by-symbol")
async def analyze_positions_by_symbol(
    request: AnalyzePositionsBySymbolRequest,
    user_id: str = Depends(get_current_user_id),
    llm_service: LLMService = Depends(get_llm_service),
    portfolio_analysis_service: PortfolioAnalysisService = Depends(
        get_portfolio_analysis_service
    ),
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
):
    ctx = await portfolio_analysis_service.build_analysis_context(
        user_id=user_id,
        account=request.account,
        positions=request.positions,
        session_id=request.session_id,
        symbol=request.symbol,
        user_prompt=request.prompt,
        action=request.action,
    )

    user_prompt = prompt_enrichment_service.build_portfolio_strategy_prompt(ctx=ctx)

    async def streamer():
        async for chunk in llm_service.analyze_option_position(
            model=request.model or settings.OPENAI_MODEL,
            system_prompt=SYSTEM_NATURAL_MESSAGE if request.prompt else SYSTEM_MESSAGE,
            user_prompt=user_prompt,
        ):
            yield chunk

    return StreamingResponse(
        streamer(),
        media_type="text/plain; charset=utf-8",
    )
