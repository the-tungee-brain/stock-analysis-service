from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse
from app.models.schwab_models import Position, SchwabAccounts
from app.services.llm_service import LLMService
from app.dependencies.service_dependencies import get_llm_service
from openai.types.shared import ResponsesModel
from app.core.prompts import (
    AnalysisAction,
    build_quick_prompt,
    build_option_prompt,
    build_portfolio_prompt,
)


router = APIRouter()


class AnalyzePositionsBySymbolRequest(BaseModel):
    account: SchwabAccounts
    positions: List[Position]
    symbol: Optional[str] = None
    prompt: Optional[str] = None
    action: AnalysisAction = AnalysisAction.FREE_FORM
    model: Optional[ResponsesModel] = "gpt-4.1-mini"


@router.post("/analyze-positions-by-symbol")
async def analyze_positions_by_symbol(
    request: AnalyzePositionsBySymbolRequest,
    llm_service: LLMService = Depends(get_llm_service),
):
    if not request.symbol:
        input_prompt = build_portfolio_prompt(
            prompt=request.prompt,
            account=request.account,
            positions=request.positions,
        )
    else:
        quick_prompt = build_quick_prompt(
            action=request.action,
            symbol=request.symbol,
            user_prompt=request.prompt,
        )
        input_prompt = build_option_prompt(
            prompt=quick_prompt,
            account=request.account,
            positions=request.positions,
        )

    async def streamer():
        async for chunk in llm_service.analyze_option_position(
            model=request.model,
            input_prompt=input_prompt,
            account=request.account,
            positions=request.positions,
        ):
            yield chunk

    return StreamingResponse(
        streamer(),
        media_type="text/plain; charset=utf-8",
    )
