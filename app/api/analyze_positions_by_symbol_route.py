from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse
from app.models.schwab_models import Position
from app.services.llm_service import LLMService
from app.dependencies.service_dependencies import get_llm_service
from openai.types.shared import ResponsesModel
from app.core.prompts import AnalysisAction, build_quick_prompt
from app.models.schwab_models import SchwabAccounts

router = APIRouter()


class AnalyzePositionsBySymbolRequest(BaseModel):
    account: SchwabAccounts
    positions: List[Position]
    symbol: Optional[str] = ""
    prompt: Optional[str] = None
    action: AnalysisAction = AnalysisAction.FREE_FORM
    model: Optional[ResponsesModel] = "gpt-4.1-mini"


@router.post("/analyze-positions-by-symbol")
async def analyze_positions_by_symbol(
    request: AnalyzePositionsBySymbolRequest,
    llm_service: LLMService = Depends(get_llm_service),
):
    input_prompt = build_quick_prompt(
        action=request.action,
        symbol=request.symbol,
        user_prompt=request.prompt,
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
