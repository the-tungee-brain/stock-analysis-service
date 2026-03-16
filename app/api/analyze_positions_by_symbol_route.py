from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse
from app.models.schwab_models import Position
from app.services.llm_service import LLMService
from app.dependencies.service_dependencies import get_llm_service
from openai.types.shared import ResponsesModel

router = APIRouter()


class AnalyzePositionsBySymbolRequest(BaseModel):
    positions: List[Position]
    prompt: Optional[str] = None
    model: Optional[ResponsesModel] = "gpt-4.1-mini"


@router.post("/analyze-positions-by-symbol")
async def analyze_positions_by_symbol(
    request: AnalyzePositionsBySymbolRequest,
    llm_service: LLMService = Depends(get_llm_service),
):
    async def streamer():
        async for chunk in llm_service.analyze_option_position(
            model=request.model,
            input_prompt=request.prompt,
            positions=request.positions,
        ):
            yield chunk

    return StreamingResponse(
        streamer(),
        media_type="text/plain; charset=utf-8",
    )
