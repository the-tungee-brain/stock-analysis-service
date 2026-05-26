from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator
from typing import List, Optional
from fastapi.responses import StreamingResponse

from app.models.schwab_models import Position, SchwabAccounts
from app.services.llm_service import LLMService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.portfolio_service import PortfolioService
from app.services.chat_service import ChatService
from app.dependencies.service_dependencies import (
    get_llm_service,
    get_portfolio_analysis_service,
    get_prompt_enrichment_service,
    get_chat_service,
)
from openai.types.shared import ResponsesModel
from app.core.prompts import (
    AnalysisAction,
    SYSTEM_NATURAL_MESSAGE,
    system_message_for_structured_analysis,
    system_message_for_structured_v1_analysis,
    should_use_natural_response,
    uses_structured_system_message,
)
from app.core.analysis_schema import wants_structured_analysis_v1
from app.core.llm_routes import LLMRoute
from app.models.analysis_models import PortfolioAnalysisV1LLMResponse
from app.auth.dependencies import get_current_user_id
from app.core.llm_config import settings

router = APIRouter()


class AnalyzePositionsBySymbolRequest(BaseModel):
    account: SchwabAccounts
    positions: List[Position]
    session_id: Optional[str] = None
    symbol: Optional[str] = None
    prompt: Optional[str] = None
    user_display_message: Optional[str] = None
    action: AnalysisAction = AnalysisAction.FREE_FORM
    model: Optional[ResponsesModel] = "gpt-4.1-mini"
    response_format: Optional[str] = None
    analysis_instructions: Optional[str] = None

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value: object) -> AnalysisAction:
        if isinstance(value, AnalysisAction):
            return value
        if value is None:
            return AnalysisAction.FREE_FORM
        return AnalysisAction.parse(str(value))


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
    chat_service: ChatService = Depends(get_chat_service),
):
    positions = PortfolioService._annotate_option_strategies(request.positions)

    structured = uses_structured_system_message(
        request.prompt,
        action=request.action,
    )
    json_v1 = wants_structured_analysis_v1(
        response_format=request.response_format,
        user_prompt=request.prompt,
        action=request.action,
    )

    session_id: Optional[str] = None
    is_first_chat = True
    recent_messages: list = []

    if not structured:
        session_prompt = chat_service.user_message_for_storage(
            prompt=request.user_display_message or request.prompt,
            action=request.action,
        )
        resolved_session_id, is_first_chat = chat_service.get_portfolio_analysis_session_id(
            user_id=user_id,
            symbol=request.symbol,
            prompt=session_prompt,
            model=request.model,
        )
        session_id = str(resolved_session_id) if resolved_session_id else None
        recent_messages = chat_service.get_chat_messages_by_session(session_id=session_id)

    include_context = chat_service.should_include_portfolio_context(
        is_first_chat=is_first_chat,
        action=request.action,
        recent_messages=recent_messages,
        user_prompt=request.prompt,
    )

    if session_id:
        chat_service.create_message(
            session_id=session_id,
            role="user",
            content=session_prompt,
        )

    assistant_content_parts: List[str] = []

    async def streamer():
        if json_v1:
            yield "Reviewing your portfolio…\n\n"
        else:
            yield "Pulling together your holdings and market context…\n\n"

        ctx = await portfolio_analysis_service.build_analysis_context(
            user_id=user_id,
            account=request.account,
            positions=positions,
            session_id=request.session_id,
            symbol=request.symbol,
            user_prompt=request.prompt,
            action=request.action,
            include_market_data=include_context,
        )

        user_prompt = prompt_enrichment_service.build_portfolio_strategy_prompt(
            ctx=ctx,
            include_context=include_context,
            json_response=json_v1,
        )
        if json_v1 and request.analysis_instructions:
            user_prompt = {
                "role": "user",
                "content": (
                    user_prompt["content"]
                    + "\n\n"
                    + request.analysis_instructions.strip()
                ),
            }

        if json_v1:
            system_prompt = system_message_for_structured_v1_analysis(
                symbol=request.symbol
            )
            parsed = await llm_service.generate_from_prompts(
                prompts=[system_prompt, user_prompt["content"]],
                response_model=PortfolioAnalysisV1LLMResponse,
                route=LLMRoute.NEWS,
                model=request.model or settings.OPENAI_MODEL,
                max_output_tokens=settings.MAX_OUTPUT_TOKENS_STREAM,
            )
            payload = parsed.model_dump_json()
            assistant_content_parts.append(payload)
            yield payload
            return

        system_prompt = (
            SYSTEM_NATURAL_MESSAGE
            if should_use_natural_response(request.prompt, action=request.action)
            else system_message_for_structured_analysis(symbol=request.symbol)
        )
        llm_history = [] if structured else recent_messages
        async for chunk in llm_service.analyze_option_position(
            model=request.model or settings.OPENAI_MODEL,
            system_prompt=system_prompt,
            user_prompt=[*llm_history, user_prompt],
        ):
            assistant_content_parts.append(chunk)
            yield chunk

        if session_id:
            assistant_content = "".join(assistant_content_parts)
            if assistant_content:
                chat_service.create_message(
                    session_id=session_id,
                    role="assistant",
                    content=assistant_content,
                )

    return StreamingResponse(
        streamer(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
