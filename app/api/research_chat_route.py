from typing import List, Optional
import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from openai.types.shared import ResponsesModel
from pydantic import BaseModel

from app.auth.dependencies import get_current_user_id
from app.core.llm_config import settings
from app.dependencies.service_dependencies import (
    get_chat_service,
    get_company_research_service,
    get_llm_service,
    get_prompt_enrichment_service,
)
from app.services.chat_service import ChatService
from app.services.company_research_service import CompanyResearchService
from app.services.llm_service import LLMService
from app.services.prompt_enrichment_service import (
    RESEARCH_CHAT_SYSTEM_MESSAGE,
    PromptEnrichmentService,
)

router = APIRouter()


class ResearchChatRequest(BaseModel):
    symbol: str
    prompt: str
    model: Optional[ResponsesModel] = "gpt-4.1-mini"


@router.post("/research/chat")
async def research_chat(
    request: ResearchChatRequest,
    user_id: str = Depends(get_current_user_id),
    company_research_service: CompanyResearchService = Depends(
        get_company_research_service
    ),
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
    chat_service: ChatService = Depends(get_chat_service),
    llm_service: LLMService = Depends(get_llm_service),
):
    symbol = request.symbol.strip().upper()
    prompt = request.prompt.strip()
    if not symbol or not prompt:
        return StreamingResponse(
            iter(["Please provide a symbol and a question."]),
            media_type="text/plain; charset=utf-8",
        )

    ctx = await asyncio.to_thread(
        company_research_service.build_context,
        symbol=symbol,
    )

    session_id, is_first_chat = chat_service.get_research_chat_session_id(
        user_id=user_id,
        symbol=symbol,
        prompt=prompt,
        model=request.model,
    )
    recent_messages = chat_service.get_chat_messages_by_session(session_id=session_id)

    user_message = prompt_enrichment_service.build_research_chat_user_message(
        ctx=ctx,
        user_prompt=prompt,
        include_context=is_first_chat,
    )

    if session_id:
        chat_service.create_message(
            session_id=session_id,
            role=user_message["role"],
            content=user_message["content"],
        )

    assistant_content_parts: List[str] = []

    async def streamer():
        async for chunk in llm_service.analyze_option_position(
            model=request.model or settings.OPENAI_MODEL,
            system_prompt=RESEARCH_CHAT_SYSTEM_MESSAGE,
            user_prompt=[*recent_messages, user_message],
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
    )
