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
    get_portfolio_analysis_service,
    get_portfolio_service,
    get_prompt_enrichment_service,
    get_schwab_auth_service,
)
from app.services.chat_service import ChatService
from app.services.company_research_service import CompanyResearchService
from app.services.llm_service import LLMService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired
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
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    portfolio_analysis_service: PortfolioAnalysisService = Depends(
        get_portfolio_analysis_service
    ),
):
    symbol = request.symbol.strip().upper()
    prompt = request.prompt.strip()
    if not symbol or not prompt:
        return StreamingResponse(
            iter(["Please provide a symbol and a question."]),
            media_type="text/plain; charset=utf-8",
        )

    session_id, is_first_chat = chat_service.get_research_chat_session_id(
        user_id=user_id,
        symbol=symbol,
        prompt=prompt,
        model=request.model,
    )
    recent_messages = chat_service.get_chat_messages_by_session(session_id=session_id)

    if session_id:
        chat_service.create_message(
            session_id=session_id,
            role="user",
            content=prompt,
        )

    assistant_content_parts: List[str] = []

    async def streamer():
        yield "Looking up company data and your question…\n\n"

        ctx = await asyncio.to_thread(
            company_research_service.build_context,
            symbol=symbol,
        )

        holdings_block = None
        intelligence_block = None
        option_chain_block = None
        try:
            schwab_token = schwab_auth_service.get_valid_token_by_user_id(
                user_id=user_id
            )
            account_map = portfolio_service.get_enriched_account(
                access_token=schwab_token.access_token
            )
            account = account_map["account"]
            positions = account.securitiesAccount.positions
            holdings_block, intelligence_block, option_chain_block = await asyncio.to_thread(
                portfolio_analysis_service.build_research_chat_holdings_context,
                user_id=user_id,
                symbol=symbol,
                account=account,
                positions=positions,
                access_token=schwab_token.access_token,
            )
        except SchwabReauthRequired:
            holdings_block = None
            intelligence_block = None
            option_chain_block = None
        except Exception:
            holdings_block = None
            intelligence_block = None
            option_chain_block = None

        user_message = prompt_enrichment_service.build_research_chat_user_message(
            ctx=ctx,
            user_prompt=prompt,
            include_context=is_first_chat,
            holdings_block=holdings_block,
            intelligence_block=intelligence_block,
            option_chain_block=option_chain_block,
        )

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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
