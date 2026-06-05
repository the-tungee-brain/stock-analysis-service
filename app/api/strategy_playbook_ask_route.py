from __future__ import annotations

import asyncio
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from openai.types.shared import ResponsesModel
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user_id
from app.core.llm_config import settings
from app.core.llm_model_policy import resolve_llm_model
from app.core.prompts import (
    AnalysisAction,
    SYSTEM_NATURAL_MESSAGE,
    should_use_natural_response,
    system_message_for_structured_analysis,
)
from app.dependencies.service_dependencies import (
    get_chat_service,
    get_company_research_service,
    get_llm_service,
    get_portfolio_analysis_service,
    get_portfolio_service,
    get_prompt_enrichment_service,
    get_schwab_auth_service,
)
from app.models.strategy_models import InvestmentStrategy, StrategyNextAction
from app.services.chat_service import ChatService
from app.services.company_research_service import CompanyResearchService
from app.services.llm_service import LLMService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired
from app.services.prompt_enrichment_service import (
    PromptEnrichmentService,
)
from app.services.strategy.strategy_playbook_prompts import (
    build_playbook_ask_prompt,
    playbook_action_askable,
    playbook_ask_display_message,
    playbook_ask_prefers_research_chat,
    playbook_research_system_message,
)

router = APIRouter()


class PlaybookAskRequest(BaseModel):
    symbol: str
    action_type: Literal[
        "research", "options", "buy", "rebalance", "monitor"
    ] = Field(alias="actionType")
    action_title: str = Field(alias="actionTitle")
    action_reason: str = Field(default="", alias="actionReason")
    strategy: InvestmentStrategy
    model: Optional[ResponsesModel] = "gpt-4.1-mini"
    chat_session_id: Optional[str] = Field(default=None, alias="chatSessionId")
    new_chat_session: bool = Field(default=False, alias="newChatSession")

    model_config = {"populate_by_name": True}


@router.post("/strategy/playbook/ask")
async def strategy_playbook_ask(
    request: PlaybookAskRequest,
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
    action = StrategyNextAction(
        type=request.action_type,
        title=request.action_title.strip(),
        reason=request.action_reason.strip(),
        symbol=symbol,
    )

    if not symbol or not action.title:
        return StreamingResponse(
            iter(["Please provide a symbol and playbook action."]),
            media_type="text/plain; charset=utf-8",
        )

    if not playbook_action_askable(action):
        return StreamingResponse(
            iter(["This playbook action cannot be sent to AI."]),
            media_type="text/plain; charset=utf-8",
        )

    secret_prompt = build_playbook_ask_prompt(action, request.strategy)
    display_message = playbook_ask_display_message(
        action,
        strategy=request.strategy,
    )
    model = resolve_llm_model(request.model, user_id)

    if playbook_ask_prefers_research_chat(action):
        return await _stream_research_playbook_ask(
            user_id=user_id,
            symbol=symbol,
            secret_prompt=secret_prompt,
            display_message=display_message,
            strategy=request.strategy,
            model=model,
            chat_session_id=request.chat_session_id,
            new_chat_session=request.new_chat_session,
            company_research_service=company_research_service,
            prompt_enrichment_service=prompt_enrichment_service,
            chat_service=chat_service,
            llm_service=llm_service,
            portfolio_service=portfolio_service,
            schwab_auth_service=schwab_auth_service,
            portfolio_analysis_service=portfolio_analysis_service,
        )

    return await _stream_portfolio_playbook_ask(
        user_id=user_id,
        symbol=symbol,
        secret_prompt=secret_prompt,
        display_message=display_message,
        model=model,
        chat_session_id=request.chat_session_id,
        new_chat_session=request.new_chat_session,
        chat_service=chat_service,
        llm_service=llm_service,
        portfolio_service=portfolio_service,
        schwab_auth_service=schwab_auth_service,
        portfolio_analysis_service=portfolio_analysis_service,
        prompt_enrichment_service=prompt_enrichment_service,
    )


async def _stream_research_playbook_ask(
    *,
    user_id: str,
    symbol: str,
    secret_prompt: str,
    display_message: str,
    strategy: InvestmentStrategy,
    model: Optional[ResponsesModel],
    chat_session_id: Optional[str],
    new_chat_session: bool,
    company_research_service: CompanyResearchService,
    prompt_enrichment_service: PromptEnrichmentService,
    chat_service: ChatService,
    llm_service: LLMService,
    portfolio_service: PortfolioService,
    schwab_auth_service: SchwabAuthService,
    portfolio_analysis_service: PortfolioAnalysisService,
) -> StreamingResponse:
    session_id, is_first_chat = chat_service.get_research_chat_session_id(
        user_id=user_id,
        symbol=symbol,
        prompt=display_message,
        model=model,
        chat_session_id=chat_session_id,
        new_chat_session=new_chat_session,
    )
    recent_messages = chat_service.get_chat_messages_by_session(session_id=session_id)

    if session_id:
        chat_service.create_message(
            session_id=session_id,
            role="user",
            content=display_message,
        )

    assistant_content_parts: List[str] = []

    async def streamer():
        yield "Looking up company data for your playbook question…\n\n"

        ctx = await asyncio.to_thread(
            company_research_service.build_context,
            symbol=symbol,
            include_news=True,
            include_press_releases=True,
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
        except (SchwabReauthRequired, Exception):
            holdings_block = None
            intelligence_block = None
            option_chain_block = None

        user_message = prompt_enrichment_service.build_playbook_research_user_message(
            ctx=ctx,
            user_prompt=secret_prompt,
            holdings_block=holdings_block,
            intelligence_block=intelligence_block,
            option_chain_block=option_chain_block,
        )

        playbook_history = recent_messages[-8:]
        async for chunk in llm_service.analyze_option_position(
            model=model or settings.OPENAI_MODEL,
            system_prompt=playbook_research_system_message(strategy=strategy),
            user_prompt=[*playbook_history, user_message],
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
            **({"X-Chat-Session-Id": str(session_id)} if session_id else {}),
        },
    )


async def _stream_portfolio_playbook_ask(
    *,
    user_id: str,
    symbol: str,
    secret_prompt: str,
    display_message: str,
    model: Optional[ResponsesModel],
    chat_session_id: Optional[str],
    new_chat_session: bool,
    chat_service: ChatService,
    llm_service: LLMService,
    portfolio_service: PortfolioService,
    schwab_auth_service: SchwabAuthService,
    portfolio_analysis_service: PortfolioAnalysisService,
    prompt_enrichment_service: PromptEnrichmentService,
) -> StreamingResponse:
    resolved_session_id, is_first_chat = chat_service.get_portfolio_analysis_session_id(
        user_id=user_id,
        symbol=symbol,
        prompt=display_message,
        model=model,
        chat_session_id=chat_session_id,
        new_chat_session=new_chat_session,
    )
    session_id = str(resolved_session_id) if resolved_session_id else None
    recent_messages = chat_service.get_chat_messages_by_session(session_id=session_id)

    if session_id:
        chat_service.create_message(
            session_id=session_id,
            role="user",
            content=display_message,
        )

    assistant_content_parts: List[str] = []

    async def streamer():
        yield "Pulling together your holdings and market context…\n\n"

        try:
            schwab_token = schwab_auth_service.get_valid_token_by_user_id(
                user_id=user_id
            )
            account_map = portfolio_service.get_enriched_account(
                access_token=schwab_token.access_token
            )
            account = account_map["account"]
            raw_positions = account.securitiesAccount.positions
            positions = PortfolioService._annotate_option_strategies(
                portfolio_analysis_service._positions_for_symbol(raw_positions, symbol)
            )
        except (SchwabReauthRequired, Exception):
            yield (
                "Connect Schwab to review your live positions for this playbook action."
            )
            return

        ctx = await portfolio_analysis_service.build_analysis_context(
            user_id=user_id,
            account=account,
            positions=positions,
            session_id=None,
            symbol=symbol,
            user_prompt=secret_prompt,
            action=AnalysisAction.FREE_FORM,
            include_market_data=is_first_chat,
        )

        include_context = chat_service.should_include_portfolio_context(
            is_first_chat=is_first_chat,
            action=AnalysisAction.FREE_FORM,
            recent_messages=recent_messages,
            user_prompt=secret_prompt,
        )

        user_prompt = prompt_enrichment_service.build_portfolio_strategy_prompt(
            ctx=ctx,
            include_context=include_context,
            json_response=False,
        )

        system_prompt = (
            SYSTEM_NATURAL_MESSAGE
            if should_use_natural_response(secret_prompt, action=AnalysisAction.FREE_FORM)
            else system_message_for_structured_analysis(symbol=symbol)
        )

        async for chunk in llm_service.analyze_option_position(
            model=model or settings.OPENAI_MODEL,
            system_prompt=system_prompt,
            user_prompt=[*recent_messages, user_prompt],
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

    response_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if session_id:
        response_headers["X-Chat-Session-Id"] = str(session_id)

    return StreamingResponse(
        streamer(),
        media_type="text/plain; charset=utf-8",
        headers=response_headers,
    )
