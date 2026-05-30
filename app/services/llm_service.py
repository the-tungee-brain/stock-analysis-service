import asyncio
import json
from typing import AsyncGenerator, List, Dict, Any, Type, TypeVar, Optional

from pydantic import BaseModel, ValidationError
from openai.types.shared import ResponsesModel

from app.adapters.cache.llm_output_cache import LLMOutputCache
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.builders.news_analytics_builder import NewsAnalyticsBuilder
from app.builders.prompt_builder import PromptBuilder
from app.core.llm_config import settings
from app.core.llm_model_policy import resolve_background_llm_model, resolve_llm_model
from app.core.llm_json import validate_llm_model
from app.core.llm_routes import LLMRoute
from app.models.finnhub_news_models import NewsItem, NewsResponse
from app.models.news_analytics_models import (
    CombinedNewsLLMOutput,
    EnrichedNewsItem,
    StockNewsView,
)
from app.services.news_service import COMPANY_NEWS_LLM_LIMIT

T = TypeVar("T", bound=BaseModel)


class LLMService:
    def __init__(
        self,
        openai_adapter: OpenAIAdapter,
        news_analytics_builder: NewsAnalyticsBuilder,
        prompt_builder: PromptBuilder,
        llm_output_cache: LLMOutputCache | None = None,
    ):
        self.openai_adapter = openai_adapter
        self.news_analytics_builder = news_analytics_builder
        self.prompt_builder = prompt_builder
        self.llm_output_cache = llm_output_cache

    async def analyze_option_position(
        self,
        model: Optional[ResponsesModel],
        system_prompt: Optional[str],
        user_prompt: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.openai_adapter.generate_stream(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS_STREAM,
        ):
            yield chunk

    @staticmethod
    def build_headlines_only_view(
        symbol: str,
        news: NewsResponse,
        *,
        pro: bool = False,
    ) -> StockNewsView:
        """Headlines without LLM synthesis."""
        if not news.root:
            empty_summary = (
                "No recent market headlines for this symbol."
                if pro
                else "No recent market headlines for this symbol."
            )
            return StockNewsView(
                symbol=symbol,
                overall_sentiment="neutral",
                summary=empty_summary,
                insights=[],
                risks=[],
                dominant_driver="No recent headlines.",
                market_impact_horizon="medium_term",
                actionability_score=1,
                investorTakeaway="",
                deepAnalysis="",
                items=[],
                aiEnrichment=False,
            )

        summary = (
            "Headlines are loaded below. Run Analyze news for sentiment, summaries, "
            "and an AI brief."
            if pro
            else (
                "Headlines are available below. Upgrade to Pro for an AI news brief, "
                "per-story sentiment, and actionable context."
            )
        )

        return StockNewsView(
            symbol=symbol,
            overall_sentiment="neutral",
            summary=summary,
            insights=[],
            risks=[],
            dominant_driver="",
            market_impact_horizon="medium_term",
            actionability_score=1,
            investorTakeaway="",
            deepAnalysis="",
            items=[LLMService._passthrough_enriched_item(src) for src in news.root],
            aiEnrichment=False,
        )

    @staticmethod
    def _passthrough_enriched_item(src: NewsItem) -> EnrichedNewsItem:
        summary = (src.summary or "").strip() or src.headline
        return EnrichedNewsItem(
            id=src.id,
            datetime=src.datetime.isoformat(),
            headline=src.headline,
            source=src.source,
            original_summary=src.summary or "",
            sentiment="neutral",
            confidence=0.35,
            summary=summary,
            topics=[],
            url=src.url,
            image=src.image,
        )

    async def analyze_news(
        self,
        symbol: str,
        prompts: List[str],
        news: NewsResponse,
        user_id: str | None = None,
        *,
        model: Optional[ResponsesModel] = None,
    ) -> StockNewsView:
        if not news.root:
            return StockNewsView(
                symbol=symbol,
                overall_sentiment="neutral",
                summary="No recent news found for this symbol.",
                insights=[],
                risks=[],
                dominant_driver="No recent news.",
                market_impact_horizon="medium_term",
                actionability_score=1,
                investorTakeaway="No recent news to analyze. Check back later or review the company's business profile.",
                deepAnalysis="No recent news articles were found for this symbol in the past day. Without news flow, focus on fundamentals, business model, and price performance for your research.",
                items=[],
                aiEnrichment=True,
            )

        fingerprint = ",".join(str(item.id) for item in news.root[:COMPANY_NEWS_LLM_LIMIT])
        combined = await self.generate_from_prompts(
            prompts,
            CombinedNewsLLMOutput,
            route=LLMRoute.NEWS,
            symbol=symbol,
            context_fingerprint=f"combined:{fingerprint}",
            user_id=user_id,
            model=model,
        )

        llm_by_id = {item.id: item for item in combined.items}
        enriched_items: list[EnrichedNewsItem] = []
        for src in news.root:
            obj = llm_by_id.get(src.id)
            if obj is None:
                enriched_items.append(self._passthrough_enriched_item(src))
                continue
            enriched_items.append(
                EnrichedNewsItem(
                    id=src.id,
                    datetime=src.datetime.isoformat(),
                    headline=src.headline,
                    source=src.source,
                    original_summary=src.summary or "",
                    sentiment=obj.sentiment,
                    confidence=float(obj.confidence),
                    summary=obj.summary,
                    topics=list(obj.topics),
                    url=src.url,
                    image=src.image,
                )
            )

        horizon = combined.market_impact_horizon
        if horizon not in {"immediate", "medium_term", "long_term"}:
            horizon = "medium_term"

        return StockNewsView(
            symbol=symbol,
            overall_sentiment=combined.overall_sentiment,
            summary=combined.summary,
            insights=list(combined.insights),
            risks=list(combined.risks),
            dominant_driver=combined.dominant_driver
            or "No dominant news driver identified.",
            market_impact_horizon=horizon,
            actionability_score=max(1, min(5, int(combined.actionability_score))),
            investorTakeaway=combined.investorTakeaway,
            deepAnalysis=combined.deepAnalysis,
            items=enriched_items,
            aiEnrichment=True,
        )

    async def generate_stream_from_prompts(
        self,
        prompts: List[str],
        *,
        route: LLMRoute,
        user_id: str | None = None,
        model: Optional[ResponsesModel] = None,
    ) -> AsyncGenerator[str, None]:
        resolved_model = model or resolve_background_llm_model(user_id, route)
        max_tokens = settings.max_tokens_for_route(route)
        async for chunk in self.openai_adapter.generate_stream_from_prompts(
            model=resolved_model,
            prompts=prompts,
            max_output_tokens=max_tokens,
        ):
            yield chunk

    async def generate_from_prompts(
        self,
        prompts: List[str],
        response_model: Type[T],
        *,
        route: LLMRoute,
        symbol: str | None = None,
        context_fingerprint: str | None = None,
        user_id: str | None = None,
        model: Optional[ResponsesModel] = None,
        max_output_tokens: int | None = None,
    ) -> T:
        resolved_model = model or resolve_background_llm_model(user_id, route)
        resolved_max_tokens = max_output_tokens or settings.max_tokens_for_route(route)

        if (
            self.llm_output_cache is not None
            and symbol
            and context_fingerprint
        ):
            try:
                cached = self.llm_output_cache.get(
                    route=route,
                    symbol=symbol,
                    fingerprint=context_fingerprint,
                )
                if cached is not None:
                    return response_model.model_validate_json(cached)
            except Exception:
                pass

        ai_response = await self.openai_adapter.generate(
            model=resolved_model,
            prompts=prompts,
            response_model=response_model,
            max_output_tokens=resolved_max_tokens,
        )

        try:
            parsed = self._parse_response(ai_response, response_model)
        except (ValidationError, ValueError, json.JSONDecodeError):
            retry_prompts = [
                prompts[0],
                (
                    f"{prompts[1]}\n\n"
                    "Your previous answer was not valid JSON. "
                    f"Return ONLY a valid JSON object matching the required schema for {response_model.__name__}."
                ),
            ]
            ai_response = await self.openai_adapter.generate(
                model=resolved_model,
                prompts=retry_prompts,
                response_model=response_model,
                max_output_tokens=resolved_max_tokens,
            )
            parsed = self._parse_response(ai_response, response_model)

        if (
            self.llm_output_cache is not None
            and symbol
            and context_fingerprint
        ):
            try:
                self.llm_output_cache.put(
                    route=route,
                    symbol=symbol,
                    fingerprint=context_fingerprint,
                    payload=parsed.model_dump_json(),
                )
            except Exception:
                pass

        return parsed

    @staticmethod
    def _parse_response(raw: str | BaseModel, response_model: Type[T]) -> T:
        if isinstance(raw, str):
            return validate_llm_model(raw, response_model)
        return response_model.model_validate(raw)
