from typing import AsyncGenerator
from app.adapters.llm.openai_adapter import OpenAIAdapter
from typing import Optional, TypeVar
from openai.types.shared import ResponsesModel
from app.builders.news_analytics_builder import NewsAnalyticsBuilder
from app.models.finnhub_news_models import NewsResponse
from typing import List, Dict, Any, Type
from app.models.news_analytics_models import StockNewsView
from app.builders.prompt_builder import PromptBuilder
from app.core.llm_config import settings
from app.core.llm_json import validate_llm_model
from pydantic import BaseModel, ValidationError
import json

T = TypeVar("T", bound=BaseModel)


class LLMService:
    def __init__(
        self,
        openai_adapter: OpenAIAdapter,
        news_analytics_builder: NewsAnalyticsBuilder,
        prompt_builder: PromptBuilder,
    ):
        self.openai_adapter = openai_adapter
        self.news_analytics_builder = news_analytics_builder
        self.prompt_builder = prompt_builder

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
        ):
            yield chunk

    async def analyze_news(
        self,
        symbol: str,
        prompts: List[str],
        news: NewsResponse,
        model: Optional[ResponsesModel] = settings.OPENAI_MODEL,
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
            )
        enriched_news = await self.news_analytics_builder.get_enriched_news_items(
            model=model, prompts=prompts, news=news.root
        )
        return await self.prompt_builder.get_enriched_news_sentiment(
            model=model,
            symbol=symbol,
            enriched_news=enriched_news,
        )

    async def generate_from_prompts(
        self,
        prompts: List[str],
        response_model: Type[T],
    ) -> T:
        ai_response = await self.openai_adapter.generate(
            model=settings.OPENAI_MODEL,
            prompts=prompts,
            response_model=response_model,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        )

        try:
            if isinstance(ai_response, str):
                return validate_llm_model(ai_response, response_model)
            return response_model.model_validate(ai_response)
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
                model=settings.OPENAI_MODEL,
                prompts=retry_prompts,
                response_model=response_model,
                max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            )
            if isinstance(ai_response, str):
                return validate_llm_model(ai_response, response_model)
            return response_model.model_validate(ai_response)
