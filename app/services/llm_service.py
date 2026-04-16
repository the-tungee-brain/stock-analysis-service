from typing import AsyncGenerator
from app.adapters.llm.openai_adapter import OpenAIAdapter
from typing import Optional
from openai.types.shared import ResponsesModel
from app.builders.news_analytics_builder import NewsAnalyticsBuilder
from app.models.finnhub_news_models import NewsResponse
from typing import List, Dict, Any
from app.models.news_analytics_models import StockNewsView
from app.builders.prompt_builder import PromptBuilder
from app.core.llm_config import settings
from app.models.company_research_models import AISummary
import json


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

    async def generate_stock_summary(self, prompts: List[str]) -> AISummary:
        ai_response = await self.openai_adapter.generate(prompts=prompts)

        if isinstance(ai_response, str):
            try:
                data = json.loads(ai_response)
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM returned invalid JSON: {e}") from e
        else:
            data = ai_response

        return AISummary.model_validate(data)
