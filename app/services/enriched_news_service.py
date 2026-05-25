from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.adapters.cache.enriched_news_cache import EnrichedNewsCache
from app.models.company_research_models import EnrichedNewsSummary
from app.models.finnhub_news_models import NewsResponse
from app.models.news_analytics_models import StockNewsView

if TYPE_CHECKING:
    from app.services.llm_service import LLMService
    from app.services.news_service import NewsService
    from app.services.prompt_enrichment_service import PromptEnrichmentService


class EnrichedNewsService:
    def __init__(
        self,
        enriched_news_cache: EnrichedNewsCache | None = None,
        news_service: NewsService | None = None,
        prompt_enrichment_service: PromptEnrichmentService | None = None,
        llm_service: LLMService | None = None,
    ):
        self.enriched_news_cache = enriched_news_cache
        self.news_service = news_service
        self.prompt_enrichment_service = prompt_enrichment_service
        self.llm_service = llm_service

    def get_cached_view(self, symbol: str) -> StockNewsView | None:
        if self.enriched_news_cache is None:
            return None
        try:
            return self.enriched_news_cache.get(symbol=symbol)
        except Exception:
            return None

    def get_cached_summary(self, symbol: str) -> EnrichedNewsSummary | None:
        if self.enriched_news_cache is None:
            return None
        try:
            view = self.enriched_news_cache.get(symbol=symbol)
        except Exception:
            return None
        if view is None:
            return None
        return self._to_summary(view)

    def store_view(self, symbol: str, view: StockNewsView) -> None:
        if self.enriched_news_cache is None:
            return
        try:
            self.enriched_news_cache.put(symbol=symbol, view=view)
        except Exception:
            pass

    def invalidate(self, symbol: str) -> None:
        if self.enriched_news_cache is None:
            return
        try:
            self.enriched_news_cache.delete(symbol=symbol)
        except Exception:
            pass

    async def ensure_enriched(
        self,
        symbol: str,
        *,
        news: NewsResponse | None = None,
    ) -> EnrichedNewsSummary | None:
        cached = self.get_cached_summary(symbol=symbol)
        if cached is not None:
            return cached

        if (
            self.news_service is None
            or self.prompt_enrichment_service is None
            or self.llm_service is None
        ):
            return None

        try:
            if news is None:
                news = await asyncio.to_thread(
                    self.news_service.get_company_news,
                    symbol=symbol,
                    lookback_days=7,
                )
            prompts = self.prompt_enrichment_service.enrich_news_prompt(
                symbol=symbol,
                news=news,
            )
            view = await self.llm_service.analyze_news(
                symbol=symbol,
                prompts=prompts,
                news=news,
            )
            self.store_view(symbol=symbol, view=view)
            return self._to_summary(view)
        except Exception:
            return None

    @staticmethod
    def _to_summary(view: StockNewsView) -> EnrichedNewsSummary:
        return EnrichedNewsSummary(
            overall_sentiment=view.overall_sentiment,
            summary=view.summary,
            insights=list(view.insights),
            risks=list(view.risks),
            dominant_driver=view.dominant_driver,
            actionability_score=view.actionability_score,
            investor_takeaway=view.investorTakeaway,
        )
