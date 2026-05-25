from unittest.mock import AsyncMock, MagicMock

from app.api.get_company_news_route import get_company_news
import asyncio
from app.models.finnhub_news_models import NewsResponse
from app.models.news_analytics_models import StockNewsView


def _sample_news_view(*, summary: str, overall_sentiment: str = "neutral") -> StockNewsView:
    return StockNewsView(
        symbol="AAPL",
        overall_sentiment=overall_sentiment,
        summary=summary,
        insights=[],
        risks=[],
        dominant_driver="Product cycle",
        market_impact_horizon="medium_term",
        actionability_score=3,
        investorTakeaway="Hold steady.",
        deepAnalysis="Analysis.",
        items=[],
    )


def test_get_company_news_returns_cached_view_without_refresh():
    cached = _sample_news_view(summary="Cached")

    enriched_news_service = MagicMock()
    enriched_news_service.get_cached_view.return_value = cached
    news_service = MagicMock()

    result = asyncio.run(
        get_company_news(
            symbol="AAPL",
            refresh=False,
            news_service=news_service,
            prompt_enrichment_service=MagicMock(),
            llm_service=MagicMock(),
            enriched_news_service=enriched_news_service,
        )
    )

    assert result is cached
    enriched_news_service.invalidate.assert_not_called()
    news_service.get_company_news.assert_not_called()


def test_get_company_news_refresh_bypasses_cache():
    news = NewsResponse(root=[])
    fresh = _sample_news_view(summary="Fresh", overall_sentiment="bullish")
    fresh = fresh.model_copy(update={"insights": ["Insight"]})

    enriched_news_service = MagicMock()
    enriched_news_service.get_cached_view.return_value = _sample_news_view(
        summary="Stale"
    )

    news_service = MagicMock()
    news_service.get_company_news.return_value = news

    prompt_enrichment_service = MagicMock()
    prompt_enrichment_service.enrich_news_prompt.return_value = ["system", "user"]

    llm_service = MagicMock()
    llm_service.analyze_news = AsyncMock(return_value=fresh)

    result = asyncio.run(
        get_company_news(
            symbol="AAPL",
            refresh=True,
            news_service=news_service,
            prompt_enrichment_service=prompt_enrichment_service,
            llm_service=llm_service,
            enriched_news_service=enriched_news_service,
        )
    )

    enriched_news_service.invalidate.assert_called_once_with(symbol="AAPL")
    news_service.invalidate_company_news_cache.assert_called_once_with(
        symbol="AAPL",
        lookback_days=7,
    )
    enriched_news_service.get_cached_view.assert_not_called()
    news_service.get_company_news.assert_called_once()
    llm_service.analyze_news.assert_awaited_once()
    enriched_news_service.store_view.assert_called_once_with(symbol="AAPL", view=fresh)
    assert result.summary == "Fresh"
