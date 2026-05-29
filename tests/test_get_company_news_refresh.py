from unittest.mock import AsyncMock, MagicMock

import asyncio
from datetime import datetime, timezone

from app.api.get_company_news_route import get_company_news
from app.core.llm_config import settings
from app.models.finnhub_news_models import NewsItem, NewsResponse
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


def test_get_company_news_returns_cached_view_without_refresh(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"paid-user"}))
    cached = _sample_news_view(summary="Cached")

    enriched_news_service = MagicMock()
    enriched_news_service.get_cached_view.return_value = cached
    news_service = MagicMock()

    result = asyncio.run(
        get_company_news(
            symbol="AAPL",
            refresh=False,
            user_id="paid-user",
            news_service=news_service,
            prompt_enrichment_service=MagicMock(),
            llm_service=MagicMock(),
            enriched_news_service=enriched_news_service,
        )
    )

    assert result is cached
    enriched_news_service.invalidate.assert_not_called()
    news_service.get_company_news.assert_not_called()


def test_get_company_news_free_user_skips_enriched_cache(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    news_item = NewsItem(
        category="company news",
        datetime=datetime(2026, 5, 20, tzinfo=timezone.utc),
        headline="Test headline",
        id=1,
        related="AAPL",
        source="Reuters",
        summary="Raw summary",
        url="https://example.com/story",
    )
    news = NewsResponse(root=[news_item])

    enriched_news_service = MagicMock()
    enriched_news_service.get_cached_view.return_value = _sample_news_view(
        summary="Pro cache"
    )
    news_service = MagicMock()
    news_service.get_company_news.return_value = news
    llm_service = MagicMock()

    result = asyncio.run(
        get_company_news(
            symbol="AAPL",
            refresh=False,
            user_id="free-user",
            news_service=news_service,
            prompt_enrichment_service=MagicMock(),
            llm_service=llm_service,
            enriched_news_service=enriched_news_service,
        )
    )

    enriched_news_service.get_cached_view.assert_not_called()
    llm_service.analyze_news.assert_not_called()
    enriched_news_service.store_view.assert_not_called()
    assert result.aiEnrichment is False
    assert len(result.items) == 1
    assert result.items[0].summary == "Raw summary"


def test_get_company_news_refresh_bypasses_cache(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"paid-user"}))
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
            user_id="paid-user",
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
    llm_service.analyze_news.assert_awaited_once_with(
        symbol="AAPL",
        prompts=["system", "user"],
        news=news,
        user_id="paid-user",
    )
    enriched_news_service.store_view.assert_called_once_with(symbol="AAPL", view=fresh)
    assert result.summary == "Fresh"
