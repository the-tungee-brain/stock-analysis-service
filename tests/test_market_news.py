from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from app.models.finnhub_news_models import NewsItem, NewsResponse
from app.models.intelligence_models import MarketNewsItem, PortfolioDigest, PortfolioIntelligence
from app.services.intelligence.portfolio_intelligence_service import PortfolioIntelligenceService
from app.services.news_service import (
    MARKET_NEWS_DISPLAY_LIMIT,
    MARKET_NEWS_PROMPT_LIMIT,
    NewsService,
)
from app.services.prompt_enrichment_service import PromptEnrichmentService
from tests.test_position_prompt_metrics import _make_account


def _news_item(
    headline: str,
    hours_ago: int,
    item_id: int,
    *,
    image: str | None = None,
) -> NewsItem:
    return NewsItem(
        category="general",
        datetime=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        headline=headline,
        id=item_id,
        related="",
        source="Reuters",
        summary="Summary",
        url="https://example.com/news",
        image=image,
    )


def test_get_market_news_filters_to_recent_and_limits_display_count():
    builder = MagicMock()
    builder.get_market_news.return_value = NewsResponse(
        root=[
            _news_item("Fresh headline", 1, 1),
            _news_item("Stale headline", 48, 2),
            _news_item("Another fresh", 3, 3),
            _news_item("Also fresh", 5, 4),
            _news_item("Still fresh", 6, 5),
            _news_item("Too many", 7, 6),
        ]
    )

    service = NewsService(finnhub_builder=builder)
    response = service.get_market_news(limit=3)

    assert [item.headline for item in response.root] == [
        "Fresh headline",
        "Another fresh",
        "Also fresh",
    ]


def test_portfolio_digest_includes_macro_news_from_single_finnhub_call():
    news_service = MagicMock()
    news_service.get_market_news.return_value = NewsResponse(
        root=[
            _news_item(
                "Fed holds rates steady",
                2,
                10,
                image="https://example.com/fed.jpg",
            )
        ]
    )

    service = PortfolioIntelligenceService(
        peer_comparison_service=MagicMock(),
        enriched_news_service=MagicMock(),
        news_service=news_service,
    )

    intelligence = service.build_portfolio_intelligence(
        positions=[],
        account=_make_account(),
    )

    assert intelligence.digest is not None
    assert intelligence.digest.macro_news == [
        MarketNewsItem(
            headline="Fed holds rates steady",
            source="Reuters",
            url="https://example.com/news",
            image="https://example.com/fed.jpg",
        )
    ]
    news_service.get_market_news.assert_called_once()


def test_prompt_uses_fewer_headlines_than_email_display():
    digest = PortfolioDigest(
        macro_regime="VIX at 18.0",
        macro_news=[
            MarketNewsItem(headline=f"Headline {idx}", source="Reuters")
            for idx in range(MARKET_NEWS_DISPLAY_LIMIT)
        ],
    )
    intelligence = PortfolioIntelligence(signals=[], digest=digest, alerts=[])
    block = PromptEnrichmentService.format_portfolio_intelligence_block(intelligence)

    assert block is not None
    assert block.count("Headline") == MARKET_NEWS_PROMPT_LIMIT
    assert "Headline 4" not in block
