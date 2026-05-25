from unittest.mock import MagicMock

from app.models.company_research_models import (
    EnrichedNewsSummary,
    NewsHeadline,
    ResearchContext,
)
from app.models.news_analytics_models import EnrichedNewsItem, StockNewsView
from app.services.intelligence.portfolio_intelligence_service import (
    PortfolioIntelligenceService,
)
from tests.test_position_prompt_metrics import _make_account, _make_position


def _service(enriched_news_service: MagicMock) -> PortfolioIntelligenceService:
    return PortfolioIntelligenceService(
        peer_comparison_service=MagicMock(),
        enriched_news_service=enriched_news_service,
    )


def test_portfolio_news_digest_uses_enriched_article_url():
    enriched_news_service = MagicMock()
    enriched_news_service.get_cached_summary.return_value = EnrichedNewsSummary(
        overall_sentiment="bullish",
        summary="Strong quarter.",
        dominant_driver="Earnings beat expectations",
    )
    enriched_news_service.get_cached_view.return_value = StockNewsView(
        symbol="AAPL",
        overall_sentiment="bullish",
        summary="Strong quarter.",
        insights=[],
        risks=[],
        dominant_driver="Earnings beat expectations",
        market_impact_horizon="immediate",
        actionability_score=4,
        investorTakeaway="Hold.",
        deepAnalysis="",
        items=[
            EnrichedNewsItem(
                id=1,
                datetime="2026-05-24T12:00:00+00:00",
                headline="Apple beats earnings",
                source="Reuters",
                original_summary="Beat on EPS.",
                sentiment="bullish",
                confidence=0.9,
                summary="Beat on EPS.",
                topics=["earnings"],
                url="https://example.com/aapl-earnings",
            )
        ],
    )

    ctx = ResearchContext(symbol="AAPL")
    service = _service(enriched_news_service)
    items = service._portfolio_news_digest(
        research_contexts=[ctx],
        positions=[_make_position(symbol="AAPL", market_value=50_000)],
        account=_make_account(liquidation_value=100_000),
    )

    assert len(items) == 1
    assert items[0].symbol == "AAPL"
    assert items[0].url == "https://example.com/aapl-earnings"


def test_portfolio_news_digest_falls_back_to_raw_news_url():
    enriched_news_service = MagicMock()
    enriched_news_service.get_cached_summary.return_value = None
    enriched_news_service.get_cached_view.return_value = None

    ctx = ResearchContext(
        symbol="NVDA",
        news=[
            NewsHeadline(
                headline="NVIDIA raises guidance",
                url="https://example.com/nvda-news",
            )
        ],
    )
    service = _service(enriched_news_service)
    items = service._portfolio_news_digest(
        research_contexts=[ctx],
        positions=[_make_position(symbol="NVDA", market_value=40_000)],
        account=_make_account(liquidation_value=100_000),
    )

    assert len(items) == 1
    assert items[0].url == "https://example.com/nvda-news"
