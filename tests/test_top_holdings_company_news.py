from app.models.company_research_models import NewsHeadline, ResearchContext
from app.models.intelligence_models import (
    HoldingCompanyNewsItem,
    PortfolioDigest,
    PortfolioIntelligence,
    PortfolioNewsItem,
)
from app.services.intelligence.portfolio_intelligence_service import (
    TOP_HOLDINGS_COMPANY_NEWS_HEADLINES_PER_SYMBOL,
    TOP_HOLDINGS_COMPANY_NEWS_SYMBOL_LIMIT,
    PortfolioIntelligenceService,
)
from app.services.prompt_enrichment_service import PromptEnrichmentService
from tests.test_position_prompt_metrics import _make_account, _make_position


def _service() -> PortfolioIntelligenceService:
    from unittest.mock import MagicMock

    return PortfolioIntelligenceService(
        peer_comparison_service=MagicMock(),
        enriched_news_service=MagicMock(),
        news_service=MagicMock(),
    )


def test_top_holdings_company_news_uses_prefetched_headlines_only():
    service = _service()
    account = _make_account(liquidation_value=100_000)
    positions = [
        _make_position(symbol="AAPL", market_value=40_000),
        _make_position(symbol="MSFT", market_value=20_000),
    ]
    contexts = [
        ResearchContext(
            symbol="AAPL",
            news=[
                NewsHeadline(
                    headline="Apple unveils AI features",
                    summary="Product launch event.",
                    source="Reuters",
                    url="https://example.com/aapl-1",
                ),
                NewsHeadline(
                    headline="Apple supplier update",
                    summary="Supply chain note.",
                    source="Bloomberg",
                ),
                NewsHeadline(headline="Third headline should be skipped"),
            ],
        ),
        ResearchContext(
            symbol="MSFT",
            news=[
                NewsHeadline(
                    headline="Microsoft cloud growth",
                    summary="Azure beat estimates.",
                    source="WSJ",
                )
            ],
        ),
        ResearchContext(symbol="NVDA", news=[]),
    ]

    items = service._top_holdings_company_news(
        research_contexts=contexts,
        positions=positions,
        account=account,
    )

    assert len(items) == 3
    assert items[0].symbol == "AAPL"
    assert items[0].headline == "Apple unveils AI features"
    assert items[0].url == "https://example.com/aapl-1"
    assert items[0].weight_pct == 40.0
    assert items[1].symbol == "AAPL"
    assert items[2].symbol == "MSFT"
    assert all(
        item.symbol != "NVDA" or item.headline for item in items
    )
    assert sum(1 for item in items if item.symbol == "AAPL") == (
        TOP_HOLDINGS_COMPANY_NEWS_HEADLINES_PER_SYMBOL
    )


def test_format_portfolio_intelligence_omits_redundant_company_news_block():
    digest = PortfolioDigest(
        top_news=[
            PortfolioNewsItem(
                symbol="AAPL",
                headline="Apple unveils AI features",
                sentiment="bullish",
                weight_pct=40.0,
                url="https://example.com/aapl-1",
            )
        ],
        top_holdings_company_news=[
            HoldingCompanyNewsItem(
                symbol="AAPL",
                headline="Apple unveils AI features",
                source="Reuters",
                summary="Product launch event.",
                url="https://example.com/aapl-1",
                weight_pct=40.0,
            )
        ],
    )
    intelligence = PortfolioIntelligence(signals=[], digest=digest, alerts=[])
    block = PromptEnrichmentService.format_portfolio_intelligence_block(intelligence)

    assert block is not None
    assert "Top holdings news digest" in block
    assert "Apple unveils AI features" in block
    assert "Top holdings company news" not in block


def test_company_news_limits_are_small_for_prompt_size():
    assert TOP_HOLDINGS_COMPANY_NEWS_SYMBOL_LIMIT == 5
    assert TOP_HOLDINGS_COMPANY_NEWS_HEADLINES_PER_SYMBOL == 2
