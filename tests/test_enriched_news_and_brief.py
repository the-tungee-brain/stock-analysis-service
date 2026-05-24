from unittest.mock import AsyncMock, MagicMock
import asyncio

from app.core.prompts import AnalysisAction
from app.models.company_research_models import EnrichedNewsSummary
from app.models.intelligence_models import PortfolioIntelligence
from app.models.news_analytics_models import StockNewsView
from app.services.enriched_news_service import EnrichedNewsService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from tests.test_position_prompt_metrics import _make_account, _make_position


def test_should_auto_enrich_news_for_daily_summary():
    assert PortfolioAnalysisService._should_auto_enrich_news(
        AnalysisAction.DAILY_SUMMARY
    )
    assert not PortfolioAnalysisService._should_auto_enrich_news(
        AnalysisAction.TAX_ANGLE
    )


def test_ensure_enriched_returns_cache_without_llm():
    cache = MagicMock()
    cache.get.return_value = StockNewsView(
        symbol="AAPL",
        overall_sentiment="neutral",
        summary="Cached summary",
        insights=["Insight"],
        risks=["Risk"],
        dominant_driver="Product cycle",
        market_impact_horizon="medium_term",
        actionability_score=3,
        investorTakeaway="Hold steady.",
        deepAnalysis="Cached.",
        items=[],
    )

    llm_service = MagicMock()
    service = EnrichedNewsService(
        enriched_news_cache=cache,
        news_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        llm_service=llm_service,
    )

    summary = asyncio.run(service.ensure_enriched("AAPL"))

    assert summary is not None
    assert summary.summary == "Cached summary"
    llm_service.analyze_news.assert_not_called()


def test_ensure_enriched_fetches_on_cache_miss():
    cache = MagicMock()
    cache.get.return_value = None

    news_service = MagicMock()
    news = MagicMock()
    news.root = [MagicMock()]
    news_service.get_company_news.return_value = news

    prompt_enrichment_service = MagicMock()
    prompt_enrichment_service.enrich_news_prompt.return_value = ["system", "user"]

    llm_service = MagicMock()
    llm_service.analyze_news = AsyncMock(
        return_value=StockNewsView(
            symbol="AAPL",
            overall_sentiment="bullish",
            summary="Fresh summary",
            insights=[],
            risks=[],
            dominant_driver="Earnings beat",
            market_impact_horizon="immediate",
            actionability_score=4,
            investorTakeaway="Watch guidance.",
            deepAnalysis="Fresh.",
            items=[],
        )
    )

    service = EnrichedNewsService(
        enriched_news_cache=cache,
        news_service=news_service,
        prompt_enrichment_service=prompt_enrichment_service,
        llm_service=llm_service,
    )

    summary = asyncio.run(service.ensure_enriched("AAPL"))

    assert summary is not None
    assert summary.summary == "Fresh summary"
    cache.put.assert_called_once()
    llm_service.analyze_news.assert_awaited_once()


def test_build_portfolio_brief_returns_intelligence():
    portfolio_intelligence_service = MagicMock()
    portfolio_intelligence_service.attach_enriched_news.side_effect = lambda ctx: ctx
    portfolio_intelligence_service.build_portfolio_intelligence.return_value = (
        PortfolioIntelligence(
            signals=[],
            digest=None,
            alerts=[],
        )
    )

    company_research_service = MagicMock()
    company_research_service.build_context.return_value = MagicMock(
        snapshot=MagicMock(sector="Technology")
    )

    market_service = MagicMock()
    market_service.get_enriched_quote_snapshot.return_value = {}

    transaction_service = MagicMock()
    transaction_service.build_recent_activity_summary.return_value = MagicMock(
        suggested_actions=[]
    )

    service = PortfolioAnalysisService(
        market_service=market_service,
        schwab_auth_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        company_research_service=company_research_service,
        transaction_service=transaction_service,
        portfolio_intelligence_service=portfolio_intelligence_service,
    )

    account = _make_account()
    positions = [_make_position(symbol="AAPL", market_value=50_000)]

    brief = service.build_portfolio_brief(
        user_id="user-1",
        account=account,
        positions=positions,
        access_token="token",
    )

    assert isinstance(brief, PortfolioIntelligence)
    portfolio_intelligence_service.build_portfolio_intelligence.assert_called_once()
