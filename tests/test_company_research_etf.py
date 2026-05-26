from unittest.mock import MagicMock

from app.models.company_research_models import EtfHoldingsContext, ResearchContext
from app.services.company_research_service import CompanyResearchService


def test_build_context_for_etf_skips_sec_and_loads_holdings():
    ticker_builder = MagicMock()
    ticker_builder.get_by_symbol.return_value = MagicMock(asset_type="ETF")

    etf_holdings = EtfHoldingsContext(ticker="SPY", total_holdings=504)
    etf_service = MagicMock()
    etf_service.build_holdings_context.return_value = etf_holdings

    fundamentals_builder = MagicMock()
    fundamentals_builder.build_etf_metrics.return_value = {
        "dividend_yield": "1.25%",
        "expense_ratio": "0.09%",
    }

    company_profile = MagicMock()
    company_profile.get_snapshot.return_value = None
    company_profile.get_peers.return_value = []

    market_service = MagicMock()
    market_service.get_performance.return_value = None

    news_service = MagicMock()
    news_service.get_company_news.return_value = MagicMock(root=[])

    sec_service = MagicMock()
    earnings_service = MagicMock()

    service = CompanyResearchService(
        company_profile_service=company_profile,
        market_service=market_service,
        news_service=news_service,
        fundamentals_builder=fundamentals_builder,
        sec_research_service=sec_service,
        earnings_service=earnings_service,
        ticker_symbol_builder=ticker_builder,
        etf_research_service=etf_service,
    )

    ctx = service._build_context("SPY")

    assert ctx.asset_type == "ETF"
    assert ctx.etf_holdings == etf_holdings
    assert ctx.sec_fundamentals == []
    assert ctx.earnings is None
    assert ctx.peers == []
    assert any(metric.label == "Expense ratio" for metric in ctx.fundamentals)
    sec_service.lookup.assert_not_called()
    earnings_service.build_research_context.assert_not_called()
    company_profile.get_peers.assert_not_called()
    etf_service.build_holdings_context.assert_called_once_with(symbol="SPY")
