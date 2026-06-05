from unittest.mock import MagicMock

from app.models.company_research_models import EtfHoldingsContext
from app.services.company_research_service import CompanyResearchService


def _make_etf_research_service(*, news_service: MagicMock | None = None):
    ticker_builder = MagicMock()
    ticker_builder.get_by_symbol.return_value = MagicMock(asset_type="ETF")

    etf_service = MagicMock()
    etf_service.build_holdings_context.return_value = EtfHoldingsContext(
        ticker="SPY",
        total_holdings=504,
    )

    fundamentals_builder = MagicMock()
    fundamentals_builder.build_etf_metrics.return_value = {}

    company_profile = MagicMock()
    company_profile.get_snapshot.return_value = None
    company_profile.get_peers.return_value = []

    market_service = MagicMock()
    market_service.get_performance.return_value = None

    return CompanyResearchService(
        company_profile_service=company_profile,
        market_service=market_service,
        news_service=news_service or MagicMock(),
        fundamentals_builder=fundamentals_builder,
        sec_research_service=MagicMock(),
        earnings_service=MagicMock(),
        ticker_symbol_builder=ticker_builder,
        etf_research_service=etf_service,
    )


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


def test_build_context_default_does_not_call_company_news():
    news_service = MagicMock()
    news_service.get_company_news.return_value = MagicMock(root=[])
    service = _make_etf_research_service(news_service=news_service)

    ctx = service.build_context("SPY")

    assert ctx.news == []
    news_service.get_company_news.assert_not_called()


def test_build_context_include_news_calls_company_news():
    news_service = MagicMock()
    news_service.get_company_news.return_value = MagicMock(root=[])
    service = _make_etf_research_service(news_service=news_service)

    service.build_context("SPY", include_news=True)

    news_service.get_company_news.assert_called_once_with(
        symbol="SPY",
        lookback_days=7,
    )


def test_build_context_default_does_not_call_press_releases():
    news_service = MagicMock()
    news_service.get_press_releases.return_value = MagicMock(root=[])
    service = _make_etf_research_service(news_service=news_service)

    ctx = service.build_context("SPY")

    assert ctx.press_releases == []
    news_service.get_press_releases.assert_not_called()


def test_build_context_include_press_releases_calls_press_releases():
    news_service = MagicMock()
    news_service.get_press_releases.return_value = MagicMock(root=[])
    service = _make_etf_research_service(news_service=news_service)

    service.build_context("SPY", include_press_releases=True)

    news_service.get_press_releases.assert_called_once_with(
        symbol="SPY",
        lookback_days=30,
    )


def test_company_research_uses_asset_type_service_for_resolution():
    asset_type_service = MagicMock()
    asset_type_service.resolve.return_value = "ETF"
    ticker_builder = MagicMock()
    ticker_builder.get_by_symbol.return_value = MagicMock(asset_type="STOCK")

    etf_holdings = EtfHoldingsContext(ticker="SPY", total_holdings=504)
    etf_service = MagicMock()
    etf_service.build_holdings_context.return_value = etf_holdings

    fundamentals_builder = MagicMock()
    fundamentals_builder.build_etf_metrics.return_value = {}

    service = CompanyResearchService(
        company_profile_service=MagicMock(get_snapshot=MagicMock(return_value=None)),
        market_service=MagicMock(get_performance=MagicMock(return_value=None)),
        news_service=MagicMock(),
        fundamentals_builder=fundamentals_builder,
        sec_research_service=MagicMock(),
        earnings_service=MagicMock(),
        ticker_symbol_builder=ticker_builder,
        etf_research_service=etf_service,
        asset_type_service=asset_type_service,
    )

    ctx = service._build_context("SPY")

    assert ctx.asset_type == "ETF"
    asset_type_service.resolve.assert_called_once_with("SPY")
    ticker_builder.get_by_symbol.assert_not_called()


def test_company_research_uses_research_symbol_data_service_for_asset_type():
    research_symbol_data_service = MagicMock()
    research_symbol_data_service.get_asset_type.return_value = "ETF"
    research_symbol_data_service.get_snapshot.return_value = None
    research_symbol_data_service.get_performance.return_value = None

    ticker_builder = MagicMock()
    ticker_builder.get_by_symbol.return_value = MagicMock(asset_type="STOCK")

    etf_service = MagicMock()
    etf_service.build_holdings_context.return_value = None

    fundamentals_builder = MagicMock()
    fundamentals_builder.build_etf_metrics.return_value = {}

    service = CompanyResearchService(
        company_profile_service=MagicMock(),
        market_service=MagicMock(),
        news_service=MagicMock(),
        fundamentals_builder=fundamentals_builder,
        sec_research_service=MagicMock(),
        earnings_service=MagicMock(),
        ticker_symbol_builder=ticker_builder,
        etf_research_service=etf_service,
        research_symbol_data_service=research_symbol_data_service,
    )

    ctx = service._build_context("SPY")

    assert ctx.asset_type == "ETF"
    research_symbol_data_service.get_asset_type.assert_called_once_with("SPY")
    ticker_builder.get_by_symbol.assert_not_called()


def test_company_research_snapshot_loader_uses_facade_and_preserves_gap():
    research_symbol_data_service = MagicMock()
    research_symbol_data_service.get_asset_type.return_value = "ETF"
    research_symbol_data_service.get_snapshot.side_effect = RuntimeError("snapshot")
    research_symbol_data_service.get_performance.return_value = None

    company_profile = MagicMock()
    market_service = MagicMock()
    fundamentals_builder = MagicMock()
    fundamentals_builder.build_etf_metrics.return_value = {}

    service = CompanyResearchService(
        company_profile_service=company_profile,
        market_service=market_service,
        news_service=MagicMock(),
        fundamentals_builder=fundamentals_builder,
        sec_research_service=MagicMock(),
        earnings_service=MagicMock(),
        etf_research_service=MagicMock(
            build_holdings_context=MagicMock(return_value=None)
        ),
        research_symbol_data_service=research_symbol_data_service,
    )

    ctx = service._build_context("SPY")

    assert ctx.snapshot is None
    assert "snapshot" in ctx.data_gaps
    research_symbol_data_service.get_snapshot.assert_called_once_with("SPY")
    company_profile.get_snapshot.assert_not_called()


def test_company_research_performance_loader_uses_facade_and_preserves_gap():
    research_symbol_data_service = MagicMock()
    research_symbol_data_service.get_asset_type.return_value = "ETF"
    research_symbol_data_service.get_snapshot.return_value = None
    research_symbol_data_service.get_performance.side_effect = RuntimeError(
        "performance"
    )

    market_service = MagicMock()
    fundamentals_builder = MagicMock()
    fundamentals_builder.build_etf_metrics.return_value = {}

    service = CompanyResearchService(
        company_profile_service=MagicMock(),
        market_service=market_service,
        news_service=MagicMock(),
        fundamentals_builder=fundamentals_builder,
        sec_research_service=MagicMock(),
        earnings_service=MagicMock(),
        etf_research_service=MagicMock(
            build_holdings_context=MagicMock(return_value=None)
        ),
        research_symbol_data_service=research_symbol_data_service,
    )

    ctx = service._build_context("SPY")

    assert ctx.performance is None
    assert "performance" in ctx.data_gaps
    research_symbol_data_service.get_performance.assert_called_once_with("SPY")
    market_service.get_performance.assert_not_called()
