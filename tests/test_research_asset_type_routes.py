import asyncio
from unittest.mock import MagicMock

from app.api.get_etf_funds_route import get_etf_funds
from app.api.get_street_analysis_route import get_street_analysis
from app.models.ticker_symbol_models import TickerSymbolItem
from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.models.yfinance_funds_models import EtfFundsSnapshot
from app.services.asset_type_service import AssetTypeService


def _ticker_service(asset_type: str | None) -> MagicMock:
    service = MagicMock()
    builder = MagicMock()
    service.get_by_symbol.return_value = TickerSymbolItem(
        symbol="SPY" if asset_type in {"ETF", "FUND", "MUTUAL_FUND"} else "AAPL",
        title="Test Symbol",
        asset_type=asset_type,
    )
    builder.get_by_symbol = service.get_by_symbol
    service.ticker_symbol_builder = builder
    return service


def _asset_type_service(
    ticker_service: MagicMock,
    etf_research_service: MagicMock,
) -> AssetTypeService:
    return AssetTypeService(
        ticker_symbol_builder=ticker_service.ticker_symbol_builder,
        etf_research_service=etf_research_service,
    )


def test_asset_type_service_ticker_hit_avoids_etf_holdings_fallback():
    ticker_service = _ticker_service("ETF")
    etf_research_service = MagicMock()
    service = _asset_type_service(ticker_service, etf_research_service)

    assert service.resolve("spy") == "ETF"
    ticker_service.get_by_symbol.assert_called_once_with(symbol="SPY")
    etf_research_service.is_etf_symbol.assert_not_called()


def test_asset_type_service_missing_ticker_asset_type_uses_etf_fallback():
    ticker_service = _ticker_service(None)
    etf_research_service = MagicMock()
    etf_research_service.is_etf_symbol.return_value = True
    service = _asset_type_service(ticker_service, etf_research_service)

    assert service.resolve("spy") == "ETF"
    ticker_service.get_by_symbol.assert_called_once_with(symbol="SPY")
    etf_research_service.is_etf_symbol.assert_called_once_with(symbol="SPY")


def test_asset_type_service_preserves_fund_from_ticker_symbols():
    ticker_service = _ticker_service("FUND")
    etf_research_service = MagicMock()
    service = _asset_type_service(ticker_service, etf_research_service)

    assert service.resolve("vfiax") == "FUND"
    ticker_service.get_by_symbol.assert_called_once_with(symbol="VFIAX")
    etf_research_service.is_etf_symbol.assert_not_called()


def test_street_analysis_for_etf_avoids_context_and_yfinance_builder():
    ticker_service = _ticker_service("ETF")
    etf_research_service = MagicMock()
    yfinance_analysis_builder = MagicMock()

    result = asyncio.run(
        get_street_analysis(
            symbol="SPY",
            asset_type_service=_asset_type_service(
                ticker_service,
                etf_research_service,
            ),
            yfinance_analysis_builder=yfinance_analysis_builder,
        )
    )

    assert result.model_dump(mode="json", by_alias=True) == {"streetAnalysis": None}
    ticker_service.get_by_symbol.assert_called_once_with(symbol="SPY")
    etf_research_service.is_etf_symbol.assert_not_called()
    yfinance_analysis_builder.build.assert_not_called()


def test_street_analysis_for_fund_avoids_stock_builder():
    ticker_service = _ticker_service("FUND")
    etf_research_service = MagicMock()
    yfinance_analysis_builder = MagicMock()

    result = asyncio.run(
        get_street_analysis(
            symbol="VFIAX",
            asset_type_service=_asset_type_service(
                ticker_service,
                etf_research_service,
            ),
            yfinance_analysis_builder=yfinance_analysis_builder,
        )
    )

    assert result.model_dump(mode="json", by_alias=True) == {"streetAnalysis": None}
    ticker_service.get_by_symbol.assert_called_once_with(symbol="VFIAX")
    etf_research_service.is_etf_symbol.assert_not_called()
    yfinance_analysis_builder.build.assert_not_called()


def test_street_analysis_for_stock_calls_yfinance_builder():
    ticker_service = _ticker_service("STOCK")
    etf_research_service = MagicMock()
    expected = StreetAnalysisSnapshot(consensus_label="Buy")
    yfinance_analysis_builder = MagicMock()
    yfinance_analysis_builder.build.return_value = expected

    result = asyncio.run(
        get_street_analysis(
            symbol="aapl",
            asset_type_service=_asset_type_service(
                ticker_service,
                etf_research_service,
            ),
            yfinance_analysis_builder=yfinance_analysis_builder,
        )
    )

    assert result.street_analysis == expected
    assert result.model_dump(mode="json", by_alias=True)["streetAnalysis"][
        "consensusLabel"
    ] == "Buy"
    ticker_service.get_by_symbol.assert_called_once_with(symbol="AAPL")
    etf_research_service.is_etf_symbol.assert_not_called()
    yfinance_analysis_builder.build.assert_called_once_with(symbol="AAPL")


def test_street_analysis_builder_failure_returns_null_shape():
    ticker_service = _ticker_service("STOCK")
    etf_research_service = MagicMock()
    yfinance_analysis_builder = MagicMock()
    yfinance_analysis_builder.build.side_effect = RuntimeError("yahoo unavailable")

    result = asyncio.run(
        get_street_analysis(
            symbol="AAPL",
            asset_type_service=_asset_type_service(
                ticker_service,
                etf_research_service,
            ),
            yfinance_analysis_builder=yfinance_analysis_builder,
        )
    )

    assert result.model_dump(mode="json", by_alias=True) == {"streetAnalysis": None}


def test_etf_funds_for_non_etf_avoids_context_and_fund_builder():
    ticker_service = _ticker_service("STOCK")
    etf_research_service = MagicMock()
    yfinance_funds_builder = MagicMock()

    result = asyncio.run(
        get_etf_funds(
            symbol="AAPL",
            asset_type_service=_asset_type_service(
                ticker_service,
                etf_research_service,
            ),
            yfinance_funds_builder=yfinance_funds_builder,
        )
    )

    assert result.model_dump(mode="json", by_alias=True) == {"etfFunds": None}
    ticker_service.get_by_symbol.assert_called_once_with(symbol="AAPL")
    etf_research_service.is_etf_symbol.assert_not_called()
    yfinance_funds_builder.build.assert_not_called()


def test_etf_funds_for_etf_calls_fund_builder_directly():
    ticker_service = _ticker_service("ETF")
    etf_research_service = MagicMock()
    expected = EtfFundsSnapshot(category="Large Blend")
    yfinance_funds_builder = MagicMock()
    yfinance_funds_builder.build.return_value = expected

    result = asyncio.run(
        get_etf_funds(
            symbol="spy",
            asset_type_service=_asset_type_service(
                ticker_service,
                etf_research_service,
            ),
            yfinance_funds_builder=yfinance_funds_builder,
        )
    )

    assert result.etf_funds == expected
    assert result.model_dump(mode="json", by_alias=True)["etfFunds"]["category"] == (
        "Large Blend"
    )
    ticker_service.get_by_symbol.assert_called_once_with(symbol="SPY")
    etf_research_service.is_etf_symbol.assert_not_called()
    yfinance_funds_builder.build.assert_called_once_with(symbol="SPY")


def test_etf_funds_for_fund_calls_fund_builder_directly():
    ticker_service = _ticker_service("FUND")
    etf_research_service = MagicMock()
    expected = EtfFundsSnapshot(category="Index Fund")
    yfinance_funds_builder = MagicMock()
    yfinance_funds_builder.build.return_value = expected

    result = asyncio.run(
        get_etf_funds(
            symbol="vfiax",
            asset_type_service=_asset_type_service(
                ticker_service,
                etf_research_service,
            ),
            yfinance_funds_builder=yfinance_funds_builder,
        )
    )

    assert result.etf_funds == expected
    ticker_service.get_by_symbol.assert_called_once_with(symbol="VFIAX")
    etf_research_service.is_etf_symbol.assert_not_called()
    yfinance_funds_builder.build.assert_called_once_with(symbol="VFIAX")


def test_etf_funds_builder_failure_returns_null_shape():
    ticker_service = _ticker_service("ETF")
    etf_research_service = MagicMock()
    yfinance_funds_builder = MagicMock()
    yfinance_funds_builder.build.side_effect = RuntimeError("yahoo unavailable")

    result = asyncio.run(
        get_etf_funds(
            symbol="SPY",
            asset_type_service=_asset_type_service(
                ticker_service,
                etf_research_service,
            ),
            yfinance_funds_builder=yfinance_funds_builder,
        )
    )

    assert result.model_dump(mode="json", by_alias=True) == {"etfFunds": None}


def test_missing_ticker_asset_type_falls_back_to_etf_holdings_detection():
    ticker_service = MagicMock()
    ticker_builder = MagicMock()
    ticker_service.get_by_symbol.return_value = TickerSymbolItem(
        symbol="GLDM",
        title="Gold ETF",
        asset_type=None,
    )
    ticker_builder.get_by_symbol = ticker_service.get_by_symbol
    ticker_service.ticker_symbol_builder = ticker_builder
    etf_research_service = MagicMock()
    etf_research_service.is_etf_symbol.return_value = True
    yfinance_analysis_builder = MagicMock()

    result = asyncio.run(
        get_street_analysis(
            symbol="GLDM",
            asset_type_service=_asset_type_service(
                ticker_service,
                etf_research_service,
            ),
            yfinance_analysis_builder=yfinance_analysis_builder,
        )
    )

    assert result.street_analysis is None
    ticker_service.get_by_symbol.assert_called_once_with(symbol="GLDM")
    etf_research_service.is_etf_symbol.assert_called_once_with(symbol="GLDM")
    yfinance_analysis_builder.build.assert_not_called()
