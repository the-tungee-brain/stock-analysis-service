import math

from app.adapters.market.provider_symbol_profile_adapter import (
    ProviderSymbolProfileAdapter,
)


def test_normalized_fields_drop_non_finite_numbers():
    adapter = ProviderSymbolProfileAdapter.__new__(ProviderSymbolProfileAdapter)

    fields = adapter._normalized_fields(
        {
            "currentPrice": float("inf"),
            "regularMarketPreviousClose": "-inf",
            "marketCap": float("nan"),
            "totalAssets": math.inf,
            "volume": "-Infinity",
            "averageVolume": "NaN",
            "trailingPE": "-Infinity",
            "forwardPE": "Infinity",
            "priceToBook": math.nan,
            "dividendYield": float("inf"),
            "dividendRate": "-inf",
            "annualReportExpenseRatio": "nan",
            "beta": math.inf,
        }
    )

    assert fields["current_price"] is None
    assert fields["previous_close"] is None
    assert fields["market_cap"] is None
    assert fields["total_assets"] is None
    assert fields["volume"] is None
    assert fields["avg_volume"] is None
    assert fields["trailing_pe"] is None
    assert fields["forward_pe"] is None
    assert fields["price_to_book"] is None
    assert fields["dividend_yield"] is None
    assert fields["dividend_yield_pct"] is None
    assert fields["raw_dividend_yield"] is None
    assert fields["raw_dividend_yield_source"] is None
    assert fields["dividend_rate"] is None
    assert fields["expense_ratio"] is None
    assert fields["beta"] is None


def test_normalized_fields_store_raw_and_percent_point_dividend_yield():
    adapter = ProviderSymbolProfileAdapter.__new__(ProviderSymbolProfileAdapter)

    fields = adapter._normalized_fields(
        {
            "quoteType": "EQUITY",
            "dividendYield": 0.35,
        }
    )

    assert fields["dividend_yield"] == 0.35
    assert fields["raw_dividend_yield"] == 0.35
    assert fields["raw_dividend_yield_source"] == "yfinance.info.dividendYield"
    assert fields["dividend_yield_pct"] == 0.35


def test_normalized_fields_convert_etf_decimal_ratio_dividend_yield():
    adapter = ProviderSymbolProfileAdapter.__new__(ProviderSymbolProfileAdapter)

    fields = adapter._normalized_fields(
        {
            "quoteType": "ETF",
            "dividendYield": 0.0035,
        }
    )

    assert fields["dividend_yield"] == 0.0035
    assert fields["dividend_yield_pct"] == 0.35
