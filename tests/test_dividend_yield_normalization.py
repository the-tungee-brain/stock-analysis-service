from app.utils.dividend_yield import dividend_yield_pct_or_none


def test_equity_percent_point_yield_stays_percent_points():
    assert dividend_yield_pct_or_none(0.35, asset_type="STOCK") == 0.35


def test_equity_decimal_ratio_yield_converts_to_percent_points():
    assert dividend_yield_pct_or_none(0.0035, asset_type="STOCK") == 0.35


def test_etf_decimal_ratio_yield_converts_to_percent_points():
    assert dividend_yield_pct_or_none(0.0035, asset_type="ETF") == 0.35


def test_etf_large_raw_yield_requires_explicit_decimal_ratio_convention():
    assert dividend_yield_pct_or_none(0.35, asset_type="ETF") is None
    assert (
        dividend_yield_pct_or_none(
            0.35,
            asset_type="ETF",
            convention="decimal_ratio",
        )
        == 35.0
    )


def test_aapl_regression_never_normalizes_to_35_pct():
    assert dividend_yield_pct_or_none(0.35, asset_type="STOCK") != 35.0
