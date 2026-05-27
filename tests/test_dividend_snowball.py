import pytest

from app.utils.dividend_snowball import (
    build_scenario,
    cash_collected_since_year,
    dividend_cagr_pct,
    parse_annual_totals,
)


SCHD_ANNUAL_TOTALS = {
    2015: 0.3823,
    2018: 0.48,
    2020: 0.6763,
    2022: 0.854,
    2024: 0.995,
    2025: 1.047,
    2026: 0.257,
}

SCHD_DIVIDENDS = [
    {"date": "2025-12-10", "amount_per_share": 0.278},
    {"date": "2025-09-24", "amount_per_share": 0.26},
    {"date": "2024-12-11", "amount_per_share": 0.249},
    {"date": "2015-03-25", "amount_per_share": 0.093},
]


def test_parse_annual_totals():
    parsed = parse_annual_totals(
        {"2024": 0.995, "2025": 1.047, "bad": "x", 2023: 0.5}
    )
    assert parsed[2024] == pytest.approx(0.995)
    assert parsed[2025] == pytest.approx(1.047)
    assert 2023 not in parsed


def test_dividend_cagr_pct_uses_completed_years_only():
    cagr = dividend_cagr_pct(SCHD_ANNUAL_TOTALS, lookback_years=5)
    assert cagr is not None
    assert cagr > 5


def test_cash_collected_since_year():
    total = cash_collected_since_year(
        SCHD_DIVIDENDS,
        shares=100,
        start_year=2024,
    )
    assert total == pytest.approx(78.7)


def test_build_scenario():
    scenario = build_scenario(
        dividends=SCHD_DIVIDENDS,
        annual_totals=SCHD_ANNUAL_TOTALS,
        shares=100,
        start_year=2015,
    )
    assert scenario["shares"] == 100
    assert scenario["start_year"] == 2015
    assert scenario["annual_income_latest"] == pytest.approx(104.7)
    assert scenario["annual_income_start"] == pytest.approx(38.23)
    assert scenario["total_collected"] > 0
