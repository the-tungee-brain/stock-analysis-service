import pytest

from app.utils.dividend_snowball import (
    build_scenario,
    cash_collected_since_year,
    derive_share_price_at_start,
    dividend_cagr_pct,
    parse_annual_totals,
    simulate_drip_backtest,
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


def test_build_scenario_from_investment_and_share_price():
    scenario = build_scenario(
        dividends=SCHD_DIVIDENDS,
        annual_totals=SCHD_ANNUAL_TOTALS,
        shares=100,
        start_year=2015,
        investment_usd=10_000,
        share_price=50,
    )
    assert scenario["shares"] == pytest.approx(200)
    assert scenario["investment_usd"] == pytest.approx(10_000)
    assert scenario["share_price"] == pytest.approx(50)
    assert scenario["annual_income_latest"] == pytest.approx(209.4)


def test_derive_share_price_at_start():
    price = derive_share_price_at_start(
        current_share_price=100,
        price_cagr_pct=10,
        years_elapsed=5,
    )
    assert price == pytest.approx(100 / (1.1**5), rel=1e-4)


def test_simulate_drip_backtest_grows_share_count():
    result = simulate_drip_backtest(
        annual_totals=SCHD_ANNUAL_TOTALS,
        start_year=2015,
        end_year=2024,
        initial_investment_usd=10_000,
        share_price_at_start=20,
        price_cagr_pct=8,
        current_share_price=80,
    )
    assert result["initial_shares"] == pytest.approx(500)
    assert result["final_shares"] > result["initial_shares"]
    assert result["annual_income_latest_drip"] > 0
    assert result["portfolio_value_latest"] > 10_000
    assert result["total_dividends_reinvested"] > 0


def test_build_scenario_with_advanced_drip():
    scenario = build_scenario(
        dividends=SCHD_DIVIDENDS,
        annual_totals=SCHD_ANNUAL_TOTALS,
        shares=100,
        start_year=2015,
        investment_usd=10_000,
        share_price=80,
        reinvest_dividends=True,
        price_cagr_pct=8,
    )
    assert scenario["advanced"] is not None
    assert scenario["advanced"]["final_shares"] > scenario["advanced"]["initial_shares"]
