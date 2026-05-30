from datetime import date

import pytest

from app.utils.dividend_snowball import (
    build_historical_backtest,
    build_scenario,
    cash_collected_since_year,
    derive_share_price_at_start,
    dividend_cagr_pct,
    normalize_project_years,
    parse_annual_totals,
    resolve_dividend_yield_pct,
    simulate_drip_backtest,
    simulate_forward_projection,
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


def test_normalize_project_years():
    assert normalize_project_years(None) == 10
    assert normalize_project_years(15) == 15
    assert normalize_project_years(100) == 50


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


def test_simulate_forward_projection_flat_shares():
    current_year = 2026
    result = simulate_forward_projection(
        shares=100,
        project_years=10,
        base_dps=0.995,
        dividend_cagr_pct=5.0,
        share_price=80,
        price_cagr_pct=8.0,
        reinvest_dividends=False,
        current_year=current_year,
    )
    assert result["start_year"] == current_year
    assert result["latest_year"] == current_year + 10
    assert result["annual_income_start"] == pytest.approx(99.5)
    assert result["annual_income_latest"] > result["annual_income_start"]
    assert result["total_collected"] > result["annual_income_start"]


def test_simulate_forward_projection_flat_shares_with_price_growth():
    current_year = 2026
    result = simulate_forward_projection(
        shares=100,
        project_years=10,
        base_dps=0.995,
        dividend_cagr_pct=5.0,
        share_price=80,
        price_cagr_pct=8.0,
        reinvest_dividends=False,
        current_year=current_year,
    )
    assert result["advanced"] is not None
    assert result["advanced"]["final_shares"] == result["advanced"]["initial_shares"]
    assert result["advanced"]["portfolio_value_latest"] > 100 * 80
    assert result["advanced"]["total_dividends_reinvested"] == 0.0


def test_simulate_forward_projection_with_drip():
    current_year = 2026
    result = simulate_forward_projection(
        shares=100,
        project_years=10,
        base_dps=0.995,
        dividend_cagr_pct=5.0,
        share_price=80,
        price_cagr_pct=8.0,
        reinvest_dividends=True,
        current_year=current_year,
    )
    assert result["advanced"] is not None
    assert result["advanced"]["final_shares"] > result["advanced"]["initial_shares"]
    assert (
        result["advanced"]["annual_income_latest_drip"]
        == result["annual_income_latest"]
    )
    assert result["annual_income_latest"] > result["annual_income_start"]


def test_resolve_dividend_yield_pct_from_share_price():
    yield_pct = resolve_dividend_yield_pct(
        base_dps=1.0,
        share_price=50,
        symbol="SCHD",
    )
    assert yield_pct == pytest.approx(2.0)


def test_build_scenario_projects_from_current_year():
    current_year = date.today().year
    scenario = build_scenario(
        dividends=SCHD_DIVIDENDS,
        annual_totals=SCHD_ANNUAL_TOTALS,
        shares=100,
        project_years=10,
    )
    assert scenario["shares"] == 100
    assert scenario["start_year"] == current_year
    assert scenario["latest_year"] == current_year + 10
    assert scenario["project_years"] == 10
    assert scenario["annual_income_start"] == pytest.approx(104.7)
    assert scenario["annual_income_latest"] > scenario["annual_income_start"]
    assert scenario["total_collected"] > 0


def test_build_scenario_from_investment_and_share_price():
    scenario = build_scenario(
        dividends=SCHD_DIVIDENDS,
        annual_totals=SCHD_ANNUAL_TOTALS,
        shares=100,
        investment_usd=10_000,
        share_price=50,
        project_years=5,
    )
    assert scenario["shares"] == pytest.approx(200)
    assert scenario["investment_usd"] == pytest.approx(10_000)
    assert scenario["share_price"] == pytest.approx(50)
    assert scenario["project_years"] == 5


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
        investment_usd=10_000,
        share_price=80,
        reinvest_dividends=True,
        price_cagr_pct=8,
        project_years=10,
    )
    assert scenario["advanced"] is not None
    assert scenario["advanced"]["final_shares"] > scenario["advanced"]["initial_shares"]


def test_simulate_forward_projection_with_annual_contributions():
    current_year = 2026
    baseline = simulate_forward_projection(
        shares=100,
        project_years=10,
        base_dps=0.995,
        dividend_cagr_pct=5.0,
        share_price=80,
        price_cagr_pct=8.0,
        reinvest_dividends=True,
        current_year=current_year,
    )
    with_contrib = simulate_forward_projection(
        shares=100,
        project_years=10,
        base_dps=0.995,
        dividend_cagr_pct=5.0,
        share_price=80,
        price_cagr_pct=8.0,
        reinvest_dividends=True,
        annual_contribution_usd=5_000,
        current_year=current_year,
    )
    assert with_contrib["advanced"] is not None
    assert baseline["advanced"] is not None
    assert (
        with_contrib["advanced"]["final_shares"]
        > baseline["advanced"]["final_shares"]
    )
    assert with_contrib["advanced"]["total_annual_contributions_usd"] == pytest.approx(
        50_000
    )


def test_simulate_drip_backtest_with_annual_contributions():
    baseline = simulate_drip_backtest(
        annual_totals=SCHD_ANNUAL_TOTALS,
        start_year=2015,
        end_year=2024,
        initial_investment_usd=10_000,
        share_price_at_start=20,
        price_cagr_pct=8,
        current_share_price=80,
    )
    with_contrib = simulate_drip_backtest(
        annual_totals=SCHD_ANNUAL_TOTALS,
        start_year=2015,
        end_year=2024,
        initial_investment_usd=10_000,
        share_price_at_start=20,
        price_cagr_pct=8,
        current_share_price=80,
        annual_contribution_usd=35_000,
    )

    assert with_contrib["final_shares"] > baseline["final_shares"]
    assert with_contrib["portfolio_value_latest"] > baseline["portfolio_value_latest"]
    assert with_contrib["total_annual_contributions_usd"] == pytest.approx(315_000)


def test_build_historical_backtest_uses_start_price_share_count():
    result = build_historical_backtest(
        dividends=SCHD_DIVIDENDS,
        annual_totals=SCHD_ANNUAL_TOTALS,
        shares=100,
        start_year=2015,
        share_price=80,
        investment_usd=10_000,
        price_cagr_pct=8,
        reinvest_dividends=True,
        symbol="SCHD",
    )

    assert result is not None
    price_at_start = derive_share_price_at_start(
        current_share_price=80,
        price_cagr_pct=8,
        years_elapsed=result["end_year"] - result["start_year"],
    )
    expected_shares = 10_000 / price_at_start
    assert result["initial_shares"] == pytest.approx(expected_shares, rel=1e-4)
    assert result["drip"] is not None
    assert result["drip"]["initial_shares"] == pytest.approx(result["initial_shares"])
    assert result["cash_collected"] == pytest.approx(
        result["drip"]["total_dividends_collected"]
    )
    assert result["cash_collected_annual"] == pytest.approx(result["cash_collected"])


def test_build_historical_backtest_includes_cash_and_drip():
    result = build_historical_backtest(
        dividends=SCHD_DIVIDENDS,
        annual_totals=SCHD_ANNUAL_TOTALS,
        shares=100,
        start_year=2015,
        share_price=80,
        investment_usd=10_000,
        price_cagr_pct=8,
        symbol="SCHD",
    )

    assert result is not None
    assert result["start_year"] == 2015
    assert result["end_year"] == 2025
    assert result["cash_collected"] > 0
    assert result["cash_collected_annual"] > 0
    assert result["drip"] is not None
    assert result["drip"]["final_shares"] > result["drip"]["initial_shares"]


def test_build_historical_backtest_omits_drip_when_reinvest_disabled():
    result = build_historical_backtest(
        dividends=SCHD_DIVIDENDS,
        annual_totals=SCHD_ANNUAL_TOTALS,
        shares=100,
        start_year=2015,
        share_price=80,
        investment_usd=10_000,
        price_cagr_pct=8,
        reinvest_dividends=False,
        symbol="SCHD",
    )

    assert result is not None
    assert result["cash_collected"] > 0
    assert result["drip"] is None
