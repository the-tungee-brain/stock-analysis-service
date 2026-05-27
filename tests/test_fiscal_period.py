from datetime import date

from app.broker.fiscal_period import (
    fiscal_quarter_and_year,
    fiscal_quarter_and_year_for_earnings_report,
    format_fiscal_period,
)


def test_nvda_q1_fy2027_from_april_period_end():
    quarter, year = fiscal_quarter_and_year(
        date(2026, 4, 30),
        fiscal_year_end_month=1,
    )
    assert quarter == 1
    assert year == 2027
    assert format_fiscal_period(quarter, year) == "Q1 2027"


def test_nvda_q2_fy2027_from_august_report_date():
    quarter, year = fiscal_quarter_and_year_for_earnings_report(
        date(2026, 8, 26),
        fiscal_year_end_month=1,
    )
    assert quarter == 2
    assert year == 2027


def test_aapl_q2_fy2026_from_march_period_end():
    quarter, year = fiscal_quarter_and_year(
        date(2026, 3, 31),
        fiscal_year_end_month=9,
    )
    assert quarter == 2
    assert year == 2026
