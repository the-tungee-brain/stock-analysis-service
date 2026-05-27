from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


def fiscal_year_end_month_from_info(info: dict) -> int | None:
    """Infer fiscal year-end month (1–12) from yfinance ``lastFiscalYearEnd`` unix timestamp."""
    raw = info.get("lastFiscalYearEnd")
    if raw is None:
        return None
    try:
        ts = float(raw)
    except (TypeError, ValueError):
        return None
    try:
        fy_end = datetime.fromtimestamp(ts, tz=timezone.utc).date()
    except (OSError, OverflowError, ValueError):
        return None
    return fy_end.month


def fiscal_quarter_and_year(
    period_end: date,
    *,
    fiscal_year_end_month: int | None,
) -> tuple[int | None, int | None]:
    """
    Map a fiscal period-end date to (quarter, fiscal_year_label).

    ``fiscal_year`` is the year used in labels like "Q1 2027" (the year the fiscal year ends in).
    """
    if fiscal_year_end_month is None or not 1 <= fiscal_year_end_month <= 12:
        calendar_quarter = (period_end.month - 1) // 3 + 1
        return calendar_quarter, period_end.year

    fy_start_month = (fiscal_year_end_month % 12) + 1

    if period_end.month <= fiscal_year_end_month:
        fiscal_year = period_end.year
    else:
        fiscal_year = period_end.year + 1

    if period_end.month >= fy_start_month:
        months_into_fy = period_end.month - fy_start_month
    else:
        months_into_fy = (12 - fy_start_month) + period_end.month

    fiscal_quarter = months_into_fy // 3 + 1
    return fiscal_quarter, fiscal_year


def approximate_quarter_end_from_report_date(report_date: date) -> date:
    """Map an earnings report date to an approximate fiscal period-end (prior month-end)."""
    first_of_month = report_date.replace(day=1)
    return first_of_month - timedelta(days=1)


def fiscal_quarter_and_year_for_earnings_report(
    report_date: date,
    *,
    fiscal_year_end_month: int | None,
) -> tuple[int | None, int | None]:
    return fiscal_quarter_and_year(
        approximate_quarter_end_from_report_date(report_date),
        fiscal_year_end_month=fiscal_year_end_month,
    )


def format_fiscal_period(quarter: int | None, year: int | None) -> str:
    if quarter and year:
        return f"Q{quarter} {year}"
    if year:
        return str(year)
    return "Unknown period"
