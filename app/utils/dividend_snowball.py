from __future__ import annotations

from datetime import date
from typing import Any


def parse_annual_totals(raw: Any) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}

    totals: dict[int, float] = {}
    for year_str, amount in raw.items():
        if not isinstance(year_str, str) or not isinstance(amount, (int, float)):
            continue
        try:
            year = int(year_str)
        except ValueError:
            continue
        totals[year] = float(amount)
    return totals


def completed_annual_totals(annual_totals: dict[int, float]) -> list[tuple[int, float]]:
    current_year = date.today().year
    rows = [
        (year, total)
        for year, total in sorted(annual_totals.items())
        if year < current_year and total > 0
    ]
    return rows


def dividend_cagr_pct(
    annual_totals: dict[int, float],
    *,
    lookback_years: int,
) -> float | None:
    rows = completed_annual_totals(annual_totals)
    if len(rows) < 2:
        return None

    end_year, end_value = rows[-1]
    target_start_year = end_year - lookback_years
    start_candidates = [(year, value) for year, value in rows if year <= target_start_year]
    if start_candidates:
        start_year, start_value = max(start_candidates, key=lambda item: item[0])
    else:
        start_year, start_value = rows[0]

    elapsed_years = end_year - start_year
    if elapsed_years < 1 or start_value <= 0:
        return None

    cagr = (end_value / start_value) ** (1 / elapsed_years) - 1
    return round(cagr * 100.0, 2)


def default_scenario_start_year(annual_totals: dict[int, float]) -> int:
    current_year = date.today().year
    completed_years = [year for year in annual_totals if year < current_year]
    if not completed_years:
        return current_year - 10
    return max(min(completed_years), current_year - 10)


def cash_collected_since_year(
    dividends: list[dict[str, Any]],
    *,
    shares: float,
    start_year: int,
) -> float:
    total = 0.0
    for item in dividends:
        payment_date = item.get("date")
        amount = item.get("amount_per_share")
        if not isinstance(payment_date, str) or not isinstance(amount, (int, float)):
            continue
        try:
            year = int(payment_date[:4])
        except ValueError:
            continue
        if year >= start_year:
            total += float(amount) * shares
    return round(total, 2)


def annual_income_on_shares(
    annual_totals: dict[int, float],
    *,
    shares: float,
    current_year: int | None = None,
) -> list[dict[str, Any]]:
    today_year = current_year or date.today().year
    rows: list[dict[str, Any]] = []

    for year in sorted(annual_totals):
        total_per_share = annual_totals[year]
        if total_per_share <= 0:
            continue
        rows.append(
            {
                "year": year,
                "total_per_share": round(total_per_share, 4),
                "income_on_shares": round(total_per_share * shares, 2),
                "is_partial_year": year >= today_year,
            }
        )

    return rows


def build_scenario(
    *,
    dividends: list[dict[str, Any]],
    annual_totals: dict[int, float],
    shares: float,
    start_year: int | None = None,
) -> dict[str, Any]:
    resolved_start_year = start_year or default_scenario_start_year(annual_totals)
    current_year = date.today().year
    completed = completed_annual_totals(annual_totals)

    latest_year = completed[-1][0] if completed else current_year - 1
    start_year_value = annual_totals.get(resolved_start_year, 0.0)
    latest_year_value = annual_totals.get(latest_year, 0.0)

    return {
        "shares": shares,
        "start_year": resolved_start_year,
        "total_collected": cash_collected_since_year(
            dividends,
            shares=shares,
            start_year=resolved_start_year,
        ),
        "annual_income_latest": round(latest_year_value * shares, 2),
        "annual_income_start": round(start_year_value * shares, 2),
        "latest_year": latest_year,
    }
