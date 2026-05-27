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
    investment_usd: float | None = None,
    share_price: float | None = None,
    reinvest_dividends: bool = False,
    price_cagr_pct: float | None = None,
) -> dict[str, Any]:
    resolved_start_year = start_year or default_scenario_start_year(annual_totals)
    current_year = date.today().year
    completed = completed_annual_totals(annual_totals)

    latest_year = completed[-1][0] if completed else current_year - 1
    start_year_value = annual_totals.get(resolved_start_year, 0.0)
    latest_year_value = annual_totals.get(latest_year, 0.0)

    resolved_shares = shares
    if (
        investment_usd is not None
        and investment_usd > 0
        and share_price is not None
        and share_price > 0
    ):
        resolved_shares = investment_usd / share_price

    scenario: dict[str, Any] = {
        "shares": round(resolved_shares, 6),
        "start_year": resolved_start_year,
        "total_collected": cash_collected_since_year(
            dividends,
            shares=resolved_shares,
            start_year=resolved_start_year,
        ),
        "annual_income_latest": round(latest_year_value * resolved_shares, 2),
        "annual_income_start": round(start_year_value * resolved_shares, 2),
        "latest_year": latest_year,
    }

    if investment_usd is not None and investment_usd > 0:
        scenario["investment_usd"] = round(investment_usd, 2)
    if share_price is not None and share_price > 0:
        scenario["share_price"] = round(share_price, 4)

    if (
        reinvest_dividends
        and investment_usd is not None
        and investment_usd > 0
        and share_price is not None
        and share_price > 0
        and price_cagr_pct is not None
    ):
        years_elapsed = latest_year - resolved_start_year
        share_price_at_start = derive_share_price_at_start(
            current_share_price=share_price,
            price_cagr_pct=price_cagr_pct,
            years_elapsed=years_elapsed,
        )
        scenario["advanced"] = simulate_drip_backtest(
            annual_totals=annual_totals,
            start_year=resolved_start_year,
            end_year=latest_year,
            initial_investment_usd=investment_usd,
            share_price_at_start=share_price_at_start,
            price_cagr_pct=price_cagr_pct,
            current_share_price=share_price,
        )

    return scenario


def derive_share_price_at_start(
    *,
    current_share_price: float,
    price_cagr_pct: float,
    years_elapsed: int,
) -> float:
    if years_elapsed <= 0 or current_share_price <= 0:
        return current_share_price

    rate = price_cagr_pct / 100.0
    denominator = (1.0 + rate) ** years_elapsed
    if denominator <= 0:
        return current_share_price
    return current_share_price / denominator


def simulate_drip_backtest(
    *,
    annual_totals: dict[int, float],
    start_year: int,
    end_year: int,
    initial_investment_usd: float,
    share_price_at_start: float,
    price_cagr_pct: float,
    current_share_price: float,
) -> dict[str, Any]:
    if (
        initial_investment_usd <= 0
        or share_price_at_start <= 0
        or end_year < start_year
    ):
        return {
            "enabled": True,
            "initial_shares": 0.0,
            "final_shares": 0.0,
            "share_price_at_start": round(share_price_at_start, 4),
            "share_price_latest": round(current_share_price, 4),
            "price_cagr_pct": price_cagr_pct,
            "annual_income_latest_drip": 0.0,
            "portfolio_value_latest": 0.0,
            "total_dividends_reinvested": 0.0,
        }

    shares = initial_investment_usd / share_price_at_start
    initial_shares = shares
    price = share_price_at_start
    rate = price_cagr_pct / 100.0
    total_reinvested = 0.0

    for year in range(start_year, end_year + 1):
        dps = annual_totals.get(year, 0.0)
        if dps > 0:
            dividend_cash = dps * shares
            if year < end_year and price > 0:
                shares += dividend_cash / price
                total_reinvested += dividend_cash

        if year < end_year:
            price *= 1.0 + rate

    latest_dps = annual_totals.get(end_year, 0.0)
    portfolio_value = shares * current_share_price

    return {
        "enabled": True,
        "initial_shares": round(initial_shares, 6),
        "final_shares": round(shares, 6),
        "share_price_at_start": round(share_price_at_start, 4),
        "share_price_latest": round(current_share_price, 4),
        "price_cagr_pct": price_cagr_pct,
        "annual_income_latest_drip": round(latest_dps * shares, 2),
        "portfolio_value_latest": round(portfolio_value, 2),
        "total_dividends_reinvested": round(total_reinvested, 2),
    }
