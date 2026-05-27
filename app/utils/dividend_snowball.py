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


DEFAULT_PROJECT_YEARS = 10
PROJECT_YEAR_PRESETS = (5, 10, 15, 20, 25, 30)


def normalize_project_years(project_years: int | None) -> int:
    if project_years is None:
        return DEFAULT_PROJECT_YEARS
    return max(1, min(int(project_years), 50))


def latest_completed_dividend_per_share(annual_totals: dict[int, float]) -> float:
    completed = completed_annual_totals(annual_totals)
    if completed:
        return completed[-1][1]

    current_year = date.today().year
    positive_years = [
        (year, total)
        for year, total in sorted(annual_totals.items())
        if total > 0 and year <= current_year
    ]
    if positive_years:
        return positive_years[-1][1]
    return 0.0


def resolve_dividend_cagr_pct(
    annual_totals: dict[int, float],
    *,
    override_pct: float | None = None,
) -> float:
    if override_pct is not None:
        return override_pct

    resolved = dividend_cagr_pct(annual_totals, lookback_years=5)
    if resolved is not None:
        return resolved

    resolved = dividend_cagr_pct(annual_totals, lookback_years=10)
    return resolved if resolved is not None else 0.0


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


def simulate_forward_projection(
    *,
    shares: float,
    project_years: int,
    base_dps: float,
    dividend_cagr_pct: float,
    share_price: float | None,
    price_cagr_pct: float | None,
    reinvest_dividends: bool,
    current_year: int | None = None,
) -> dict[str, Any]:
    today_year = current_year or date.today().year
    end_year = today_year + project_years
    div_rate = dividend_cagr_pct / 100.0
    price_rate = (price_cagr_pct or 0.0) / 100.0

    annual_income_start = round(base_dps * shares, 2)
    total_collected = 0.0

    if not reinvest_dividends or share_price is None or share_price <= 0:
        for offset in range(project_years + 1):
            year_dps = base_dps * ((1.0 + div_rate) ** offset)
            total_collected += year_dps * shares

        final_dps = base_dps * ((1.0 + div_rate) ** project_years)
        return {
            "start_year": today_year,
            "latest_year": end_year,
            "project_years": project_years,
            "dividend_cagr_pct": dividend_cagr_pct,
            "annual_income_start": annual_income_start,
            "annual_income_latest": round(final_dps * shares, 2),
            "total_collected": round(total_collected, 2),
            "advanced": None,
        }

    price = share_price
    share_count = shares
    total_reinvested = 0.0

    for offset in range(project_years + 1):
        year_dps = base_dps * ((1.0 + div_rate) ** offset)
        dividend_cash = year_dps * share_count
        total_collected += dividend_cash

        if offset < project_years and price > 0:
            share_count += dividend_cash / price
            total_reinvested += dividend_cash

        if offset < project_years:
            price *= 1.0 + price_rate

    final_dps = base_dps * ((1.0 + div_rate) ** project_years)
    return {
        "start_year": today_year,
        "latest_year": end_year,
        "project_years": project_years,
        "dividend_cagr_pct": dividend_cagr_pct,
        "annual_income_start": annual_income_start,
        "annual_income_latest": round(final_dps * shares, 2),
        "total_collected": round(total_collected, 2),
        "advanced": {
            "enabled": True,
            "initial_shares": round(shares, 2),
            "final_shares": round(share_count, 2),
            "share_price_at_start": round(share_price, 2),
            "share_price_latest": round(price, 2),
            "price_cagr_pct": round(price_cagr_pct or 0.0, 2),
            "annual_income_latest_drip": round(final_dps * share_count, 2),
            "portfolio_value_latest": round(share_count * price, 2),
            "total_dividends_reinvested": round(total_reinvested, 2),
        },
    }


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
    project_years: int | None = None,
    dividend_cagr_pct: float | None = None,
) -> dict[str, Any]:
    _ = dividends
    _ = start_year

    resolved_project_years = normalize_project_years(project_years)
    resolved_shares = shares
    if (
        investment_usd is not None
        and investment_usd > 0
        and share_price is not None
        and share_price > 0
    ):
        resolved_shares = investment_usd / share_price

    base_dps = latest_completed_dividend_per_share(annual_totals)
    resolved_dividend_cagr = resolve_dividend_cagr_pct(
        annual_totals,
        override_pct=dividend_cagr_pct,
    )

    projection = simulate_forward_projection(
        shares=resolved_shares,
        project_years=resolved_project_years,
        base_dps=base_dps,
        dividend_cagr_pct=resolved_dividend_cagr,
        share_price=share_price,
        price_cagr_pct=price_cagr_pct,
        reinvest_dividends=reinvest_dividends,
    )

    scenario: dict[str, Any] = {
        "shares": round(resolved_shares, 2),
        "start_year": projection["start_year"],
        "latest_year": projection["latest_year"],
        "project_years": projection["project_years"],
        "dividend_cagr_pct": projection["dividend_cagr_pct"],
        "total_collected": projection["total_collected"],
        "annual_income_latest": projection["annual_income_latest"],
        "annual_income_start": projection["annual_income_start"],
    }

    if investment_usd is not None and investment_usd > 0:
        scenario["investment_usd"] = round(investment_usd, 2)
    if share_price is not None and share_price > 0:
        scenario["share_price"] = round(share_price, 2)
    if projection["advanced"] is not None:
        scenario["advanced"] = projection["advanced"]

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
            "share_price_at_start": round(share_price_at_start, 2),
            "share_price_latest": round(current_share_price, 2),
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
        "initial_shares": round(initial_shares, 2),
        "final_shares": round(shares, 2),
        "share_price_at_start": round(share_price_at_start, 2),
        "share_price_latest": round(current_share_price, 2),
        "price_cagr_pct": price_cagr_pct,
        "annual_income_latest_drip": round(latest_dps * shares, 2),
        "portfolio_value_latest": round(portfolio_value, 2),
        "total_dividends_reinvested": round(total_reinvested, 2),
    }
