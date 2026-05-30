from typing import Any, Optional

from app.adapters.cache.dividend_history_cache import DividendHistoryCache
from app.adapters.securitiesdb.securitiesdb_adapter import SecuritiesDbAdapter
from app.models.dividend_research_models import (
    AnnualDividendIncome,
    DividendHistoricalBacktest,
    DividendHistoryContext,
    DividendPaymentItem,
    DividendSnowballScenario,
)
from app.utils.dividend_snowball import (
    annual_income_on_shares,
    build_historical_backtest,
    build_scenario,
    dividend_cagr_pct as compute_dividend_cagr_pct,
    latest_completed_dividend_per_share,
    parse_annual_totals,
    resolve_dividend_yield_pct,
)
from app.utils.stock_price_cagr import fetch_price_cagr_pct

DEFAULT_SCENARIO_SHARES = 100.0
DEFAULT_RECENT_PAYMENTS = 16


def _parse_payment_items(
    dividend_rows: list[dict[str, Any]],
) -> list[DividendPaymentItem]:
    payments: list[DividendPaymentItem] = []
    for item in dividend_rows:
        payment_date = item.get("date")
        amount = item.get("amount_per_share")
        if isinstance(payment_date, str) and isinstance(amount, (int, float)):
            payments.append(
                DividendPaymentItem(
                    date=payment_date,
                    amount_per_share=float(amount),
                )
            )
    payments.sort(key=lambda payment: payment.date)
    return payments


class DividendResearchService:
    def __init__(
        self,
        securitiesdb_adapter: SecuritiesDbAdapter,
        dividend_history_cache: Optional[DividendHistoryCache] = None,
    ) -> None:
        self.securitiesdb_adapter = securitiesdb_adapter
        self.dividend_history_cache = dividend_history_cache

    def build_history_context(
        self,
        symbol: str,
        *,
        shares: float = DEFAULT_SCENARIO_SHARES,
        investment_usd: float | None = None,
        share_price: float | None = None,
        reinvest_dividends: bool = False,
        price_cagr_pct: float | None = None,
        project_years: int | None = None,
        dividend_cagr_pct: float | None = None,
        history_start_year: int | None = None,
        annual_contribution_usd: float = 0.0,
        include_snowball: bool = True,
    ) -> DividendHistoryContext | None:
        resolved_shares = max(float(shares), 0.0) or DEFAULT_SCENARIO_SHARES
        resolved_investment = (
            float(investment_usd) if investment_usd is not None and investment_usd > 0 else None
        )
        resolved_share_price = (
            float(share_price) if share_price is not None and share_price > 0 else None
        )
        if resolved_investment and resolved_share_price:
            resolved_shares = resolved_investment / resolved_share_price

        cache_key = None
        if self.dividend_history_cache is not None:
            cache_key = DividendHistoryCache.build_cache_key(
                shares=resolved_shares,
                investment_usd=resolved_investment,
                share_price=resolved_share_price,
                reinvest_dividends=reinvest_dividends,
                price_cagr_pct=price_cagr_pct,
                project_years=project_years,
                dividend_cagr_pct=dividend_cagr_pct,
                history_start_year=history_start_year,
                annual_contribution_usd=annual_contribution_usd,
            )
            cached = self.dividend_history_cache.get(symbol, cache_key)
            if cached is not None:
                return cached

        payload = self.securitiesdb_adapter.get_stock_dividends(symbol=symbol)
        if payload is None:
            return None

        data = payload.get("data")
        if not isinstance(data, dict):
            return None

        meta = payload.get("meta")
        meta_dict = meta if isinstance(meta, dict) else {}
        summary = data.get("summary")
        summary_dict = summary if isinstance(summary, dict) else {}

        annual_totals = parse_annual_totals(summary_dict.get("annual_totals"))
        if not annual_totals:
            return None

        raw_dividends = data.get("dividends")
        dividend_rows: list[dict[str, Any]] = []
        if isinstance(raw_dividends, list):
            for item in raw_dividends:
                if isinstance(item, dict):
                    dividend_rows.append(item)

        symbol_upper = str(data.get("ticker") or symbol).upper()
        resolved_price_cagr = (
            price_cagr_pct
            if price_cagr_pct is not None
            else fetch_price_cagr_pct(symbol_upper, lookback_years=5)
        )
        if resolved_share_price is not None and resolved_price_cagr is None:
            resolved_price_cagr = 0.0

        scenario_model: DividendSnowballScenario | None = None
        historical_backtest: DividendHistoricalBacktest | None = None
        historical_data: dict[str, Any] | None = None
        if include_snowball:
            scenario_data = build_scenario(
                dividends=dividend_rows,
                annual_totals=annual_totals,
                shares=resolved_shares,
                investment_usd=resolved_investment,
                share_price=resolved_share_price,
                reinvest_dividends=reinvest_dividends,
                price_cagr_pct=resolved_price_cagr,
                project_years=project_years,
                dividend_cagr_pct=dividend_cagr_pct,
                annual_contribution_usd=annual_contribution_usd,
            )
            scenario_model = DividendSnowballScenario.model_validate(scenario_data)

            historical_data = build_historical_backtest(
                dividends=dividend_rows,
                annual_totals=annual_totals,
                shares=resolved_shares,
                start_year=history_start_year,
                share_price=resolved_share_price,
                investment_usd=resolved_investment,
                price_cagr_pct=resolved_price_cagr,
                reinvest_dividends=reinvest_dividends,
                symbol=symbol_upper,
                annual_contribution_usd=annual_contribution_usd,
            )
            if historical_data is not None:
                historical_backtest = DividendHistoricalBacktest.model_validate(
                    historical_data
                )

        all_payments = _parse_payment_items(dividend_rows)
        recent_payments = list(reversed(all_payments[-DEFAULT_RECENT_PAYMENTS:]))

        total_dividends = summary_dict.get("total_dividends")
        if not isinstance(total_dividends, int):
            total_dividends = len(dividend_rows)

        total_splits = summary_dict.get("total_splits")
        if not isinstance(total_splits, int):
            total_splits = 0

        consecutive = summary_dict.get("consecutive_annual_increases")
        if not isinstance(consecutive, int):
            consecutive = 0

        base_dps = latest_completed_dividend_per_share(annual_totals)
        dividend_yield_pct = resolve_dividend_yield_pct(
            base_dps=base_dps,
            share_price=resolved_share_price,
            symbol=str(data.get("ticker") or symbol).upper(),
        )

        income_shares = resolved_shares
        if historical_data is not None and history_start_year is not None:
            initial_shares = historical_data.get("initial_shares")
            if isinstance(initial_shares, (int, float)) and initial_shares > 0:
                income_shares = float(initial_shares)

        context = DividendHistoryContext(
            ticker=str(data.get("ticker") or symbol).upper(),
            total_dividends=total_dividends,
            total_splits=total_splits,
            consecutive_annual_increases=consecutive,
            cagr_5y_pct=compute_dividend_cagr_pct(annual_totals, lookback_years=5),
            cagr_10y_pct=compute_dividend_cagr_pct(annual_totals, lookback_years=10),
            dividend_yield_pct=dividend_yield_pct,
            price_cagr_pct=resolved_price_cagr,
            annual_income=[
                AnnualDividendIncome.model_validate(row)
                for row in annual_income_on_shares(
                    annual_totals,
                    shares=income_shares,
                )
            ],
            recent_payments=recent_payments,
            payments=all_payments,
            scenario=scenario_model,
            historical_backtest=historical_backtest,
            data_as_of=self._extract_data_as_of(meta_dict),
            confidence_score=self._extract_confidence_score(meta_dict),
        )

        if self.dividend_history_cache is not None and cache_key is not None:
            self.dividend_history_cache.put(symbol, cache_key, context)

        return context

    @staticmethod
    def _extract_data_as_of(meta: dict[str, Any]) -> str | None:
        domains = meta.get("domains")
        if not isinstance(domains, dict):
            return None
        corporate_actions = domains.get("corporate_actions")
        if not isinstance(corporate_actions, dict):
            return None
        last_updated = corporate_actions.get("last_updated")
        return last_updated if isinstance(last_updated, str) else None

    @staticmethod
    def _extract_confidence_score(meta: dict[str, Any]) -> float | None:
        score = meta.get("confidence_score")
        if isinstance(score, (int, float)):
            return float(score)
        return None
