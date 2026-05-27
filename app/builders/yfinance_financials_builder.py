from __future__ import annotations

import logging
from datetime import date
from typing import Any, Literal

import pandas as pd
import yfinance as yf

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.models.company_research_models import (
    FinancialLineItem,
    FinancialStatementsSnapshot,
    FinancialStrength,
    FinancialsPackage,
)

logger = logging.getLogger(__name__)

StrengthRating = Literal["strong", "solid", "mixed", "weak"]

INCOME_LINE_CANDIDATES: list[tuple[str, tuple[str, ...]]] = [
    ("Total revenue", ("TotalRevenue", "OperatingRevenue")),
    ("Gross profit", ("GrossProfit",)),
    ("Operating income", ("OperatingIncome", "TotalOperatingIncomeAsReported")),
    ("Net income", ("NetIncome", "NetIncomeCommonStockholders")),
    ("EBITDA", ("EBITDA", "NormalizedEBITDA")),
    ("Diluted EPS", ("DilutedEPS",)),
]

BALANCE_LINE_CANDIDATES: list[tuple[str, tuple[str, ...]]] = [
    ("Total assets", ("TotalAssets",)),
    ("Total liabilities", ("TotalLiabilitiesNetMinorityInterest",)),
    ("Stockholders' equity", ("StockholdersEquity", "CommonStockEquity")),
    ("Cash & equivalents", ("CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments")),
    ("Total debt", ("TotalDebt",)),
    ("Current assets", ("CurrentAssets",)),
    ("Current liabilities", ("CurrentLiabilities",)),
]

CASHFLOW_LINE_CANDIDATES: list[tuple[str, tuple[str, ...]]] = [
    ("Operating cash flow", ("OperatingCashFlow", "CashFlowFromContinuingOperatingActivities")),
    ("Capital expenditure", ("CapitalExpenditure",)),
    ("Free cash flow", ("FreeCashFlow",)),
    ("Investing cash flow", ("InvestingCashFlow", "CashFlowFromContinuingInvestingActivities")),
    ("Financing cash flow", ("FinancingCashFlow", "CashFlowFromContinuingFinancingActivities")),
]


class YFinanceFinancialsBuilder:
    MAX_PERIODS = 5

    def __init__(self, yfinance_adapter: YFinanceAdapter):
        self.yfinance_adapter = yfinance_adapter

    def build(self, symbol: str) -> FinancialsPackage | None:
        symbol_upper = symbol.strip().upper()
        try:
            ticker = yf.Ticker(symbol_upper)
            quarterly = self._build_snapshot(ticker, freq="quarterly")
            annual = self._build_snapshot(ticker, freq="yearly")
        except Exception:
            logger.exception("yfinance financial statements failed for %s", symbol_upper)
            return None

        if quarterly is None and annual is None:
            return None

        strength = self._assess_strength(
            symbol=symbol_upper,
            quarterly=quarterly,
            annual=annual,
            info=self.yfinance_adapter.get_ticker_info(symbol_upper),
        )
        return FinancialsPackage(
            quarterly=quarterly,
            annual=annual,
            strength=strength,
        )

    def _build_snapshot(
        self,
        ticker: yf.Ticker,
        *,
        freq: str,
    ) -> FinancialStatementsSnapshot | None:
        try:
            income = ticker.get_income_stmt(freq=freq)
            balance = ticker.get_balance_sheet(freq=freq)
            cashflow = ticker.get_cashflow(freq=freq)
        except Exception:
            logger.debug("yfinance %s statements unavailable", freq, exc_info=True)
            return None

        periods = self._collect_periods(income, balance, cashflow)
        if not periods:
            return None

        return FinancialStatementsSnapshot(
            periods=periods,
            income_statement=self._extract_lines(income, periods, INCOME_LINE_CANDIDATES),
            balance_sheet=self._extract_lines(balance, periods, BALANCE_LINE_CANDIDATES),
            cash_flow=self._extract_lines(cashflow, periods, CASHFLOW_LINE_CANDIDATES),
        )

    def _collect_periods(self, *frames: pd.DataFrame | None) -> list[str]:
        dates: list[date] = []
        for frame in frames:
            if frame is None or frame.empty:
                continue
            for column in frame.columns:
                parsed = YFinanceAdapter._index_to_date(column)
                if parsed is not None:
                    dates.append(parsed)
        if not dates:
            return []
        unique_sorted = sorted(set(dates), reverse=True)
        return [value.isoformat() for value in unique_sorted[: self.MAX_PERIODS]]

    def _extract_lines(
        self,
        frame: pd.DataFrame | None,
        periods: list[str],
        candidates: list[tuple[str, tuple[str, ...]]],
    ) -> list[FinancialLineItem]:
        if frame is None or frame.empty:
            return []

        rows: list[FinancialLineItem] = []
        for label, keys in candidates:
            series = self._row_for_keys(frame, keys)
            if series is None:
                continue
            values: dict[str, float | None] = {}
            for period in periods:
                values[period] = self._value_for_period(series, period)
            if any(value is not None for value in values.values()):
                rows.append(FinancialLineItem(label=label, values=values))
        return rows

    @staticmethod
    def _row_for_keys(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series | None:
        index_map = {str(idx): idx for idx in frame.index}
        for key in keys:
            if key in index_map:
                return frame.loc[index_map[key]]
        lowered = {str(idx).lower(): idx for idx in frame.index}
        for key in keys:
            match = lowered.get(key.lower())
            if match is not None:
                return frame.loc[match]
        return None

    @staticmethod
    def _value_for_period(series: pd.Series, period: str) -> float | None:
        for column, value in series.items():
            parsed = YFinanceAdapter._index_to_date(column)
            if parsed is not None and parsed.isoformat() == period:
                return YFinanceAdapter._optional_float(value)
        return None

    def _assess_strength(
        self,
        *,
        symbol: str,
        quarterly: FinancialStatementsSnapshot | None,
        annual: FinancialStatementsSnapshot | None,
        info: dict[str, Any],
    ) -> FinancialStrength:
        snapshot = annual or quarterly
        revenue = self._line_values(snapshot, "Total revenue") if snapshot else {}
        net_income = self._line_values(snapshot, "Net income") if snapshot else {}
        fcf = self._line_values(snapshot, "Free cash flow") if snapshot else {}
        periods = snapshot.periods if snapshot else []

        score = 50
        strengths: list[str] = []
        risks: list[str] = []
        highlights: list[str] = []

        rev_growth = self._yoy_change(revenue, periods)
        if rev_growth is not None:
            highlights.append(f"Revenue {'increased' if rev_growth >= 0 else 'declined'} {abs(rev_growth):.1f}% year over year.")
            if rev_growth >= 20:
                score += 15
                strengths.append("Strong revenue growth versus the prior year.")
            elif rev_growth >= 5:
                score += 8
                strengths.append("Revenue is still growing year over year.")
            elif rev_growth >= 0:
                score += 3
            else:
                score -= 12
                risks.append("Revenue is shrinking year over year.")

        latest_period = periods[0] if periods else None
        if latest_period and revenue.get(latest_period) and net_income.get(latest_period):
            rev = revenue[latest_period]
            ni = net_income[latest_period]
            if rev and ni is not None:
                margin = (ni / rev) * 100
                highlights.append(f"Net margin about {margin:.1f}% on the latest period.")
                if margin >= 20:
                    score += 12
                    strengths.append("High net profit margins suggest strong pricing power or efficiency.")
                elif margin >= 10:
                    score += 6
                elif margin >= 0:
                    score += 1
                else:
                    score -= 18
                    risks.append("The latest period shows negative net income.")

        fcf_trend = self._yoy_change(fcf, periods)
        latest_fcf = fcf.get(latest_period) if latest_period else None
        if latest_fcf is not None:
            highlights.append(
                f"Free cash flow {'$' + self._fmt_compact(latest_fcf) if latest_fcf >= 0 else '-' + self._fmt_compact(abs(latest_fcf))} in the latest period."
            )
            if latest_fcf > 0:
                score += 8
                strengths.append("Positive free cash flow supports buybacks, dividends, and reinvestment.")
            else:
                score -= 10
                risks.append("Negative free cash flow — cash burn or heavy reinvestment phase.")
            if fcf_trend is not None and fcf_trend > 10:
                score += 5

        debt_equity = info.get("debtToEquity")
        if isinstance(debt_equity, (int, float)):
            highlights.append(f"Debt-to-equity (market data) about {debt_equity:.2f}.")
            if debt_equity < 50:
                score += 8
                strengths.append("Balance sheet leverage looks moderate by debt-to-equity.")
            elif debt_equity < 100:
                score += 2
            else:
                score -= 8
                risks.append("Elevated leverage — gains and losses are amplified.")

        current_ratio = info.get("currentRatio")
        if isinstance(current_ratio, (int, float)):
            if current_ratio >= 1.5:
                score += 5
                strengths.append("Current ratio suggests comfortable short-term liquidity.")
            elif current_ratio < 1.0:
                score -= 8
                risks.append("Current ratio below 1.0 — watch near-term liquidity.")

        roe = info.get("returnOnEquity")
        if isinstance(roe, (int, float)):
            roe_pct = roe * 100 if abs(roe) <= 1.5 else roe
            highlights.append(f"Return on equity about {roe_pct:.1f}%.")
            if roe_pct >= 15:
                score += 6
                strengths.append("Strong return on equity.")
            elif roe_pct < 5:
                score -= 4

        score = max(0, min(100, score))
        rating = self._rating_from_score(score)
        headline = self._headline_for_rating(symbol, rating, score)

        if not strengths:
            strengths.append("Review the statement tables for margin and cash-flow trends.")
        if not risks:
            risks.append("Compare leverage and growth assumptions to your portfolio risk tolerance.")

        return FinancialStrength(
            rating=rating,
            score=score,
            headline=headline,
            strengths=strengths[:4],
            risks=risks[:4],
            highlights=highlights[:5],
        )

    @staticmethod
    def _line_values(
        snapshot: FinancialStatementsSnapshot | None,
        label: str,
    ) -> dict[str, float | None]:
        if snapshot is None:
            return {}
        for section in (
            snapshot.income_statement,
            snapshot.balance_sheet,
            snapshot.cash_flow,
        ):
            for row in section:
                if row.label.lower() == label.lower():
                    return dict(row.values)
        return {}

    @staticmethod
    def _yoy_change(values: dict[str, float | None], periods: list[str]) -> float | None:
        if len(periods) < 2:
            return None
        latest = values.get(periods[0])
        prior = values.get(periods[1])
        if latest is None or prior is None or prior == 0:
            return None
        return ((latest - prior) / abs(prior)) * 100

    @staticmethod
    def _rating_from_score(score: int) -> StrengthRating:
        if score >= 75:
            return "strong"
        if score >= 55:
            return "solid"
        if score >= 35:
            return "mixed"
        return "weak"

    @staticmethod
    def _headline_for_rating(symbol: str, rating: StrengthRating, score: int) -> str:
        labels = {
            "strong": "Financial profile looks strong",
            "solid": "Financial profile looks solid",
            "mixed": "Financial profile is mixed",
            "weak": "Financial profile looks weak",
        }
        return f"{labels[rating]} for {symbol.upper()} (score {score}/100)."

    @staticmethod
    def _fmt_compact(value: float) -> str:
        abs_val = abs(value)
        if abs_val >= 1_000_000_000_000:
            return f"{abs_val / 1_000_000_000_000:.1f}T"
        if abs_val >= 1_000_000_000:
            return f"{abs_val / 1_000_000_000:.1f}B"
        if abs_val >= 1_000_000:
            return f"{abs_val / 1_000_000:.1f}M"
        return f"{abs_val:,.0f}"
