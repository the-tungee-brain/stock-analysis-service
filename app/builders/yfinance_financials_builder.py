from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.adapters.market.yfinance_bootstrap import yfinance_fetch_lock
from app.builders.canonical_financial_metrics import build_canonical_metrics
from app.builders.financial_overview_generator import FinancialOverviewGenerator
from app.models.company_research_models import (
    FinancialLineItem,
    FinancialStatementsSnapshot,
    FinancialStrength,
    FinancialsPackage,
)

logger = logging.getLogger(__name__)

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
    (
        "Dividends paid",
        (
            "CommonStockDividendPaid",
            "CashDividendsPaid",
            "PaymentOfDividends",
            "CommonStockDividendsPaid",
        ),
    ),
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
            with yfinance_fetch_lock():
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
        canonical = build_canonical_metrics(info=info, snapshot=snapshot)
        overview = FinancialOverviewGenerator().generate(symbol, canonical)
        return FinancialStrength(
            rating=overview.rating,
            score=overview.score,
            headline=overview.headline,
            strengths=overview.strengths,
            risks=overview.risks,
            highlights=overview.highlights,
            key_metrics=canonical.to_key_metrics(),
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
