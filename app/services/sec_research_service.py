from typing import Literal

from app.adapters.sec.sec_edgar_adapter import SecEdgarAdapter
from app.builders.sec_cik_builder import SecCikBuilder
from app.builders.sec_financials_builder import SecFinancialsBuilder
from app.builders.sec_ratios_builder import SecRatiosBuilder
from app.models.company_research_models import FundamentalMetric
from app.models.sec_research_models import (
    SecFilingsResponse,
    SecFinancialsResponse,
    SecLookupResponse,
    SecRatiosResponse,
)


class SecResearchService:
    def __init__(
        self,
        sec_edgar_adapter: SecEdgarAdapter,
        sec_cik_builder: SecCikBuilder,
        sec_financials_builder: SecFinancialsBuilder,
        sec_ratios_builder: SecRatiosBuilder,
    ) -> None:
        self.sec_edgar_adapter = sec_edgar_adapter
        self.sec_cik_builder = sec_cik_builder
        self.sec_financials_builder = sec_financials_builder
        self.sec_ratios_builder = sec_ratios_builder

    def lookup(self, symbol: str) -> SecLookupResponse:
        return self.sec_cik_builder.build_lookup(symbol=symbol)

    def filings(self, symbol: str, limit: int = 20) -> SecFilingsResponse:
        return self.sec_cik_builder.build_filings(symbol=symbol, limit=limit)

    def financials(
        self,
        symbol: str,
        period: Literal["annual", "quarterly"] = "annual",
        limit: int = 12,
    ) -> SecFinancialsResponse:
        resolved = self.sec_cik_builder.resolve_symbol(symbol=symbol)
        facts = self.sec_edgar_adapter.get_company_facts(cik=resolved.cik_int)
        entity_name = facts.get("entityName") or resolved.ticker_title or resolved.symbol

        return self.sec_financials_builder.build(
            symbol=resolved.symbol,
            cik=resolved.cik,
            entity_name=entity_name,
            company_facts=facts,
            period=period,
            limit=limit,
        )

    def ratios(
        self,
        symbol: str,
        period: Literal["annual", "quarterly"] = "annual",
        limit: int = 12,
    ) -> SecRatiosResponse:
        financials = self.financials(symbol=symbol, period=period, limit=limit)
        return self.sec_ratios_builder.build(
            symbol=financials.symbol,
            cik=financials.cik,
            entity_name=financials.entity_name,
            financials=financials,
            limit=limit,
        )

    def latest_fundamental_metrics(self, symbol: str) -> list[FundamentalMetric]:
        return [
            FundamentalMetric(
                label=str(item["label"]),
                value=str(item["value"]),
                note=str(item.get("note") or ""),
            )
            for item in self.latest_snapshot_metrics(symbol=symbol)
            if item.get("include") and item.get("value")
        ]

    def latest_snapshot_metrics(self, symbol: str) -> list[dict[str, str | None]]:
        """Compact latest annual metrics for merging into fundamentals UI."""
        financials = self.financials(symbol=symbol, period="annual", limit=2)
        ratios = self.ratios(symbol=symbol, period="annual", limit=2)

        if not ratios.snapshots:
            return []

        latest = ratios.snapshots[0]
        revenue = self._latest_value(financials.income_statement, [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ])
        net_income = self._latest_value(financials.income_statement, ["NetIncomeLoss"])
        ocf = self._latest_value(
            financials.cash_flow, ["NetCashProvidedByUsedInOperatingActivities"]
        )
        capex = self._latest_value(
            financials.cash_flow, ["PaymentsToAcquirePropertyPlantAndEquipment"]
        )
        equity = self._latest_value(financials.balance_sheet, ["StockholdersEquity"])
        assets = self._latest_value(financials.balance_sheet, ["Assets"])
        liabilities = self._latest_value(financials.balance_sheet, ["Liabilities"])
        eps = self._latest_value(financials.income_statement, ["EarningsPerShareDiluted"])

        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf - abs(capex)

        return [
            self._metric("Revenue (SEC filed)", revenue, self._fmt_large_usd(revenue), "Latest annual revenue from SEC filings."),
            self._metric("Net income (SEC filed)", net_income, self._fmt_large_usd(net_income), "Latest annual net income from SEC filings."),
            self._metric("Gross margin", latest.gross_margin, self._fmt_pct(latest.gross_margin), "Gross profit divided by revenue."),
            self._metric("Operating margin", latest.operating_margin, self._fmt_pct(latest.operating_margin), "Operating income divided by revenue."),
            self._metric("Net margin", latest.net_margin, self._fmt_pct(latest.net_margin), "Net income divided by revenue."),
            self._metric("Return on equity", latest.roe, self._fmt_pct(latest.roe), "Net income divided by shareholder equity."),
            self._metric("Return on assets", latest.roa, self._fmt_pct(latest.roa), "Net income divided by total assets."),
            self._metric("Debt / equity", latest.debt_to_equity, self._fmt_ratio(latest.debt_to_equity), "Total liabilities divided by shareholder equity."),
            self._metric("Free cash flow", fcf, self._fmt_large_usd(fcf), "Operating cash flow minus capital expenditures."),
            self._metric("FCF margin", latest.fcf_margin, self._fmt_pct(latest.fcf_margin), "Free cash flow divided by revenue."),
            self._metric("Revenue growth (YoY)", latest.revenue_growth_yoy, self._fmt_pct(latest.revenue_growth_yoy), "Year-over-year revenue change from SEC filings."),
            self._metric("Net income growth (YoY)", latest.net_income_growth_yoy, self._fmt_pct(latest.net_income_growth_yoy), "Year-over-year net income change from SEC filings."),
            self._metric("EPS (diluted, SEC filed)", eps, self._fmt_dollar(eps), "Diluted earnings per share from SEC filings."),
            self._metric("Total assets", assets, self._fmt_large_usd(assets), "Total assets from the latest annual balance sheet."),
            self._metric("Total liabilities", liabilities, self._fmt_large_usd(liabilities), "Total liabilities from the latest annual balance sheet."),
            self._metric("Shareholders' equity", equity, self._fmt_large_usd(equity), "Shareholders' equity from the latest annual balance sheet."),
        ]

    @staticmethod
    def _latest_value(line_items, tags: list[str]) -> float | None:
        tag_set = set(tags)
        for item in line_items:
            if item.tag in tag_set and item.observations:
                return item.observations[0].value
        return None

    @staticmethod
    def _metric(label: str, raw: float | None, value: str | None, note: str) -> dict:
        if value is None:
            return {"label": label, "value": None, "note": note, "include": False}
        return {"label": label, "value": value, "note": note, "include": True}

    @staticmethod
    def _fmt_pct(value: float | None) -> str | None:
        if value is None:
            return None
        return f"{value * 100:.1f}%"

    @staticmethod
    def _fmt_ratio(value: float | None) -> str | None:
        if value is None:
            return None
        return f"{value:.2f}"

    @staticmethod
    def _fmt_dollar(value: float | None) -> str | None:
        if value is None:
            return None
        return f"${value:.2f}"

    @staticmethod
    def _fmt_large_usd(value: float | None) -> str | None:
        if value is None:
            return None
        abs_val = abs(value)
        sign = "-" if value < 0 else ""
        if abs_val >= 1_000_000_000_000:
            return f"{sign}${abs_val / 1_000_000_000_000:.1f}T"
        if abs_val >= 1_000_000_000:
            return f"{sign}${abs_val / 1_000_000_000:.1f}B"
        if abs_val >= 1_000_000:
            return f"{sign}${abs_val / 1_000_000:.1f}M"
        return f"{sign}${abs_val:,.0f}"
