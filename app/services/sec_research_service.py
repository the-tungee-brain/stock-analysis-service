from typing import Literal

from app.adapters.sec.sec_edgar_adapter import SecEdgarAdapter
from app.builders.sec_cik_builder import SecCikBuilder
from app.builders.sec_financials_builder import SecFinancialsBuilder
from app.builders.sec_ratios_builder import REVENUE_TAGS
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
        """Compact latest annual metrics aligned to a single fiscal period."""
        financials = self.financials(symbol=symbol, period="annual", limit=3)
        ratios = self.ratios(symbol=symbol, period="annual", limit=3)

        if not ratios.snapshots:
            return []

        snapshot = None
        revenue = None
        net_income = None
        period_end = ""
        fiscal_period = ""
        period_note = "Latest annual figures from SEC filings."

        for candidate in ratios.snapshots:
            end = candidate.end
            fp = candidate.fiscal_period
            candidate_revenue = SecFinancialsBuilder.value_at_period(
                financials.income_statement,
                REVENUE_TAGS,
                end=end,
                fiscal_period=fp,
            )
            candidate_net_income = SecFinancialsBuilder.value_at_period(
                financials.income_statement,
                ["NetIncomeLoss"],
                end=end,
                fiscal_period=fp,
            )
            if not self._snapshot_metrics_are_consistent(
                revenue=candidate_revenue,
                net_income=candidate_net_income,
            ):
                continue

            snapshot = candidate
            revenue = candidate_revenue
            net_income = candidate_net_income
            period_end = end
            fiscal_period = fp
            period_note = (
                f"SEC filed figures for period ending {end} ({fp}). "
                "All metrics below use the same fiscal period."
            )
            break

        if snapshot is None:
            return []

        income = financials.income_statement
        balance = financials.balance_sheet
        cash_flow = financials.cash_flow

        ocf = SecFinancialsBuilder.value_at_period(
            cash_flow,
            ["NetCashProvidedByUsedInOperatingActivities"],
            end=period_end,
            fiscal_period=fiscal_period,
        )
        capex = SecFinancialsBuilder.value_at_period(
            cash_flow,
            ["PaymentsToAcquirePropertyPlantAndEquipment"],
            end=period_end,
            fiscal_period=fiscal_period,
        )
        equity = SecFinancialsBuilder.value_at_period(
            balance,
            ["StockholdersEquity"],
            end=period_end,
            fiscal_period=fiscal_period,
        )
        assets = SecFinancialsBuilder.value_at_period(
            balance,
            ["Assets"],
            end=period_end,
            fiscal_period=fiscal_period,
        )
        liabilities = SecFinancialsBuilder.value_at_period(
            balance,
            ["Liabilities"],
            end=period_end,
            fiscal_period=fiscal_period,
        )
        eps = SecFinancialsBuilder.value_at_period(
            income,
            ["EarningsPerShareDiluted"],
            end=period_end,
            fiscal_period=fiscal_period,
        )

        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf - abs(capex)

        return [
            self._metric("Revenue (SEC filed)", revenue, self._fmt_large_usd(revenue), period_note),
            self._metric("Net income (SEC filed)", net_income, self._fmt_large_usd(net_income), period_note),
            self._metric("Gross margin", snapshot.gross_margin, self._fmt_pct(snapshot.gross_margin), "Gross profit divided by revenue."),
            self._metric("Operating margin", snapshot.operating_margin, self._fmt_pct(snapshot.operating_margin), "Operating income divided by revenue."),
            self._metric("Net margin", snapshot.net_margin, self._fmt_pct(snapshot.net_margin), "Net income divided by revenue."),
            self._metric("Return on equity", snapshot.roe, self._fmt_pct(snapshot.roe), "Net income divided by shareholder equity."),
            self._metric("Return on assets", snapshot.roa, self._fmt_pct(snapshot.roa), "Net income divided by total assets."),
            self._metric("Debt / equity", snapshot.debt_to_equity, self._fmt_ratio(snapshot.debt_to_equity), "Total liabilities divided by shareholder equity."),
            self._metric("Free cash flow", fcf, self._fmt_large_usd(fcf), period_note),
            self._metric("FCF margin", snapshot.fcf_margin, self._fmt_pct(snapshot.fcf_margin), "Free cash flow divided by revenue."),
            self._metric("Revenue growth (YoY)", snapshot.revenue_growth_yoy, self._fmt_pct(snapshot.revenue_growth_yoy), period_note),
            self._metric("Net income growth (YoY)", snapshot.net_income_growth_yoy, self._fmt_pct(snapshot.net_income_growth_yoy), period_note),
            self._metric("EPS (diluted, SEC filed)", eps, self._fmt_dollar(eps), period_note),
            self._metric("Total assets", assets, self._fmt_large_usd(assets), period_note),
            self._metric("Total liabilities", liabilities, self._fmt_large_usd(liabilities), period_note),
            self._metric("Shareholders' equity", equity, self._fmt_large_usd(equity), period_note),
        ]

    @staticmethod
    def _snapshot_metrics_are_consistent(
        *,
        revenue: float | None,
        net_income: float | None,
    ) -> bool:
        if revenue is None or net_income is None:
            return revenue is not None or net_income is not None
        if revenue <= 0:
            return True
        return net_income <= revenue * 1.05

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
