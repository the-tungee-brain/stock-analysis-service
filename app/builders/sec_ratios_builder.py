from typing import Literal

from app.models.sec_research_models import (
    FinancialLineItem,
    RatioSnapshot,
    SecFinancialsResponse,
    SecRatiosResponse,
)

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]


class SecRatiosBuilder:
    def build(
        self,
        *,
        symbol: str,
        cik: str,
        entity_name: str,
        financials: SecFinancialsResponse,
        limit: int = 12,
    ) -> SecRatiosResponse:
        income = financials.income_statement
        balance = financials.balance_sheet
        cash_flow = financials.cash_flow

        revenue_map = self._values_for_tags(income, REVENUE_TAGS)
        gross_profit_map = self._values_for_tags(income, ["GrossProfit"])
        operating_income_map = self._values_for_tags(income, ["OperatingIncomeLoss"])
        net_income_map = self._values_for_tags(income, ["NetIncomeLoss"])
        assets_map = self._values_for_tags(balance, ["Assets"])
        equity_map = self._values_for_tags(balance, ["StockholdersEquity"])
        liabilities_map = self._values_for_tags(balance, ["Liabilities"])
        ocf_map = self._values_for_tags(
            cash_flow, ["NetCashProvidedByUsedInOperatingActivities"]
        )
        capex_map = self._values_for_tags(
            cash_flow, ["PaymentsToAcquirePropertyPlantAndEquipment"]
        )
        fiscal_year_map = self._fiscal_years(income + balance + cash_flow)

        period_keys = sorted(
            set(revenue_map.keys()) | set(net_income_map.keys()),
            key=lambda k: k[0],
            reverse=True,
        )[:limit]

        snapshots: list[RatioSnapshot] = []
        for end, fp in period_keys:
            revenue = revenue_map.get((end, fp))
            gross_profit = gross_profit_map.get((end, fp))
            operating_income = operating_income_map.get((end, fp))
            net_income = net_income_map.get((end, fp))
            assets = assets_map.get((end, fp))
            equity = equity_map.get((end, fp))
            liabilities = liabilities_map.get((end, fp))
            ocf = ocf_map.get((end, fp))
            capex = capex_map.get((end, fp))

            fcf = None
            if ocf is not None and capex is not None:
                fcf = ocf - abs(capex)

            snapshots.append(
                RatioSnapshot(
                    end=end,
                    fiscal_period=fp,
                    fiscal_year=fiscal_year_map.get((end, fp)),
                    gross_margin=self._safe_div(gross_profit, revenue),
                    operating_margin=self._safe_div(operating_income, revenue),
                    net_margin=self._safe_div(net_income, revenue),
                    roe=self._safe_div(net_income, equity),
                    roa=self._safe_div(net_income, assets),
                    debt_to_equity=self._safe_div(liabilities, equity),
                    free_cash_flow=fcf,
                    fcf_margin=self._safe_div(fcf, revenue),
                    revenue_growth_yoy=self._yoy_growth(revenue_map, end, fp, revenue),
                    net_income_growth_yoy=self._yoy_growth(
                        net_income_map, end, fp, net_income
                    ),
                )
            )

        return SecRatiosResponse(
            symbol=symbol,
            cik=cik,
            entity_name=entity_name,
            period=financials.period,
            snapshots=snapshots,
        )

    @staticmethod
    def _values_for_tags(
        line_items: list[FinancialLineItem], tag_candidates: list[str]
    ) -> dict[tuple[str, str], float]:
        tags = set(tag_candidates)
        for item in line_items:
            if item.tag in tags:
                return {
                    (obs.end, obs.fiscal_period): obs.value for obs in item.observations
                }
        return {}

    @staticmethod
    def _fiscal_years(
        line_items: list[FinancialLineItem],
    ) -> dict[tuple[str, str], int]:
        years: dict[tuple[str, str], int] = {}
        for item in line_items:
            for obs in item.observations:
                if obs.fiscal_year is not None:
                    years[(obs.end, obs.fiscal_period)] = obs.fiscal_year
        return years

    @staticmethod
    def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator

    def _yoy_growth(
        self,
        series: dict[tuple[str, str], float],
        end: str,
        fp: str,
        current: float | None,
    ) -> float | None:
        if current is None:
            return None

        try:
            end_year = int(end[:4])
            prior_end = f"{end_year - 1}{end[4:]}"
        except ValueError:
            return None

        prior = series.get((prior_end, fp))
        if prior in (None, 0):
            return None
        return (current - prior) / abs(prior)
