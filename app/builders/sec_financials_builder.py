from typing import Any, Literal

from app.models.sec_research_models import (
    FinancialLineItem,
    FinancialObservation,
    SecFinancialsResponse,
)

ANNUAL_FORMS = {"10-K", "10-K/A"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}

INCOME_STATEMENT_SPECS: list[tuple[str, list[str]]] = [
    ("revenue", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ]),
    ("cost_of_revenue", [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
    ]),
    ("gross_profit", ["GrossProfit"]),
    ("operating_income", ["OperatingIncomeLoss"]),
    ("net_income", ["NetIncomeLoss"]),
    ("research_and_development", ["ResearchAndDevelopmentExpense"]),
    ("eps_diluted", ["EarningsPerShareDiluted"]),
    ("eps_basic", ["EarningsPerShareBasic"]),
]

BALANCE_SHEET_SPECS: list[tuple[str, list[str]]] = [
    ("assets", ["Assets"]),
    ("liabilities", ["Liabilities"]),
    ("stockholders_equity", ["StockholdersEquity"]),
    ("cash_and_equivalents", ["CashAndCashEquivalentsAtCarryingValue"]),
    ("long_term_debt", ["LongTermDebt", "LongTermDebtNoncurrent"]),
    ("current_assets", ["AssetsCurrent"]),
    ("current_liabilities", ["LiabilitiesCurrent"]),
]

CASH_FLOW_SPECS: list[tuple[str, list[str]]] = [
    ("operating_cash_flow", ["NetCashProvidedByUsedInOperatingActivities"]),
    ("capital_expenditures", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    ("investing_cash_flow", ["NetCashProvidedByUsedInInvestingActivities"]),
    ("financing_cash_flow", ["NetCashProvidedByUsedInFinancingActivities"]),
]


class SecFinancialsBuilder:
    def build(
        self,
        *,
        symbol: str,
        cik: str,
        entity_name: str,
        company_facts: dict[str, Any],
        period: Literal["annual", "quarterly"],
        limit: int = 12,
    ) -> SecFinancialsResponse:
        usgaap = company_facts.get("facts", {}).get("us-gaap", {})

        return SecFinancialsResponse(
            symbol=symbol,
            cik=cik,
            entity_name=entity_name,
            period=period,
            currency="USD",
            income_statement=self._build_section(
                usgaap=usgaap,
                specs=INCOME_STATEMENT_SPECS,
                period=period,
                limit=limit,
            ),
            balance_sheet=self._build_section(
                usgaap=usgaap,
                specs=BALANCE_SHEET_SPECS,
                period=period,
                limit=limit,
            ),
            cash_flow=self._build_section(
                usgaap=usgaap,
                specs=CASH_FLOW_SPECS,
                period=period,
                limit=limit,
            ),
        )

    def _build_section(
        self,
        *,
        usgaap: dict[str, Any],
        specs: list[tuple[str, list[str]]],
        period: Literal["annual", "quarterly"],
        limit: int,
    ) -> list[FinancialLineItem]:
        items: list[FinancialLineItem] = []
        for _key, tag_candidates in specs:
            matched_tag, fact = self._pick_fact_with_tag(
                usgaap=usgaap, tag_candidates=tag_candidates
            )
            if not fact or not matched_tag:
                continue

            unit, observations = self._extract_observations(
                fact=fact,
                period=period,
                limit=limit,
            )
            if not observations:
                continue

            items.append(
                FinancialLineItem(
                    tag=matched_tag,
                    label=fact.get("label") or matched_tag,
                    unit=unit,
                    observations=observations,
                )
            )
        return items

    @staticmethod
    def _pick_fact_with_tag(
        usgaap: dict[str, Any], tag_candidates: list[str]
    ) -> tuple[str | None, dict[str, Any] | None]:
        for tag in tag_candidates:
            if tag in usgaap:
                return tag, usgaap[tag]
        return None, None

    @staticmethod
    def value_at_period(
        line_items: list[FinancialLineItem],
        tag_candidates: list[str],
        *,
        end: str,
        fiscal_period: str,
    ) -> float | None:
        tags = set(tag_candidates)
        for item in line_items:
            if item.tag not in tags:
                continue
            for obs in item.observations:
                if obs.end == end and obs.fiscal_period == fiscal_period:
                    return obs.value
        return None

    @staticmethod
    def _pick_fact(
        usgaap: dict[str, Any], tag_candidates: list[str]
    ) -> dict[str, Any] | None:
        _, fact = SecFinancialsBuilder._pick_fact_with_tag(
            usgaap=usgaap, tag_candidates=tag_candidates
        )
        return fact

    def _extract_observations(
        self,
        *,
        fact: dict[str, Any],
        period: Literal["annual", "quarterly"],
        limit: int,
    ) -> tuple[str, list[FinancialObservation]]:
        preferred_units = ["USD", "USD/shares", "shares", "pure"]
        unit_key = next((u for u in preferred_units if u in fact.get("units", {})), None)
        if unit_key is None and fact.get("units"):
            unit_key = next(iter(fact["units"]))
        if unit_key is None:
            return "USD", []

        raw_obs = fact["units"][unit_key]
        deduped = self._dedupe_observations(raw_obs=raw_obs, period=period)
        deduped.sort(key=lambda o: o["end"], reverse=True)
        deduped = deduped[:limit]

        observations = [
            FinancialObservation(
                end=o["end"],
                start=o.get("start"),
                value=float(o["val"]),
                fiscal_year=o.get("fy"),
                fiscal_period=o.get("fp") or "",
                form=o.get("form") or "",
                filed=o.get("filed") or "",
            )
            for o in deduped
        ]
        return unit_key, observations

    def _dedupe_observations(
        self,
        *,
        raw_obs: list[dict[str, Any]],
        period: Literal["annual", "quarterly"],
    ) -> list[dict[str, Any]]:
        allowed_forms = ANNUAL_FORMS if period == "annual" else QUARTERLY_FORMS
        allowed_fps = {"FY"} if period == "annual" else {"Q1", "Q2", "Q3", "Q4"}

        best_by_period: dict[tuple[str, str], dict[str, Any]] = {}
        for obs in raw_obs:
            form = obs.get("form") or ""
            fp = obs.get("fp") or ""
            end = obs.get("end")
            if not end or form not in allowed_forms or fp not in allowed_fps:
                continue

            key = (end, fp)
            current = best_by_period.get(key)
            if current is None or (obs.get("filed") or "") >= (current.get("filed") or ""):
                best_by_period[key] = obs

        return list(best_by_period.values())

    @staticmethod
    def latest_value_by_end(
        line_items: list[FinancialLineItem], tag_prefix: str
    ) -> dict[str, float]:
        for item in line_items:
            if item.tag.startswith(tag_prefix) or tag_prefix in item.tag:
                return {obs.end: obs.value for obs in item.observations}
        return {}

    @staticmethod
    def values_by_period(
        line_items: list[FinancialLineItem], tag_candidates: list[str]
    ) -> dict[tuple[str, str], float]:
        tags = set(tag_candidates)
        for item in line_items:
            if item.tag in tags:
                return {(obs.end, obs.fiscal_period): obs.value for obs in item.observations}
        return {}
