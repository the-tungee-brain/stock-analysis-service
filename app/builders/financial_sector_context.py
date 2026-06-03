from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics


class CompanyArchetype(str, Enum):
    GENERIC = "generic"
    HYPERGROWTH_TECH = "hypergrowth_tech"
    BANK = "bank"
    UTILITY = "utility"
    REIT = "reit"
    BIOTECH = "biotech"
    CAPITAL_INTENSIVE = "capital_intensive"
    MATURE_DIVIDEND = "mature_dividend"


@dataclass(frozen=True)
class FinancialCompanyContext:
    symbol: str
    sector: str | None = None
    industry: str | None = None

    @property
    def archetype(self) -> CompanyArchetype:
        return classify_archetype(self.sector, self.industry)

    @property
    def allows_dividend_commentary(self) -> bool:
        return self.archetype in {
            CompanyArchetype.UTILITY,
            CompanyArchetype.REIT,
            CompanyArchetype.MATURE_DIVIDEND,
        }

    @property
    def business_context_label(self) -> str:
        parts: list[str] = []
        if self.sector:
            parts.append(self.sector.strip())
        if self.industry and (not self.sector or self.industry.strip().lower() not in self.sector.lower()):
            parts.append(self.industry.strip())
        archetype_labels = {
            CompanyArchetype.HYPERGROWTH_TECH: "Hypergrowth software",
            CompanyArchetype.BANK: "Financial institution",
            CompanyArchetype.UTILITY: "Regulated utility",
            CompanyArchetype.REIT: "Real estate / REIT",
            CompanyArchetype.BIOTECH: "Biotech / pharma",
            CompanyArchetype.CAPITAL_INTENSIVE: "Capital-intensive operator",
            CompanyArchetype.MATURE_DIVIDEND: "Mature cash-generative",
        }
        parts.append(archetype_labels.get(self.archetype, "Operating company"))
        return " · ".join(parts[:2])


def classify_archetype(sector: str | None, industry: str | None) -> CompanyArchetype:
    blob = f"{sector or ''} {industry or ''}".lower()

    if any(token in blob for token in ("reit", "real estate investment", "equity reit")):
        return CompanyArchetype.REIT
    if any(
        token in blob
        for token in (
            "bank",
            "banks",
            "insurance",
            "capital markets",
            "asset management",
            "financial services",
        )
    ):
        return CompanyArchetype.BANK
    if any(token in blob for token in ("utility", "utilities", "electric", "water utility")):
        return CompanyArchetype.UTILITY
    if any(
        token in blob
        for token in (
            "biotechnology",
            "biotech",
            "pharmaceutical",
            "drug manufacturer",
            "life sciences",
        )
    ):
        return CompanyArchetype.BIOTECH
    if any(
        token in blob
        for token in (
            "software",
            "internet",
            "semiconductor",
            "technology",
            "interactive media",
            "ai ",
            "artificial intelligence",
        )
    ):
        return CompanyArchetype.HYPERGROWTH_TECH
    if any(
        token in blob
        for token in (
            "oil & gas",
            "oil and gas",
            "mining",
            "steel",
            "airline",
            "automotive",
            "industrial",
            "manufacturing",
            "transportation",
        )
    ):
        return CompanyArchetype.CAPITAL_INTENSIVE
    if any(token in blob for token in ("consumer staples", "tobacco", "telecom")):
        return CompanyArchetype.MATURE_DIVIDEND
    return CompanyArchetype.GENERIC


def derive_profile(
    c: CanonicalFinancialMetrics,
    ctx: FinancialCompanyContext,
    category: "_CategorySignals",
) -> str:
    archetype = ctx.archetype
    rg = c.revenue_growth_yoy
    nm = c.net_margin_pct
    gm = c.gross_margin_pct
    de = c.debt_to_equity
    fcf = c.free_cash_flow_latest

    high_growth = rg is not None and rg > 50
    growth_risk = (nm is not None and nm < 0) or (fcf is not None and fcf < 0)
    commodity_like = gm is not None and gm < 30

    if archetype == CompanyArchetype.BANK:
        if category.balance_sheet >= 0.45 and category.profitability >= 0.35:
            return "Financially Strong"
        if category.profitability < 0 or category.balance_sheet < -0.2:
            return "Leveraged Turnaround"
        return "Profitable Compounder"

    if archetype == CompanyArchetype.REIT:
        if category.cash_flow >= 0.45 and de is not None and de > 1.0:
            return "Capital-Intensive Operator"
        if category.cash_flow >= 0.35:
            return "Cash-Generating Value"
        return "Leveraged Turnaround"

    if archetype == CompanyArchetype.UTILITY:
        if category.cash_flow >= 0.4 and category.balance_sheet >= 0.1:
            return "Mature Stable Business"
        if de is not None and de > 2.5:
            return "Capital-Intensive Operator"
        return "Cash-Generating Value"

    if archetype == CompanyArchetype.BIOTECH:
        if fcf is not None and fcf < 0 and (nm is None or nm < 0):
            return "Speculative Growth"
        if category.profitability >= 0.5:
            return "Profitable Compounder"
        return "High Growth / High Risk"

    if archetype == CompanyArchetype.HYPERGROWTH_TECH:
        if high_growth and growth_risk:
            return "High Growth / High Risk"
        if high_growth and category.profitability >= 0.45:
            return "Profitable Compounder"
        if high_growth:
            return "Speculative Growth"
        if category.profitability >= 0.6 and category.cash_flow >= 0.5:
            return "Financially Strong"
        return "Profitable Compounder"

    if archetype == CompanyArchetype.CAPITAL_INTENSIVE:
        if commodity_like and de is not None and de > 1.5:
            return "Capital-Intensive Operator"
        if category.cash_flow >= 0.45 and category.profitability >= 0.25:
            return "Cash-Generating Value"
        if growth_risk:
            return "Leveraged Turnaround"
        return "Mature Stable Business"

    if archetype == CompanyArchetype.MATURE_DIVIDEND:
        if category.cash_flow >= 0.45 and category.profitability >= 0.35:
            return "Cash-Generating Value"
        return "Mature Stable Business"

    if high_growth and growth_risk:
        return "High Growth / High Risk"
    if high_growth and (nm is None or nm < 8):
        return "Speculative Growth"
    if category.profitability >= 0.6 and category.growth >= 0.4 and category.cash_flow >= 0.5:
        return "Profitable Compounder"
    if category.profitability >= 0.6 and category.cash_flow >= 0.5 and category.balance_sheet >= 0.2:
        if rg is None or abs(rg) < 12:
            return "Financially Strong"
    if (
        category.profitability >= 0.5
        and category.cash_flow >= 0.4
        and (rg is None or abs(rg) < 6)
    ):
        return "Mature Stable Business"
    if category.cash_flow >= 0.5 and (rg is None or rg < 15):
        return "Cash-Generating Value"
    if (rg is not None and rg < 0) and category.profitability < 0:
        return "Leveraged Turnaround"
    if de is not None and de > 2 and commodity_like:
        return "Capital-Intensive Operator"

    composite = category.weighted_signal()
    if composite >= 0.55:
        return "Financially Strong"
    if composite >= 0.3:
        return "Profitable Compounder"
    if composite <= -0.25 and high_growth:
        return "High Growth / High Risk"
    if composite <= -0.25:
        return "Leveraged Turnaround"
    if high_growth:
        return "Speculative Growth"
    return "Profitable Compounder"


@dataclass(frozen=True)
class _CategorySignals:
    growth: float
    profitability: float
    balance_sheet: float
    cash_flow: float

    def weighted_signal(self) -> float:
        return (
            0.30 * self.growth
            + 0.30 * self.profitability
            + 0.25 * self.cash_flow
            + 0.15 * self.balance_sheet
        )


def archetype_observations(
    c: CanonicalFinancialMetrics,
    ctx: FinancialCompanyContext,
) -> list[tuple[str, str, float, int]]:
    """Returns list of (text, kind, materiality, score_delta)."""
    archetype = ctx.archetype
    rows: list[tuple[str, str, float, int]] = []

    if archetype == CompanyArchetype.HYPERGROWTH_TECH:
        rev = c.format_revenue_growth()
        gross = c.format_gross_margin()
        if rev and c.revenue_growth_yoy and c.revenue_growth_yoy > 30:
            rows.append(
                (
                    f"Hypergrowth profile ({rev} revenue) with {gross or 'high'} gross margins — scaling economics matter more than near-term profits.",
                    "strength",
                    82,
                    10,
                )
            )
        if c.free_cash_flow_latest is not None and c.free_cash_flow_latest < 0:
            rows.append(
                (
                    "Cash burn funds product and capacity expansion — execution and adoption risk stay elevated.",
                    "risk",
                    78,
                    -10,
                )
            )

    elif archetype == CompanyArchetype.BANK:
        roe = c.return_on_equity_pct
        de = c.format_debt_equity()
        net = c.format_net_margin()
        if roe is not None and roe >= 10:
            rows.append(
                (
                    f"Return on equity of {roe:.0f}% supports earnings power relative to book capital.",
                    "strength",
                    80,
                    8,
                )
            )
        if de:
            rows.append(
                (
                    f"Leverage of {de} should be read against regulatory capital and loan-book quality, not industrial peers.",
                    "risk" if (c.debt_to_equity or 0) > 1.2 else "strength",
                    70,
                    -6 if (c.debt_to_equity or 0) > 1.2 else 4,
                )
            )
        if net:
            rows.append(
                (
                    f"Net margin of {net} reflects spread and credit discipline in the banking model.",
                    "strength" if (c.net_margin_pct or 0) > 8 else "risk",
                    65,
                    5,
                )
            )

    elif archetype == CompanyArchetype.UTILITY:
        fcf = c.format_free_cash_flow()
        de = c.format_debt_equity()
        if fcf:
            rows.append(
                (
                    f"Regulated cash generation ({fcf}) anchors the investment case more than revenue growth.",
                    "strength" if (c.free_cash_flow_latest or 0) > 0 else "risk",
                    76,
                    8,
                )
            )
        if de and (c.debt_to_equity or 0) > 1.0:
            rows.append(
                (
                    f"Utility-style leverage ({de}) is typical — focus on rate base growth and allowed returns.",
                    "risk",
                    55,
                    -3,
                )
            )

    elif archetype == CompanyArchetype.REIT:
        fcf = c.format_free_cash_flow()
        de = c.format_debt_equity()
        if fcf:
            rows.append(
                (
                    f"Funds-from-operations proxy ({fcf}) and payout capacity drive REIT quality more than GAAP net income.",
                    "strength" if (c.free_cash_flow_latest or 0) > 0 else "risk",
                    80,
                    7,
                )
            )
        if de:
            rows.append(
                (
                    f"Property leverage at {de} is structural — refinancing and occupancy trends matter.",
                    "risk" if (c.debt_to_equity or 0) > 2 else "strength",
                    72,
                    -8 if (c.debt_to_equity or 0) > 2 else 3,
                )
            )

    elif archetype == CompanyArchetype.BIOTECH:
        fcf = c.format_free_cash_flow()
        if fcf and (c.free_cash_flow_latest or 0) < 0:
            rows.append(
                (
                    f"Negative free cash flow ({fcf}) highlights funding runway and pipeline execution risk.",
                    "risk",
                    90,
                    -14,
                )
            )
        if c.net_margin_pct is not None and c.net_margin_pct < 0:
            rows.append(
                (
                    "Earnings losses are common pre-commercialization — cash and balance-sheet runway matter most.",
                    "risk",
                    85,
                    -12,
                )
            )

    elif archetype == CompanyArchetype.CAPITAL_INTENSIVE:
        gross = c.format_gross_margin()
        de = c.format_debt_equity()
        if gross and (c.gross_margin_pct or 100) < 35:
            rows.append(
                (
                    f"Thin gross margins ({gross}) reflect asset-heavy, capital-intensive operations.",
                    "risk",
                    70,
                    -5,
                )
            )
        if de and (c.debt_to_equity or 0) > 1.5:
            rows.append(
                (
                    f"High capital intensity shows up in leverage of {de} and ongoing reinvestment needs.",
                    "risk",
                    75,
                    -8,
                )
            )

    return rows


def capitalize_sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    return stripped[0].upper() + stripped[1:]


def build_verdict_phrase(
    positive: list[str],
    negative: list[str],
    ctx: FinancialCompanyContext,
) -> str:
    if positive and negative:
        return capitalize_sentence(
            f"{_join(positive)} are offset by {_join(negative)}."
        )
    if positive:
        return capitalize_sentence(
            f"{_join(positive)} support {_archetype_quality_phrase(ctx)}."
        )
    if negative:
        return capitalize_sentence(
            f"{_join(negative)} weigh on {_archetype_quality_phrase(ctx)}."
        )
    return capitalize_sentence(
        f"Mixed financial signals across {_archetype_quality_phrase(ctx)}."
    )


def _join(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return f"{phrases[0]} and {phrases[1]}"


def _archetype_quality_phrase(ctx: FinancialCompanyContext) -> str:
    mapping = {
        CompanyArchetype.BANK: "bank financial profile",
        CompanyArchetype.UTILITY: "regulated utility profile",
        CompanyArchetype.REIT: "REIT-style profile",
        CompanyArchetype.BIOTECH: "biotech funding profile",
        CompanyArchetype.HYPERGROWTH_TECH: "hypergrowth software profile",
        CompanyArchetype.CAPITAL_INTENSIVE: "capital-intensive profile",
        CompanyArchetype.MATURE_DIVIDEND: "mature cash-generative profile",
    }
    return mapping.get(ctx.archetype, "the overall financial profile")
