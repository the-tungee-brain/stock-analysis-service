from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics
from app.builders.financial_metrics_validation import validate_overview_result
from app.builders.financial_score_percentiles import rank_label_for_score
from app.builders.financial_sector_context import (
    FinancialCompanyContext,
    _CategorySignals,
    archetype_observations,
    build_verdict_phrase,
    derive_profile,
)
from app.models.company_research_models import (
    FinancialCategoryScore,
    FinancialScoreBreakdown,
)

StrengthRating = Literal["strong", "solid", "mixed", "weak"]
ObservationKind = Literal["strength", "risk"]

GROWTH_WEIGHT = 0.30
PROFITABILITY_WEIGHT = 0.30
CASH_FLOW_WEIGHT = 0.25
BALANCE_SHEET_WEIGHT = 0.15

@dataclass(frozen=True)
class _CategoryProfile:
    growth: float = 0.0
    profitability: float = 0.0
    balance_sheet: float = 0.0
    cash_flow: float = 0.0


@dataclass(frozen=True)
class FinancialMetricsSnapshot:
    revenue_growth_yoy: float | None = None
    gross_margin_pct: float | None = None
    net_margin_pct: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    free_cash_flow_latest: float | None = None
    free_cash_flow_yoy_pct: float | None = None
    return_on_equity_pct: float | None = None
    payout_ratio_pct: float | None = None
    fcf_dividend_coverage: float | None = None


@dataclass(frozen=True)
class ScoredObservation:
    text: str
    kind: ObservationKind
    materiality: float
    score_delta: int


@dataclass(frozen=True)
class FinancialOverviewResult:
    profile: str
    score: int
    financial_verdict: str
    score_explanation: str
    business_context: str
    score_breakdown: FinancialScoreBreakdown
    rating: StrengthRating
    headline: str
    strengths: list[str]
    risks: list[str]
    highlights: list[str]


class FinancialOverviewGenerator:
    MAX_STRENGTHS = 3
    MAX_RISKS = 3
    MAX_HIGHLIGHTS = 6

    _BANNED_PHRASES = (
        "investors should monitor",
        "the company faces challenges",
        "the future depends on execution",
        "review the statement tables",
        "compare leverage and growth",
        "risk tolerance",
    )

    def generate(
        self,
        symbol: str,
        canonical: CanonicalFinancialMetrics,
        *,
        sector: str | None = None,
        industry: str | None = None,
    ) -> FinancialOverviewResult:
        ctx = FinancialCompanyContext(
            symbol=symbol,
            sector=sector,
            industry=industry,
        )
        observations = self._build_observations(canonical, ctx)

        strengths = self._top_observations(
            [o for o in observations if o.kind == "strength"],
            self.MAX_STRENGTHS,
        )
        risks = self._top_observations(
            [o for o in observations if o.kind == "risk"],
            self.MAX_RISKS,
        )

        breakdown = self._category_scores_0_100(canonical)
        score = self._weighted_overall_score(breakdown)
        signals = self._category_signals(canonical)
        profile = derive_profile(canonical, ctx, signals)
        rating = self._rating_from_score(score)
        financial_verdict = self._build_financial_verdict(
            canonical,
            breakdown,
            ctx,
            strengths,
            risks,
        )
        headline = self._headline(symbol, canonical)
        highlights = self._build_highlights(canonical, ctx)

        result = FinancialOverviewResult(
            profile=profile,
            score=score,
            financial_verdict=financial_verdict,
            score_explanation=financial_verdict,
            business_context=ctx.business_context_label,
            score_breakdown=breakdown,
            rating=rating,
            headline=headline,
            strengths=strengths,
            risks=risks,
            highlights=highlights[: self.MAX_HIGHLIGHTS],
        )
        validate_overview_result(result, canonical, ctx)
        return result

    @staticmethod
    def _is_high_growth_cash_story(c: CanonicalFinancialMetrics) -> bool:
        high_growth = c.revenue_growth_yoy is not None and c.revenue_growth_yoy > 50
        cash_pressure = c.free_cash_flow_latest is not None and c.free_cash_flow_latest < 0
        return high_growth and cash_pressure

    def _build_observations(
        self,
        c: CanonicalFinancialMetrics,
        ctx: FinancialCompanyContext,
    ) -> list[ScoredObservation]:
        out: list[ScoredObservation] = []
        for text, kind, materiality, delta in archetype_observations(c, ctx):
            out.append(
                ScoredObservation(
                    text=text,
                    kind=kind,  # type: ignore[arg-type]
                    materiality=materiality,
                    score_delta=delta,
                )
            )
        out.extend(self._revenue_growth_observations(c))
        out.extend(self._gross_margin_observations(c))
        out.extend(self._net_margin_observations(c))
        out.extend(self._debt_equity_observations(c))
        out.extend(self._current_ratio_observations(c))
        out.extend(self._free_cash_flow_observations(c))
        out.extend(self._roe_observations(c))
        if ctx.allows_dividend_commentary:
            out.extend(self._payout_observations(c))
        filtered = [o for o in out if not self._is_banned(o.text)]
        if not ctx.allows_dividend_commentary:
            filtered = [
                o
                for o in filtered
                if "payout" not in o.text.lower() and "dividend" not in o.text.lower()
            ]
        return filtered

    def _revenue_growth_observations(
        self, c: CanonicalFinancialMetrics
    ) -> list[ScoredObservation]:
        growth = c.revenue_growth_yoy
        display = c.format_revenue_growth()
        if growth is None or display is None:
            return []

        if growth > 100:
            return [
                ScoredObservation(
                    text=(
                        f"Revenue grew {display} year over year — exceptional growth "
                        "that points to rapid commercial adoption."
                    ),
                    kind="strength",
                    materiality=88,
                    score_delta=18,
                )
            ]
        if growth >= 20:
            return [
                ScoredObservation(
                    text=f"Revenue growth remains strong at {display} year over year.",
                    kind="strength",
                    materiality=72,
                    score_delta=12,
                )
            ]
        if growth >= 0:
            return [
                ScoredObservation(
                    text=f"Revenue growth is modest at {display} year over year.",
                    kind="strength",
                    materiality=35,
                    score_delta=4,
                )
            ]
        return [
            ScoredObservation(
                text=f"Revenue is contracting {display} year over year.",
                kind="risk",
                materiality=80,
                score_delta=-14,
            )
        ]

    def _gross_margin_observations(
        self, c: CanonicalFinancialMetrics
    ) -> list[ScoredObservation]:
        margin = c.gross_margin_pct
        display = c.format_gross_margin()
        if margin is None or display is None:
            return []

        if margin > 70:
            return [
                ScoredObservation(
                    text=(
                        f"Gross margin of {display} reflects premium pricing power "
                        "and software-like unit economics."
                    ),
                    kind="strength",
                    materiality=58,
                    score_delta=8,
                )
            ]
        if margin >= 30:
            return [
                ScoredObservation(
                    text=f"Gross margin of {display} indicates healthy economics at the product level.",
                    kind="strength",
                    materiality=40,
                    score_delta=5,
                )
            ]
        return [
            ScoredObservation(
                text=(
                    f"Gross margin of {display} is thin — typical of commodity "
                    "or capital-intensive businesses."
                ),
                kind="risk",
                materiality=55,
                score_delta=-6,
            )
        ]

    def _net_margin_observations(
        self, c: CanonicalFinancialMetrics
    ) -> list[ScoredObservation]:
        margin = c.net_margin_pct
        display = c.format_net_margin()
        if margin is None or display is None:
            return []

        if margin > 15:
            return [
                ScoredObservation(
                    text=f"Net margin of {display} signals strong bottom-line profitability.",
                    kind="strength",
                    materiality=85,
                    score_delta=14,
                )
            ]
        if margin >= 0:
            return [
                ScoredObservation(
                    text=f"Net margin of {display} — profitable, but with limited profit per revenue dollar.",
                    kind="strength",
                    materiality=45,
                    score_delta=5,
                )
            ]
        if margin >= -20:
            return [
                ScoredObservation(
                    text=f"Net margin of {display} — the business is unprofitable on a net basis.",
                    kind="risk",
                    materiality=92,
                    score_delta=-18,
                )
            ]
        return [
            ScoredObservation(
                text=(
                    f"Net margin of {display} — deep losses absorb a large share of each revenue dollar."
                ),
                kind="risk",
                materiality=98,
                score_delta=-24,
            )
        ]

    def _debt_equity_observations(
        self, c: CanonicalFinancialMetrics
    ) -> list[ScoredObservation]:
        ratio = c.debt_to_equity
        ratio_label = c.format_debt_equity()
        if ratio is None or ratio_label is None:
            return []

        if ratio < 0.5:
            materiality = 50
            if self._is_high_growth_cash_story(c):
                materiality = 18
            return [
                ScoredObservation(
                    text=f"Debt/equity of {ratio_label} reflects conservative leverage.",
                    kind="strength",
                    materiality=materiality,
                    score_delta=8,
                )
            ]
        if ratio <= 2:
            return [
                ScoredObservation(
                    text=f"Debt/equity of {ratio_label} is within a manageable range.",
                    kind="strength",
                    materiality=28,
                    score_delta=3,
                )
            ]
        if ratio <= 5:
            return [
                ScoredObservation(
                    text=f"Debt/equity of {ratio_label} — leverage is elevated relative to equity.",
                    kind="risk",
                    materiality=78,
                    score_delta=-12,
                )
            ]
        return [
            ScoredObservation(
                text=(
                    f"Debt/equity of {ratio_label} — balance-sheet risk is significant "
                    "relative to the equity base."
                ),
                kind="risk",
                materiality=96,
                score_delta=-20,
            )
        ]

    def _current_ratio_observations(
        self, c: CanonicalFinancialMetrics
    ) -> list[ScoredObservation]:
        ratio = c.current_ratio
        display = c.format_current_ratio()
        if ratio is None or display is None:
            return []

        if ratio > 2:
            return [
                ScoredObservation(
                    text=f"Current ratio of {display} indicates strong near-term liquidity.",
                    kind="strength",
                    materiality=42,
                    score_delta=6,
                )
            ]
        if ratio >= 1:
            return [
                ScoredObservation(
                    text=f"Current ratio of {display} suggests adequate short-term liquidity.",
                    kind="strength",
                    materiality=22,
                    score_delta=2,
                )
            ]
        return [
            ScoredObservation(
                text=f"Current ratio of {display} — potential pressure meeting near-term obligations.",
                kind="risk",
                materiality=70,
                score_delta=-10,
            )
        ]

    def _free_cash_flow_observations(
        self, c: CanonicalFinancialMetrics
    ) -> list[ScoredObservation]:
        fcf = c.free_cash_flow_latest
        amount = c.format_free_cash_flow()
        if fcf is None or amount is None:
            return []

        trend = c.free_cash_flow_yoy_pct

        if fcf > 0:
            if trend is not None and trend > 10:
                return [
                    ScoredObservation(
                        text=(
                            f"Free cash flow of {amount} is positive and grew {trend:.0f}% year over year — "
                            "strong operational flexibility."
                        ),
                        kind="strength",
                        materiality=76,
                        score_delta=12,
                    )
                ]
            if fcf >= 1_000_000_000 or (trend is not None and trend >= 0):
                return [
                    ScoredObservation(
                        text=f"Free cash flow of {amount} is positive — supports reinvestment, debt paydown, or distributions.",
                        kind="strength",
                        materiality=68,
                        score_delta=9,
                    )
                ]
            return [
                ScoredObservation(
                    text=f"Free cash flow of {amount} is positive but modest — limited balance-sheet flexibility.",
                    kind="strength",
                    materiality=38,
                    score_delta=4,
                )
            ]

        if trend is not None and trend < -10:
            return [
                ScoredObservation(
                    text=(
                        f"Free cash flow of {amount} is negative and deteriorated {abs(trend):.0f}% year over year — "
                        "financing or a turnaround in operations is required."
                    ),
                    kind="risk",
                    materiality=90,
                    score_delta=-16,
                )
            ]
        return [
            ScoredObservation(
                text=f"Free cash flow of {amount} is negative — cash is consumed after capex.",
                kind="risk",
                materiality=82,
                score_delta=-12,
            )
        ]

    def _roe_observations(self, c: CanonicalFinancialMetrics) -> list[ScoredObservation]:
        roe = c.return_on_equity_pct
        if roe is None:
            return []

        if roe >= 20:
            return [
                ScoredObservation(
                    text=f"Return on equity of {roe:.0f}% shows efficient use of shareholder capital.",
                    kind="strength",
                    materiality=48,
                    score_delta=6,
                )
            ]
        if roe < 5:
            return [
                ScoredObservation(
                    text=f"Return on equity of {roe:.0f}% is weak relative to typical public companies.",
                    kind="risk",
                    materiality=40,
                    score_delta=-4,
                )
            ]
        return []

    def _payout_observations(self, c: CanonicalFinancialMetrics) -> list[ScoredObservation]:
        payout = c.payout_ratio_pct
        coverage = c.fcf_dividend_coverage
        if payout is None and coverage is None:
            return []

        out: list[ScoredObservation] = []
        if payout is not None and payout > 100:
            out.append(
                ScoredObservation(
                    text=(
                        f"Payout ratio of {payout:.0f}% exceeds earnings — dividends rely on "
                        "balance-sheet cash or non-operating sources."
                    ),
                    kind="risk",
                    materiality=62,
                    score_delta=-8,
                )
            )
        elif payout is not None and payout >= 80:
            out.append(
                ScoredObservation(
                    text=f"Payout ratio of {payout:.0f}% leaves little earnings cushion for downturns.",
                    kind="risk",
                    materiality=45,
                    score_delta=-4,
                )
            )

        if coverage is not None:
            if coverage >= 1.5:
                out.append(
                    ScoredObservation(
                        text=f"Free cash flow covers dividends {coverage:.1f}x over the latest period.",
                        kind="strength",
                        materiality=44,
                        score_delta=5,
                    )
                )
            elif coverage < 1.0:
                out.append(
                    ScoredObservation(
                        text=(
                            f"Dividends exceed free cash flow ({coverage:.1f}x coverage) "
                            "in the latest period."
                        ),
                        kind="risk",
                        materiality=64,
                        score_delta=-9,
                    )
                )
        return out

    def _build_highlights(
        self,
        c: CanonicalFinancialMetrics,
        ctx: FinancialCompanyContext,
    ) -> list[str]:
        lines: list[str] = []
        rev = c.format_revenue_growth()
        if rev is not None:
            direction = "up" if (c.revenue_growth_yoy or 0) >= 0 else "down"
            lines.append(f"Revenue {direction} {rev} YoY.")
        gross = c.format_gross_margin()
        if gross is not None:
            lines.append(f"Gross margin {gross}.")
        net = c.format_net_margin()
        if net is not None:
            lines.append(f"Net margin {net}.")
        debt = c.format_debt_equity()
        if debt is not None:
            lines.append(f"Debt/equity {debt}.")
        current = c.format_current_ratio()
        if current is not None:
            lines.append(f"Current ratio {current}.")
        fcf = c.format_free_cash_flow()
        if fcf is not None:
            lines.append(f"Free cash flow {fcf} (latest period).")
        if c.return_on_equity_pct is not None:
            lines.append(f"ROE {c.return_on_equity_pct:.1f}%.")
        if ctx.allows_dividend_commentary and c.payout_ratio_pct is not None:
            lines.append(f"Payout ratio {c.payout_ratio_pct:.0f}%.")
        return lines

    def _category_signals(self, c: CanonicalFinancialMetrics) -> _CategorySignals:
        profile = self._category_profile(c)
        return _CategorySignals(
            growth=profile.growth,
            profitability=profile.profitability,
            balance_sheet=profile.balance_sheet,
            cash_flow=profile.cash_flow,
        )

    def _category_profile(self, c: CanonicalFinancialMetrics) -> _CategoryProfile:
        growth = 0.0
        if c.revenue_growth_yoy is not None:
            rg = c.revenue_growth_yoy
            if rg > 100:
                growth = 1.0
            elif rg >= 20:
                growth = 0.75
            elif rg >= 5:
                growth = 0.45
            elif rg >= 0:
                growth = 0.15
            else:
                growth = -0.7

        profitability = 0.0
        if c.net_margin_pct is not None:
            nm = c.net_margin_pct
            if nm > 15:
                profitability = 1.0
            elif nm >= 5:
                profitability = 0.6
            elif nm >= 0:
                profitability = 0.25
            elif nm >= -20:
                profitability = -0.6
            else:
                profitability = -1.0

        balance = 0.0
        if c.debt_to_equity is not None:
            de = c.debt_to_equity
            if de < 0.5:
                balance += 0.55
            elif de <= 2:
                balance += 0.15
            elif de <= 5:
                balance -= 0.55
            else:
                balance -= 1.0
        if c.current_ratio is not None:
            cr = c.current_ratio
            if cr > 2:
                balance += 0.35
            elif cr >= 1:
                balance += 0.1
            else:
                balance -= 0.45
        balance = max(-1.0, min(1.0, balance))

        cash_flow = 0.0
        if c.free_cash_flow_latest is not None:
            if c.free_cash_flow_latest > 0:
                cash_flow = 0.55
                if c.free_cash_flow_yoy_pct is not None and c.free_cash_flow_yoy_pct > 10:
                    cash_flow = 0.9
                elif c.free_cash_flow_yoy_pct is not None and c.free_cash_flow_yoy_pct < -10:
                    cash_flow = 0.25
            else:
                cash_flow = -0.75

        return _CategoryProfile(
            growth=growth,
            profitability=profitability,
            balance_sheet=balance,
            cash_flow=cash_flow,
        )

    @staticmethod
    def _normalized_to_score(value: float) -> int:
        return int(round(max(0, min(100, (value + 1) / 2 * 100))))

    def _category_scores_0_100(
        self, c: CanonicalFinancialMetrics
    ) -> FinancialScoreBreakdown:
        profile = self._category_profile(c)

        def growth_score() -> int:
            if c.revenue_growth_yoy is None:
                return 50
            return self._normalized_to_score(profile.growth)

        def profitability_score() -> int:
            if c.net_margin_pct is None:
                return 50
            return self._normalized_to_score(profile.profitability)

        def balance_score() -> int:
            if c.debt_to_equity is None and c.current_ratio is None:
                return 50
            return self._normalized_to_score(profile.balance_sheet)

        def cash_score() -> int:
            if c.free_cash_flow_latest is None:
                return 50
            return self._normalized_to_score(profile.cash_flow)

        def wrap(value: int) -> FinancialCategoryScore:
            return FinancialCategoryScore(
                score=value,
                rank_label=rank_label_for_score(value),
            )

        return FinancialScoreBreakdown(
            growth=wrap(growth_score()),
            profitability=wrap(profitability_score()),
            balance_sheet=wrap(balance_score()),
            cash_flow=wrap(cash_score()),
        )

    @staticmethod
    def _weighted_overall_score(breakdown: FinancialScoreBreakdown) -> int:
        composite = (
            GROWTH_WEIGHT * breakdown.growth.score
            + PROFITABILITY_WEIGHT * breakdown.profitability.score
            + CASH_FLOW_WEIGHT * breakdown.cash_flow.score
            + BALANCE_SHEET_WEIGHT * breakdown.balance_sheet.score
        )
        return int(round(max(0, min(100, composite))))

    def _build_financial_verdict(
        self,
        canonical: CanonicalFinancialMetrics,
        breakdown: FinancialScoreBreakdown,
        ctx: FinancialCompanyContext,
        strengths: list[str],
        risks: list[str],
    ) -> str:
        positive = self._score_drivers(breakdown, canonical, positive=True)
        negative = self._score_drivers(breakdown, canonical, positive=False)
        if positive or negative:
            return build_verdict_phrase(positive, negative, ctx)
        if strengths and risks:
            return build_verdict_phrase(
                [strengths[0].rstrip(".")],
                [risks[0].rstrip(".")],
                ctx,
            )
        if strengths:
            return strengths[0]
        if risks:
            return risks[0]
        return build_verdict_phrase([], [], ctx)

    def _score_drivers(
        self,
        breakdown: FinancialScoreBreakdown,
        canonical: CanonicalFinancialMetrics,
        *,
        positive: bool,
    ) -> list[str]:
        drivers: list[tuple[int, str]] = []

        def add(condition: bool, phrase: str, weight: int) -> None:
            if condition:
                drivers.append((weight, phrase))

        if positive:
            if breakdown.growth.score >= 72:
                rev = canonical.format_revenue_growth()
                add(
                    True,
                    f"exceptional revenue growth ({rev})" if rev else "exceptional revenue growth",
                    breakdown.growth.score,
                )
            elif breakdown.growth.score >= 58:
                add(True, "solid revenue growth", breakdown.growth.score)
            if breakdown.profitability.score >= 72:
                net = canonical.format_net_margin()
                add(
                    True,
                    f"strong margins ({net})" if net else "strong profitability",
                    breakdown.profitability.score,
                )
            elif breakdown.profitability.score >= 58:
                add(True, "healthy profitability", breakdown.profitability.score)
            if breakdown.cash_flow.score >= 72:
                fcf = canonical.format_free_cash_flow()
                add(
                    True,
                    f"strong free cash flow ({fcf})" if fcf else "strong free cash flow",
                    breakdown.cash_flow.score,
                )
            elif breakdown.cash_flow.score >= 58:
                add(True, "positive cash generation", breakdown.cash_flow.score)
            if breakdown.balance_sheet.score >= 68:
                add(
                    True,
                    "conservative leverage and liquidity",
                    breakdown.balance_sheet.score,
                )
        else:
            if breakdown.growth.score <= 38:
                add(True, "weak or contracting revenue", 100 - breakdown.growth.score)
            if breakdown.profitability.score <= 35:
                net = canonical.format_net_margin()
                add(
                    True,
                    f"deep losses ({net})" if net and (canonical.net_margin_pct or 0) < -20 else "weak profitability",
                    100 - breakdown.profitability.score,
                )
            elif breakdown.profitability.score <= 45:
                add(True, "thin profitability", 100 - breakdown.profitability.score)
            if breakdown.cash_flow.score <= 35:
                add(True, "negative free cash flow", 100 - breakdown.cash_flow.score)
            if breakdown.balance_sheet.score <= 38:
                debt = canonical.format_debt_equity()
                if debt and (canonical.debt_to_equity or 0) > 2:
                    add(
                        True,
                        f"elevated leverage ({debt})",
                        100 - breakdown.balance_sheet.score,
                    )
                elif (canonical.current_ratio or 2) < 1:
                    add(True, "liquidity pressure", 100 - breakdown.balance_sheet.score)
                else:
                    add(
                        True,
                        "balance-sheet strain",
                        100 - breakdown.balance_sheet.score,
                    )

        drivers.sort(key=lambda item: item[0], reverse=True)
        return [phrase for _, phrase in drivers[:2]]

    @staticmethod
    def _join_phrases(phrases: list[str]) -> str:
        if not phrases:
            return ""
        if len(phrases) == 1:
            return phrases[0].capitalize()
        return f"{phrases[0].capitalize()} and {phrases[1]}"

    def _headline(
        self,
        symbol: str,
        canonical: CanonicalFinancialMetrics,
    ) -> str:
        hook = self._metric_hook(canonical)
        ticker = symbol.upper()
        if hook:
            return f"{ticker} — {hook}"
        return f"{ticker} financial snapshot"

    @staticmethod
    def _metric_hook(canonical: CanonicalFinancialMetrics) -> str | None:
        hooks: list[tuple[float, str]] = []

        net = canonical.format_net_margin()
        if canonical.net_margin_pct is not None and canonical.net_margin_pct < -20 and net:
            hooks.append((abs(canonical.net_margin_pct), f"net margin {net}"))

        debt = canonical.format_debt_equity()
        if canonical.debt_to_equity is not None and canonical.debt_to_equity > 2 and debt:
            hooks.append((canonical.debt_to_equity * 10, f"debt/equity {debt}"))

        rev = canonical.format_revenue_growth()
        if canonical.revenue_growth_yoy is not None and canonical.revenue_growth_yoy > 50 and rev:
            hooks.append((canonical.revenue_growth_yoy, f"revenue {rev} YoY"))

        fcf = canonical.format_free_cash_flow()
        if canonical.free_cash_flow_latest is not None and canonical.free_cash_flow_latest < 0 and fcf:
            hooks.append(
                (
                    abs(canonical.free_cash_flow_latest) / 1_000_000_000,
                    f"FCF {fcf}",
                )
            )

        if not hooks:
            if canonical.net_margin_pct is not None and canonical.net_margin_pct > 15 and net:
                return f"net margin {net}"
            if rev:
                return f"revenue {rev} YoY"
            return None

        hooks.sort(key=lambda item: item[0], reverse=True)
        return hooks[0][1]

    @staticmethod
    def _top_observations(
        observations: list[ScoredObservation],
        limit: int,
    ) -> list[str]:
        ranked = sorted(observations, key=lambda o: o.materiality, reverse=True)
        return [o.text for o in ranked[:limit]]

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
    def _is_banned(text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in FinancialOverviewGenerator._BANNED_PHRASES)

    @staticmethod
    def normalize_pct(raw: float | None) -> float | None:
        if raw is None or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        if abs(value) <= 1.5:
            return value * 100
        return value

    @staticmethod
    def normalize_debt_to_equity(raw: float | None) -> float | None:
        if raw is None or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        if value > 10:
            return value / 100.0
        return value

    @staticmethod
    def _format_debt_equity(ratio: float) -> str:
        if ratio >= 10:
            return f"{ratio:.1f}x"
        return f"{ratio:.2f}x"

    @staticmethod
    def _fmt_compact_currency(value: float) -> str:
        sign = "-" if value < 0 else ""
        abs_val = abs(value)
        if abs_val >= 1_000_000_000_000:
            return f"{sign}${abs_val / 1_000_000_000_000:.1f}T"
        if abs_val >= 1_000_000_000:
            return f"{sign}${abs_val / 1_000_000_000:.1f}B"
        if abs_val >= 1_000_000:
            return f"{sign}${abs_val / 1_000_000:.1f}M"
        return f"{sign}${abs_val:,.0f}"
