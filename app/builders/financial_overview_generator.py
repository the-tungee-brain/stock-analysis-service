from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StrengthRating = Literal["strong", "solid", "mixed", "weak"]
ObservationKind = Literal["strength", "risk"]


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
    rating: StrengthRating
    score: int
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

    def generate(self, symbol: str, metrics: FinancialMetricsSnapshot) -> FinancialOverviewResult:
        observations = self._build_observations(metrics)
        highlights = self._build_highlights(metrics)

        strengths = self._top_observations(
            [o for o in observations if o.kind == "strength"],
            self.MAX_STRENGTHS,
        )
        risks = self._top_observations(
            [o for o in observations if o.kind == "risk"],
            self.MAX_RISKS,
        )

        score = 50 + sum(o.score_delta for o in observations)
        score = max(0, min(100, score))
        rating = self._rating_from_score(score)
        verdict = self._derive_verdict(metrics)
        headline = self._headline(symbol, verdict, metrics)

        return FinancialOverviewResult(
            rating=rating,
            score=score,
            headline=headline,
            strengths=strengths,
            risks=risks,
            highlights=highlights[: self.MAX_HIGHLIGHTS],
        )

    @staticmethod
    def _is_high_growth_cash_story(m: FinancialMetricsSnapshot) -> bool:
        high_growth = m.revenue_growth_yoy is not None and m.revenue_growth_yoy > 50
        cash_pressure = m.free_cash_flow_latest is not None and m.free_cash_flow_latest < 0
        return high_growth and cash_pressure

    def _build_observations(self, m: FinancialMetricsSnapshot) -> list[ScoredObservation]:
        out: list[ScoredObservation] = []
        out.extend(self._revenue_growth_observations(m))
        out.extend(self._gross_margin_observations(m))
        out.extend(self._net_margin_observations(m))
        out.extend(self._debt_equity_observations(m))
        out.extend(self._current_ratio_observations(m))
        out.extend(self._free_cash_flow_observations(m))
        out.extend(self._roe_observations(m))
        out.extend(self._payout_observations(m))
        return [o for o in out if not self._is_banned(o.text)]

    def _revenue_growth_observations(self, m: FinancialMetricsSnapshot) -> list[ScoredObservation]:
        growth = m.revenue_growth_yoy
        if growth is None:
            return []

        if growth > 100:
            return [
                ScoredObservation(
                    text=(
                        f"Revenue grew {growth:.0f}% year over year — exceptional growth "
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
                    text=f"Revenue growth remains strong at {growth:.0f}% year over year.",
                    kind="strength",
                    materiality=72,
                    score_delta=12,
                )
            ]
        if growth >= 0:
            return [
                ScoredObservation(
                    text=f"Revenue growth is modest at {growth:.0f}% year over year.",
                    kind="strength",
                    materiality=35,
                    score_delta=4,
                )
            ]
        return [
            ScoredObservation(
                text=f"Revenue is contracting {abs(growth):.0f}% year over year.",
                kind="risk",
                materiality=80,
                score_delta=-14,
            )
        ]

    def _gross_margin_observations(self, m: FinancialMetricsSnapshot) -> list[ScoredObservation]:
        margin = m.gross_margin_pct
        if margin is None:
            return []

        if margin > 70:
            return [
                ScoredObservation(
                    text=(
                        f"Gross margin of {margin:.0f}% reflects premium pricing power "
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
                    text=f"Gross margin of {margin:.0f}% indicates healthy economics at the product level.",
                    kind="strength",
                    materiality=40,
                    score_delta=5,
                )
            ]
        return [
            ScoredObservation(
                text=(
                    f"Gross margin of {margin:.0f}% is thin — typical of commodity "
                    "or capital-intensive businesses."
                ),
                kind="risk",
                materiality=55,
                score_delta=-6,
            )
        ]

    def _net_margin_observations(self, m: FinancialMetricsSnapshot) -> list[ScoredObservation]:
        margin = m.net_margin_pct
        if margin is None:
            return []

        if margin > 15:
            return [
                ScoredObservation(
                    text=f"Net margin of {margin:.0f}% signals strong bottom-line profitability.",
                    kind="strength",
                    materiality=85,
                    score_delta=14,
                )
            ]
        if margin >= 0:
            return [
                ScoredObservation(
                    text=f"Net margin of {margin:.0f}% — profitable, but with limited profit per revenue dollar.",
                    kind="strength",
                    materiality=45,
                    score_delta=5,
                )
            ]
        if margin >= -20:
            return [
                ScoredObservation(
                    text=f"Net margin of {margin:.0f}% — the business is unprofitable on a net basis.",
                    kind="risk",
                    materiality=92,
                    score_delta=-18,
                )
            ]
        return [
            ScoredObservation(
                text=(
                    f"Net margin of {margin:.0f}% — deep losses absorb a large share of each revenue dollar."
                ),
                kind="risk",
                materiality=98,
                score_delta=-24,
            )
        ]

    def _debt_equity_observations(self, m: FinancialMetricsSnapshot) -> list[ScoredObservation]:
        ratio = m.debt_to_equity
        if ratio is None:
            return []

        ratio_label = self._format_debt_equity(ratio)
        if ratio < 0.5:
            materiality = 50
            if self._is_high_growth_cash_story(m):
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

    def _current_ratio_observations(self, m: FinancialMetricsSnapshot) -> list[ScoredObservation]:
        ratio = m.current_ratio
        if ratio is None:
            return []

        if ratio > 2:
            return [
                ScoredObservation(
                    text=f"Current ratio of {ratio:.1f} indicates strong near-term liquidity.",
                    kind="strength",
                    materiality=42,
                    score_delta=6,
                )
            ]
        if ratio >= 1:
            return [
                ScoredObservation(
                    text=f"Current ratio of {ratio:.1f} suggests adequate short-term liquidity.",
                    kind="strength",
                    materiality=22,
                    score_delta=2,
                )
            ]
        return [
            ScoredObservation(
                text=f"Current ratio of {ratio:.1f} — potential pressure meeting near-term obligations.",
                kind="risk",
                materiality=70,
                score_delta=-10,
            )
        ]

    def _free_cash_flow_observations(self, m: FinancialMetricsSnapshot) -> list[ScoredObservation]:
        fcf = m.free_cash_flow_latest
        if fcf is None:
            return []

        trend = m.free_cash_flow_yoy_pct
        amount = self._fmt_compact_currency(fcf)

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

    def _roe_observations(self, m: FinancialMetricsSnapshot) -> list[ScoredObservation]:
        roe = m.return_on_equity_pct
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

    def _payout_observations(self, m: FinancialMetricsSnapshot) -> list[ScoredObservation]:
        payout = m.payout_ratio_pct
        coverage = m.fcf_dividend_coverage
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

    def _build_highlights(self, m: FinancialMetricsSnapshot) -> list[str]:
        lines: list[str] = []
        if m.revenue_growth_yoy is not None:
            direction = "up" if m.revenue_growth_yoy >= 0 else "down"
            lines.append(f"Revenue {direction} {abs(m.revenue_growth_yoy):.1f}% YoY.")
        if m.gross_margin_pct is not None:
            lines.append(f"Gross margin {m.gross_margin_pct:.1f}%.")
        if m.net_margin_pct is not None:
            lines.append(f"Net margin {m.net_margin_pct:.1f}%.")
        if m.debt_to_equity is not None:
            lines.append(f"Debt/equity {self._format_debt_equity(m.debt_to_equity)}.")
        if m.current_ratio is not None:
            lines.append(f"Current ratio {m.current_ratio:.2f}.")
        if m.free_cash_flow_latest is not None:
            lines.append(f"Free cash flow {self._fmt_compact_currency(m.free_cash_flow_latest)} (latest period).")
        if m.return_on_equity_pct is not None:
            lines.append(f"ROE {m.return_on_equity_pct:.1f}%.")
        if m.payout_ratio_pct is not None:
            lines.append(f"Payout ratio {m.payout_ratio_pct:.0f}%.")
        return lines

    def _derive_verdict(self, m: FinancialMetricsSnapshot) -> str:
        rg = m.revenue_growth_yoy
        nm = m.net_margin_pct
        gm = m.gross_margin_pct
        de = m.debt_to_equity
        fcf = m.free_cash_flow_latest

        high_growth = rg is not None and rg > 50
        moderate_growth = rg is not None and 5 <= rg <= 50
        contracting = rg is not None and rg < 0

        strong_profit = nm is not None and nm > 15
        modest_profit = nm is not None and 0 <= nm <= 15
        unprofitable = nm is not None and nm < 0
        deep_loss = nm is not None and nm < -20

        low_leverage = de is not None and de < 0.5
        high_leverage = de is not None and de > 2
        extreme_leverage = de is not None and de > 5

        positive_fcf = fcf is not None and fcf > 0
        negative_fcf = fcf is not None and fcf < 0
        commodity_like = gm is not None and gm < 30

        if deep_loss and extreme_leverage:
            return "Deep losses with acute balance-sheet strain"
        if deep_loss and negative_fcf:
            return "Loss-making with negative free cash flow"
        if high_growth and (unprofitable or negative_fcf):
            return "High-growth, speculative profile"
        if high_growth and strong_profit and positive_fcf:
            return "High-growth with scaling profitability"
        if extreme_leverage:
            return "Highly leveraged balance sheet"
        if high_leverage and commodity_like:
            return "Capital-intensive and highly leveraged"
        if strong_profit and low_leverage and positive_fcf and (rg is None or abs(rg) < 8):
            return "Mature, financially strong profile"
        if strong_profit and positive_fcf and moderate_growth:
            return "Profitable with solid growth"
        if modest_profit and moderate_growth:
            return "Profitable with moderate growth"
        if positive_fcf and (m.payout_ratio_pct or 0) >= 40 and modest_profit:
            return "Cash-generative value profile"
        if contracting and unprofitable:
            return "Turnaround situation"
        if contracting and modest_profit:
            return "Shrinking revenue, still profitable"
        if unprofitable and positive_fcf:
            return "Accounting losses despite positive free cash flow"
        if negative_fcf and strong_profit:
            return "Profitable on earnings, cash-hungry on capex"
        if positive_fcf:
            return "Cash-generative operations"
        if unprofitable:
            return "Earnings losses dominate the profile"
        return "Mixed financial profile"

    def _headline(
        self,
        symbol: str,
        verdict: str,
        metrics: FinancialMetricsSnapshot,
    ) -> str:
        hook = self._metric_hook(metrics)
        ticker = symbol.upper()
        if hook:
            return f"{verdict} for {ticker} — {hook}"
        return f"{verdict} for {ticker}."

    @staticmethod
    def _metric_hook(metrics: FinancialMetricsSnapshot) -> str | None:
        hooks: list[tuple[float, str]] = []

        if metrics.net_margin_pct is not None and metrics.net_margin_pct < -20:
            hooks.append(
                (abs(metrics.net_margin_pct), f"net margin {metrics.net_margin_pct:.0f}%")
            )
        if metrics.debt_to_equity is not None and metrics.debt_to_equity > 2:
            hooks.append(
                (
                    metrics.debt_to_equity * 10,
                    f"debt/equity {FinancialOverviewGenerator._format_debt_equity(metrics.debt_to_equity)}",
                )
            )
        if metrics.revenue_growth_yoy is not None and metrics.revenue_growth_yoy > 50:
            hooks.append(
                (metrics.revenue_growth_yoy, f"revenue +{metrics.revenue_growth_yoy:.0f}% YoY")
            )
        if metrics.free_cash_flow_latest is not None and metrics.free_cash_flow_latest < 0:
            hooks.append(
                (
                    abs(metrics.free_cash_flow_latest) / 1_000_000_000,
                    f"FCF {FinancialOverviewGenerator._fmt_compact_currency(metrics.free_cash_flow_latest)}",
                )
            )

        if not hooks:
            if metrics.net_margin_pct is not None and metrics.net_margin_pct > 15:
                return f"net margin {metrics.net_margin_pct:.0f}%"
            if metrics.revenue_growth_yoy is not None:
                sign = "+" if metrics.revenue_growth_yoy >= 0 else ""
                return f"revenue {sign}{metrics.revenue_growth_yoy:.0f}% YoY"
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


def build_metrics_snapshot(
    *,
    info: dict,
    revenue_by_period: dict[str, float | None],
    net_income_by_period: dict[str, float | None],
    gross_profit_by_period: dict[str, float | None],
    free_cash_flow_by_period: dict[str, float | None],
    dividends_by_period: dict[str, float | None],
    periods: list[str],
) -> FinancialMetricsSnapshot:
    latest = periods[0] if periods else None
    prior = periods[1] if len(periods) > 1 else None

    def yoy(values: dict[str, float | None]) -> float | None:
        if not latest or not prior:
            return None
        current = values.get(latest)
        previous = values.get(prior)
        if current is None or previous is None or previous == 0:
            return None
        return ((current - previous) / abs(previous)) * 100

    rev_growth = yoy(revenue_by_period)
    if rev_growth is None:
        rev_growth = FinancialOverviewGenerator.normalize_pct(info.get("revenueGrowth"))

    gross_margin: float | None = None
    if latest:
        rev = revenue_by_period.get(latest)
        gp = gross_profit_by_period.get(latest)
        if rev and gp is not None and rev != 0:
            gross_margin = (gp / rev) * 100
    if gross_margin is None:
        gross_margin = FinancialOverviewGenerator.normalize_pct(info.get("grossMargins"))

    net_margin: float | None = None
    if latest:
        rev = revenue_by_period.get(latest)
        ni = net_income_by_period.get(latest)
        if rev and ni is not None and rev != 0:
            net_margin = (ni / rev) * 100
    if net_margin is None:
        net_margin = FinancialOverviewGenerator.normalize_pct(info.get("profitMargins"))

    fcf_latest = free_cash_flow_by_period.get(latest) if latest else None
    if fcf_latest is None:
        raw_fcf = info.get("freeCashflow")
        if isinstance(raw_fcf, (int, float)):
            fcf_latest = float(raw_fcf)

    payout = FinancialOverviewGenerator.normalize_pct(
        _resolve_payout_ratio(info),
    )

    coverage: float | None = None
    if latest and fcf_latest is not None and fcf_latest > 0:
        div = dividends_by_period.get(latest)
        if div is not None:
            paid = abs(div)
            if paid > 0:
                coverage = fcf_latest / paid

    return FinancialMetricsSnapshot(
        revenue_growth_yoy=rev_growth,
        gross_margin_pct=gross_margin,
        net_margin_pct=net_margin,
        debt_to_equity=FinancialOverviewGenerator.normalize_debt_to_equity(
            info.get("debtToEquity"),
        ),
        current_ratio=(
            float(info["currentRatio"])
            if isinstance(info.get("currentRatio"), (int, float))
            else None
        ),
        free_cash_flow_latest=fcf_latest,
        free_cash_flow_yoy_pct=yoy(free_cash_flow_by_period),
        return_on_equity_pct=FinancialOverviewGenerator.normalize_pct(
            info.get("returnOnEquity"),
        ),
        payout_ratio_pct=payout,
        fcf_dividend_coverage=coverage,
    )


def _resolve_payout_ratio(info: dict) -> float | None:
    raw = info.get("payoutRatio")
    if isinstance(raw, (int, float)):
        return float(raw)

    dividend_rate = info.get("trailingAnnualDividendRate") or info.get("dividendRate")
    trailing_eps = info.get("trailingEps")
    if (
        isinstance(dividend_rate, (int, float))
        and isinstance(trailing_eps, (int, float))
        and trailing_eps > 0
    ):
        return float(dividend_rate) / float(trailing_eps)
    return None
