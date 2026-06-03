from __future__ import annotations

from dataclasses import dataclass

from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics
from app.builders.financial_sector_context import FinancialCompanyContext
from app.models.company_research_models import (
    FinancialStrength,
    FundamentalMetric,
    FundamentalsOverview,
    InvestmentThesis,
    ResearchSnapshot,
)
from app.models.yfinance_analysis_models import (
    PeriodEstimate,
    RecommendationBreakdown,
    StreetAnalysisSnapshot,
)
from app.models.yfinance_funds_models import EtfFundsSnapshot


@dataclass(frozen=True)
class _ScoredBullet:
    text: str
    materiality: float


class FundamentalsValuationGenerator:
    MAX_BULL = 3
    MAX_BEAR = 3
    MAX_SUMMARY_SENTENCES = 5

    def generate(
        self,
        *,
        symbol: str,
        snapshot: ResearchSnapshot | None,
        canonical: CanonicalFinancialMetrics | None,
        strength: FinancialStrength | None,
        street: StreetAnalysisSnapshot | None,
        metrics: list[FundamentalMetric],
        sector: str | None = None,
        industry: str | None = None,
    ) -> FundamentalsOverview:
        ctx = FinancialCompanyContext(symbol=symbol, sector=sector, industry=industry)
        bulls = self._bull_bullets(
            snapshot=snapshot,
            canonical=canonical,
            strength=strength,
            street=street,
            metrics=metrics,
            company_ctx=ctx,
        )
        bears = self._bear_bullets(
            snapshot=snapshot,
            canonical=canonical,
            strength=strength,
            street=street,
            metrics=metrics,
            company_ctx=ctx,
        )
        summary = self._valuation_summary(
            snapshot=snapshot,
            canonical=canonical,
            strength=strength,
            street=street,
            metrics=metrics,
            company_ctx=ctx,
            bulls=bulls,
            bears=bears,
        )
        return FundamentalsOverview(
            valuation_summary=summary,
            investment_thesis=InvestmentThesis(
                bull_case=self._top_bullets(bulls, self.MAX_BULL),
                bear_case=self._top_bullets(bears, self.MAX_BEAR),
            ),
        )

    def generate_etf(
        self,
        funds: EtfFundsSnapshot,
        *,
        dividend_yield_pct: float | None = None,
    ) -> FundamentalsOverview:
        bulls: list[_ScoredBullet] = []
        bears: list[_ScoredBullet] = []

        expense = funds.expense_ratio_pct
        category_expense = funds.category_expense_ratio_pct
        if expense is not None and category_expense is not None and expense < category_expense:
            bulls.append(
                _ScoredBullet(
                    text=(
                        f"Expense ratio of {expense:.2f}% is below the category average "
                        f"({category_expense:.2f}%), keeping more return in the investor's pocket."
                    ),
                    materiality=78,
                )
            )
        elif expense is not None and expense <= 0.15:
            bulls.append(
                _ScoredBullet(
                    text=f"Low {expense:.2f}% expense ratio supports long-run compounding versus pricier peers.",
                    materiality=70,
                )
            )

        if dividend_yield_pct is not None and dividend_yield_pct >= 2:
            bulls.append(
                _ScoredBullet(
                    text=f"Dividend yield near {dividend_yield_pct:.1f}% adds income while holding the basket.",
                    materiality=62,
                )
            )

        if funds.top_holdings and len(funds.top_holdings) >= 5:
            bulls.append(
                _ScoredBullet(
                    text="Broad top holdings spread reduces single-name risk versus concentrated ETFs.",
                    materiality=55,
                )
            )

        if expense is not None and category_expense is not None and expense > category_expense * 1.15:
            bears.append(
                _ScoredBullet(
                    text=(
                        f"Expense ratio of {expense:.2f}% is above category norms "
                        f"({category_expense:.2f}%), a headwind to net returns."
                    ),
                    materiality=75,
                )
            )

        if funds.holdings_turnover_pct is not None and funds.holdings_turnover_pct > 40:
            bears.append(
                _ScoredBullet(
                    text=(
                        f"High {funds.holdings_turnover_pct:.0f}% turnover can raise trading costs "
                        "and tax friction in taxable accounts."
                    ),
                    materiality=60,
                )
            )

        if funds.top_holdings and funds.top_holdings[0].weight_pct > 12:
            top = funds.top_holdings[0]
            bears.append(
                _ScoredBullet(
                    text=(
                        f"Top holding {top.name} at {top.weight_pct:.1f}% weight "
                        "concentrates outcome risk in a single name."
                    ),
                    materiality=68,
                )
            )

        sentences = [
            "Fundamentals here are about cost, yield, and what the basket prices in — not operating margins.",
        ]
        if expense is not None:
            sentences.append(f"The fund charges {expense:.2f}% annually.")
        if funds.category:
            sentences.append(f"Category: {funds.category}.")
        if category_expense is not None:
            sentences.append(
                f"Peers in the category average about {category_expense:.2f}% expense."
            )
        sentences.append(
            "Compare yield, turnover, and top weights before sizing a position."
        )

        return FundamentalsOverview(
            valuation_summary=" ".join(sentences[: self.MAX_SUMMARY_SENTENCES]),
            investment_thesis=InvestmentThesis(
                bull_case=self._top_bullets(bulls, self.MAX_BULL)
                or ["Cost structure and diversification can support a core allocation."],
                bear_case=self._top_bullets(bears, self.MAX_BEAR)
                or ["Fees and concentration still matter on a total-return basis."],
            ),
        )

    def _bull_bullets(
        self,
        *,
        snapshot: ResearchSnapshot | None,
        canonical: CanonicalFinancialMetrics | None,
        strength: FinancialStrength | None,
        street: StreetAnalysisSnapshot | None,
        metrics: list[FundamentalMetric],
        company_ctx: FinancialCompanyContext,
    ) -> list[_ScoredBullet]:
        out: list[_ScoredBullet] = []
        targets = street.price_targets if street else None
        upside = targets.upside_to_mean_pct if targets else None
        current = targets.current if targets else snapshot.price if snapshot else None
        mean_target = targets.mean if targets else None

        if upside is not None and upside >= 12:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Shares trade about {upside:.0f}% below the Street mean price target "
                        f"({self._fmt_price(mean_target)} vs {self._fmt_price(current)}), "
                        "leaving room if estimates prove conservative."
                    ),
                    materiality=88,
                )
            )
        elif upside is not None and upside >= 5:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Modest {upside:.0f}% upside to the consensus mean target "
                        "suggests the market is not fully pricing a bullish scenario."
                    ),
                    materiality=72,
                )
            )

        trailing_pe = self._parse_multiple(self._metric_value(metrics, "P/E (trailing)"))
        forward_pe = self._parse_multiple(self._metric_value(metrics, "P/E (forward)"))
        if (
            trailing_pe is not None
            and forward_pe is not None
            and forward_pe < trailing_pe * 0.85
            and forward_pe > 0
        ):
            out.append(
                _ScoredBullet(
                    text=(
                        f"Forward P/E of {forward_pe:.1f}x is below trailing {trailing_pe:.1f}x — "
                        "analysts expect earnings to catch up to the current price."
                    ),
                    materiality=76,
                )
            )
        elif trailing_pe is not None and trailing_pe <= 18 and canonical:
            rg = canonical.revenue_growth_yoy
            if rg is not None and rg >= 10:
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Trailing P/E near {trailing_pe:.1f}x with "
                            f"{canonical.format_revenue_growth() or 'solid'} revenue growth "
                            "pairs a reasonable multiple with expansion."
                        ),
                        materiality=74,
                    )
                )

        if street and street.consensus_label:
            label = street.consensus_label.lower()
            if "buy" in label and "sell" not in label:
                out.append(
                    _ScoredBullet(
                        text=f"Wall Street consensus is {street.consensus_label}, supporting a constructive setup.",
                        materiality=65,
                    )
                )

        if street and street.estimate_revision_headline:
            headline = street.estimate_revision_headline.lower()
            if any(token in headline for token in ("up", "raised", "higher", "positive")):
                out.append(
                    _ScoredBullet(
                        text=f"Estimate revisions lean positive: {street.estimate_revision_headline}",
                        materiality=70,
                    )
                )

        next_eps = self._next_period_estimate(street, "+1q") or street.next_quarter_eps if street else None
        if next_eps and next_eps.growth_pct is not None and next_eps.growth_pct >= 8:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Next-quarter EPS estimates imply "
                        f"{next_eps.growth_pct:+.0f}% growth — execution is already in the numbers."
                    ),
                    materiality=68,
                )
            )

        if strength and strength.profile in {
            "Financially Strong",
            "Profitable Compounder",
            "Cash-Generating Value",
        }:
            out.append(
                _ScoredBullet(
                    text=(
                        f"A {strength.profile} operating profile gives the market a credible "
                        "path to justify today's valuation if growth holds."
                    ),
                    materiality=58,
                )
            )

        if company_ctx.archetype.value == "bank" and canonical:
            roe = canonical.return_on_equity_pct
            if roe is not None and roe >= 10:
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Return on equity of {roe:.0f}% supports book-value-based "
                            "valuation for a bank at today's price."
                        ),
                        materiality=64,
                    )
                )

        return out

    def _bear_bullets(
        self,
        *,
        snapshot: ResearchSnapshot | None,
        canonical: CanonicalFinancialMetrics | None,
        strength: FinancialStrength | None,
        street: StreetAnalysisSnapshot | None,
        metrics: list[FundamentalMetric],
        company_ctx: FinancialCompanyContext,
    ) -> list[_ScoredBullet]:
        out: list[_ScoredBullet] = []
        targets = street.price_targets if street else None
        upside = targets.upside_to_mean_pct if targets else None
        current = targets.current if targets else snapshot.price if snapshot else None
        mean_target = targets.mean if targets else None

        if upside is not None and upside <= -5:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Price sits {abs(upside):.0f}% above the mean analyst target "
                        f"({self._fmt_price(current)} vs {self._fmt_price(mean_target)}), "
                        "so the stock must outperform estimates to work from here."
                    ),
                    materiality=90,
                )
            )

        trailing_pe = self._parse_multiple(self._metric_value(metrics, "P/E (trailing)"))
        forward_pe = self._parse_multiple(self._metric_value(metrics, "P/E (forward)"))
        if trailing_pe is not None and trailing_pe >= 35:
            growth_note = ""
            if canonical and canonical.revenue_growth_yoy is not None:
                growth_note = f" with revenue growing {canonical.format_revenue_growth()}"
            out.append(
                _ScoredBullet(
                    text=(
                        f"Trailing P/E of {trailing_pe:.1f}x prices in strong growth{growth_note} — "
                        "disappointment would compress the multiple quickly."
                    ),
                    materiality=85,
                )
            )
        elif forward_pe is not None and forward_pe >= 40:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Forward P/E near {forward_pe:.1f}x leaves little margin for "
                        "earnings misses or guide-downs."
                    ),
                    materiality=80,
                )
            )

        if canonical:
            if (
                canonical.free_cash_flow_latest is not None
                and canonical.free_cash_flow_latest < 0
                and trailing_pe is not None
                and trailing_pe >= 25
            ):
                out.append(
                    _ScoredBullet(
                        text=(
                            "Negative free cash flow alongside a premium multiple means "
                            "valuation depends on future profits, not today's cash generation."
                        ),
                        materiality=82,
                    )
                )
            if canonical.revenue_growth_yoy is not None and canonical.revenue_growth_yoy < 0:
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Revenue decline of {canonical.format_revenue_growth()} makes "
                            "today's price a bet on stabilization, not momentum."
                        ),
                        materiality=78,
                    )
                )

        if street and street.estimate_revision_headline:
            headline = street.estimate_revision_headline.lower()
            if any(token in headline for token in ("down", "cut", "lower", "negative")):
                out.append(
                    _ScoredBullet(
                        text=f"Estimate revisions are softening: {street.estimate_revision_headline}",
                        materiality=72,
                    )
                )

        if street and street.recommendation:
            sell_skew = self._sell_skew(street.recommendation)
            if sell_skew >= 0.25:
                out.append(
                    _ScoredBullet(
                        text="A meaningful share of analysts rate the stock Sell or Strong Sell.",
                        materiality=66,
                    )
                )

        if strength and strength.profile in {
            "High Growth / High Risk",
            "Speculative Growth",
            "Leveraged Turnaround",
        }:
            out.append(
                _ScoredBullet(
                    text=(
                        f"The {strength.profile} label flags that the price still requires "
                        "flawless execution on growth, margins, or balance-sheet repair."
                    ),
                    materiality=70,
                )
            )

        if company_ctx.archetype.value == "biotech" and canonical:
            if canonical.free_cash_flow_latest is not None and canonical.free_cash_flow_latest < 0:
                out.append(
                    _ScoredBullet(
                        text=(
                            "Biotech-style cash burn means valuation is tied to funding and "
                            "pipeline milestones, not near-term earnings."
                        ),
                        materiality=75,
                    )
                )

        return out

    def _valuation_summary(
        self,
        *,
        snapshot: ResearchSnapshot | None,
        canonical: CanonicalFinancialMetrics | None,
        strength: FinancialStrength | None,
        street: StreetAnalysisSnapshot | None,
        metrics: list[FundamentalMetric],
        company_ctx: FinancialCompanyContext,
        bulls: list[_ScoredBullet],
        bears: list[_ScoredBullet],
    ) -> str:
        sentences: list[str] = []
        targets = street.price_targets if street else None
        current = targets.current if targets else snapshot.price if snapshot else None
        mean_target = targets.mean if targets else None
        upside = targets.upside_to_mean_pct if targets else None

        if current is not None and mean_target is not None:
            vs = self._vs_mean_target_label(current, mean_target, upside)
            sentences.append(
                f"At {self._fmt_price(current)}, the stock is {vs} "
                f"against a mean analyst target of {self._fmt_price(mean_target)}."
            )
        elif current is not None:
            sentences.append(f"The stock last traded near {self._fmt_price(current)}.")

        trailing_pe = self._parse_multiple(self._metric_value(metrics, "P/E (trailing)"))
        forward_pe = self._parse_multiple(self._metric_value(metrics, "P/E (forward)"))
        if trailing_pe is not None and forward_pe is not None:
            if forward_pe < trailing_pe * 0.9:
                sentences.append(
                    f"Forward P/E ({forward_pe:.1f}x) sits below trailing ({trailing_pe:.1f}x), "
                    "so the market is pricing earnings improvement."
                )
            elif forward_pe > trailing_pe * 1.1:
                sentences.append(
                    f"Forward P/E ({forward_pe:.1f}x) exceeds trailing ({trailing_pe:.1f}x), "
                    "implying earnings pressure or one-off boosts in the rearview."
                )
            else:
                sentences.append(
                    f"Trailing and forward P/E are both near {trailing_pe:.1f}x — "
                    "expectations are relatively stable."
                )
        elif trailing_pe is not None:
            sentences.append(f"Shares trade at about {trailing_pe:.1f}x trailing earnings.")

        if street and street.growth_context_headline:
            sentences.append(street.growth_context_headline.rstrip(".") + ".")
        elif canonical and canonical.revenue_growth_yoy is not None:
            sentences.append(
                f"Revenue is {canonical.format_revenue_growth()} year over year, "
                "which shapes how much growth is already embedded in the price."
            )

        if strength:
            sentences.append(
                f"Operationally the company screens as {strength.profile}; "
                "this tab focuses on whether the price compensates for that profile."
            )

        if bulls and bears:
            sentences.append(
                "The bull case hinges on estimates and multiples holding up; "
                "the bear case centers on what happens if they do not."
            )
        elif bears:
            sentences.append(
                "Upside from here likely requires estimate beats or multiple expansion."
            )
        else:
            sentences.append(
                "Execution on the next few quarters will determine whether today's valuation looks fair."
            )

        return " ".join(sentences[: self.MAX_SUMMARY_SENTENCES])

    @staticmethod
    def _top_bullets(bullets: list[_ScoredBullet], limit: int) -> list[str]:
        ranked = sorted(bullets, key=lambda item: item.materiality, reverse=True)
        return [item.text for item in ranked[:limit]]

    @staticmethod
    def _metric_value(metrics: list[FundamentalMetric], label: str) -> str | None:
        target = label.lower()
        for metric in metrics:
            if metric.label.lower() == target:
                return metric.value
        return None

    @staticmethod
    def _parse_multiple(raw: str | None) -> float | None:
        if not raw:
            return None
        cleaned = raw.lower().replace("x", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _fmt_price(value: float | None) -> str:
        if value is None:
            return "—"
        return f"${value:,.2f}"

    @staticmethod
    def _vs_mean_target_label(
        current: float,
        mean: float,
        upside_pct: float | None,
    ) -> str:
        if upside_pct is not None:
            if upside_pct >= 1:
                return f"roughly {upside_pct:.0f}% below"
            if upside_pct <= -1:
                return f"roughly {abs(upside_pct):.0f}% above"
        pct = ((current - mean) / mean) * 100 if mean else 0
        if pct <= -1:
            return f"roughly {abs(pct):.0f}% below"
        if pct >= 1:
            return f"roughly {pct:.0f}% above"
        return "in line with"

    @staticmethod
    def _sell_skew(recommendation: RecommendationBreakdown) -> float:
        total = (
            recommendation.strong_buy
            + recommendation.buy
            + recommendation.hold
            + recommendation.sell
            + recommendation.strong_sell
        )
        if total <= 0:
            return 0.0
        return (recommendation.sell + recommendation.strong_sell) / total

    @staticmethod
    def _next_period_estimate(
        street: StreetAnalysisSnapshot | None,
        period_key: str,
    ) -> PeriodEstimate | None:
        if not street:
            return None
        for row in street.eps_estimates:
            if row.period_key == period_key:
                return row
        return None
