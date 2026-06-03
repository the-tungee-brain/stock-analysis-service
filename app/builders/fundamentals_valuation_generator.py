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
    ValuationSignal,
)
from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.models.yfinance_funds_models import EtfFundsSnapshot


@dataclass(frozen=True)
class _ScoredBullet:
    text: str
    materiality: float


class FundamentalsValuationGenerator:
    MAX_BULL = 3
    MAX_BEAR = 3

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
        signals = self._valuation_signals(
            snapshot=snapshot,
            canonical=canonical,
            street=street,
            metrics=metrics,
        )
        bulls = self._bull_bullets(
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
        conclusion = self._valuation_conclusion(
            snapshot=snapshot,
            canonical=canonical,
            strength=strength,
            street=street,
            metrics=metrics,
            bulls=bulls,
            bears=bears,
        )
        summary = self._valuation_summary(
            snapshot=snapshot,
            canonical=canonical,
            metrics=metrics,
            street=street,
        )
        return FundamentalsOverview(
            valuation_conclusion=conclusion,
            valuation_summary=summary,
            valuation_signals=signals,
            investment_thesis=InvestmentThesis(
                bull_case=self._top_bullets(bulls, self.MAX_BULL),
                bear_case=self._top_bullets(bears, self.MAX_BEAR),
            ),
            street_context=self._street_context(street),
        )

    def generate_etf(
        self,
        funds: EtfFundsSnapshot,
        *,
        dividend_yield_pct: float | None = None,
    ) -> FundamentalsOverview:
        bulls: list[_ScoredBullet] = []
        bears: list[_ScoredBullet] = []
        signals: list[ValuationSignal] = []

        expense = funds.expense_ratio_pct
        category_expense = funds.category_expense_ratio_pct
        if expense is not None:
            signals.append(
                ValuationSignal(label="Expense ratio", value=f"{expense:.2f}%")
            )
        if dividend_yield_pct is not None:
            signals.append(
                ValuationSignal(label="Dividend yield", value=f"{dividend_yield_pct:.2f}%")
            )

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
        if dividend_yield_pct is not None and dividend_yield_pct >= 2:
            bulls.append(
                _ScoredBullet(
                    text=f"Dividend yield near {dividend_yield_pct:.1f}% adds income while holding the basket.",
                    materiality=62,
                )
            )
        if expense is not None and category_expense is not None and expense > category_expense * 1.15:
            bears.append(
                _ScoredBullet(
                    text=(
                        f"Expense ratio of {expense:.2f}% is above category norms "
                        f"({category_expense:.2f}%), a drag on long-run returns."
                    ),
                    materiality=75,
                )
            )

        conclusion = (
            "The fund's attractiveness at today's price depends mainly on fees, yield, "
            "and whether the underlying basket matches your risk budget."
        )
        if expense is not None and category_expense is not None and expense > category_expense:
            conclusion = (
                f"The fund charges {expense:.2f}% annually versus a {category_expense:.2f}% "
                "category average — cost is the main valuation hurdle from here."
            )

        return FundamentalsOverview(
            valuation_conclusion=conclusion,
            valuation_summary=conclusion,
            valuation_signals=signals,
            investment_thesis=InvestmentThesis(
                bull_case=self._top_bullets(bulls, self.MAX_BULL)
                or ["Low cost and broad exposure can support a core allocation."],
                bear_case=self._top_bullets(bears, self.MAX_BEAR)
                or ["Fees and concentration still matter on a total-return basis."],
            ),
        )

    def _valuation_signals(
        self,
        *,
        snapshot: ResearchSnapshot | None,
        canonical: CanonicalFinancialMetrics | None,
        street: StreetAnalysisSnapshot | None,
        metrics: list[FundamentalMetric],
    ) -> list[ValuationSignal]:
        signals: list[ValuationSignal] = []
        pb = self._metric_value(metrics, "Price / book")
        if pb:
            signals.append(ValuationSignal(label="Price / book", value=pb))

        trailing_pe = self._metric_value(metrics, "P/E (trailing)")
        if trailing_pe:
            signals.append(ValuationSignal(label="P/E (trailing)", value=trailing_pe))

        eps = self._metric_value(metrics, "EPS (trailing)") or self._metric_value(
            metrics, "EPS (forward)"
        )
        if eps:
            signals.append(ValuationSignal(label="EPS", value=eps))

        targets = street.price_targets if street else None
        if targets and targets.current is not None and targets.mean is not None:
            gap = self._format_target_gap(
                targets.current, targets.mean, targets.upside_to_mean_pct
            )
            signals.append(ValuationSignal(label="Analyst target gap", value=gap))

        if canonical:
            rev = canonical.format_revenue_growth()
            if rev:
                signals.append(ValuationSignal(label="Revenue growth", value=rev))
            net = canonical.format_net_margin()
            if net:
                signals.append(ValuationSignal(label="Net margin", value=net))
            fcf = canonical.format_free_cash_flow()
            if fcf:
                signals.append(ValuationSignal(label="Free cash flow", value=fcf))

        if snapshot and snapshot.peRatio is not None and not any(
            signal.label.startswith("P/E") for signal in signals
        ):
            signals.append(
                ValuationSignal(label="P/E (trailing)", value=f"{snapshot.peRatio:.1f}x")
            )

        return signals[:8]

    def _bull_bullets(
        self,
        *,
        canonical: CanonicalFinancialMetrics | None,
        strength: FinancialStrength | None,
        street: StreetAnalysisSnapshot | None,
        metrics: list[FundamentalMetric],
        company_ctx: FinancialCompanyContext,
    ) -> list[_ScoredBullet]:
        out: list[_ScoredBullet] = []

        if canonical and canonical.revenue_growth_yoy is not None:
            rg = canonical.revenue_growth_yoy
            display = canonical.format_revenue_growth()
            if rg >= 15:
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Revenue growth of {display} YoY shows strong commercial momentum "
                            "and rising demand for the product set."
                        ),
                        materiality=90 if rg >= 30 else 78,
                    )
                )
            elif rg >= 5:
                out.append(
                    _ScoredBullet(
                        text=f"Revenue is still expanding at {display} YoY, supporting a growth narrative.",
                        materiality=62,
                    )
                )

        if canonical:
            gross = canonical.format_gross_margin()
            net = canonical.format_net_margin()
            if canonical.gross_margin_pct is not None and canonical.gross_margin_pct >= 45:
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Gross margin of {gross} indicates pricing power and "
                            "scalable unit economics as revenue compounds."
                        ),
                        materiality=76,
                    )
                )
            elif canonical.net_margin_pct is not None and canonical.net_margin_pct >= 12:
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Net margin of {net} shows the business converts revenue "
                            "into profit at a healthy rate."
                        ),
                        materiality=72,
                    )
                )

            fcf = canonical.format_free_cash_flow()
            if canonical.free_cash_flow_latest is not None and canonical.free_cash_flow_latest > 0:
                trend = ""
                if canonical.free_cash_flow_yoy_pct is not None and canonical.free_cash_flow_yoy_pct > 8:
                    trend = f", up {canonical.free_cash_flow_yoy_pct:.0f}% YoY"
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Free cash flow of {fcf}{trend} funds reinvestment, "
                            "deleveraging, or shareholder returns without relying on new capital."
                        ),
                        materiality=85,
                    )
                )

            if (
                canonical.revenue_growth_yoy is not None
                and canonical.revenue_growth_yoy >= 20
                and canonical.gross_margin_pct is not None
                and canonical.gross_margin_pct >= 40
            ):
                out.append(
                    _ScoredBullet(
                        text=(
                            "High growth with solid gross margins points to contracted or "
                            "recurring demand rather than one-off volume spikes."
                        ),
                        materiality=70,
                    )
                )

        rev_est = self._next_period_estimate(street, "+1q", revenue=True)
        if rev_est and rev_est.growth_pct is not None and rev_est.growth_pct >= 10:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Forward revenue estimates imply {rev_est.growth_pct:+.0f}% growth, "
                        "suggesting backlog or pipeline conversion into reported sales."
                    ),
                    materiality=68,
                )
            )

        if strength and strength.score >= 60:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Operating momentum aligns with a {strength.profile} profile "
                        f"(financial health score {strength.score}/100)."
                    ),
                    materiality=55,
                )
            )

        if company_ctx.archetype.value == "bank" and canonical:
            roe = canonical.return_on_equity_pct
            if roe is not None and roe >= 10:
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Return on equity of {roe:.0f}% reflects earning power on "
                            "tangible book — a core fundamental support for the stock."
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
        trailing_pe = self._parse_multiple(self._metric_value(metrics, "P/E (trailing)"))
        forward_pe = self._parse_multiple(self._metric_value(metrics, "P/E (forward)"))
        price_book = self._parse_multiple(self._metric_value(metrics, "Price / book"))

        if trailing_pe is not None and trailing_pe >= 30:
            growth_note = ""
            if canonical and canonical.revenue_growth_yoy is not None:
                growth_note = f" despite {canonical.format_revenue_growth()} revenue growth"
            out.append(
                _ScoredBullet(
                    text=(
                        f"Trailing P/E of {trailing_pe:.1f}x embeds a premium valuation{growth_note} — "
                        "multiples can compress if growth or margins disappoint."
                    ),
                    materiality=88,
                )
            )
        elif forward_pe is not None and forward_pe >= 35:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Forward P/E of {forward_pe:.1f}x prices in a demanding earnings path "
                        "with limited room for execution slips."
                    ),
                    materiality=82,
                )
            )

        if price_book is not None and price_book >= 4:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Price/book of {price_book:.1f}x is rich relative to accounting value "
                        "unless ROE and growth stay elevated."
                    ),
                    materiality=70,
                )
            )

        if canonical and canonical.net_margin_pct is not None:
            if canonical.net_margin_pct < 0:
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Net margin of {canonical.format_net_margin()} means profitability "
                            "does not yet support the current valuation on earnings alone."
                        ),
                        materiality=86,
                    )
                )
            elif canonical.net_margin_pct < 5:
                out.append(
                    _ScoredBullet(
                        text=(
                            f"Thin net margin of {canonical.format_net_margin()} leaves little "
                            "cushion if costs rise or pricing weakens."
                        ),
                        materiality=68,
                    )
                )

        if canonical and canonical.debt_to_equity is not None and canonical.debt_to_equity > 2:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Debt/equity of {canonical.format_debt_equity()} raises financial risk "
                        "if rates rise or cash flow slows."
                    ),
                    materiality=78,
                )
            )

        if canonical and canonical.current_ratio is not None and canonical.current_ratio < 1:
            out.append(
                _ScoredBullet(
                    text=(
                        f"Current ratio of {canonical.format_current_ratio()} signals "
                        "liquidity pressure in a downturn."
                    ),
                    materiality=74,
                )
            )

        targets = street.price_targets if street else None
        upside = targets.upside_to_mean_pct if targets else None
        if upside is not None and upside <= -5:
            current = targets.current if targets else snapshot.price if snapshot else None
            mean_target = targets.mean if targets else None
            out.append(
                _ScoredBullet(
                    text=(
                        f"Price is {abs(upside):.0f}% above the mean analyst target "
                        f"({self._fmt_price(current)} vs {self._fmt_price(mean_target)}) — "
                        "returns depend on estimates moving higher, not just meeting them."
                    ),
                    materiality=84,
                )
            )

        if canonical and canonical.free_cash_flow_latest is not None and canonical.free_cash_flow_latest < 0:
            out.append(
                _ScoredBullet(
                    text=(
                        "Negative free cash flow means the equity story requires a turnaround "
                        "in operations or external funding."
                    ),
                    materiality=80,
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
                        f"Execution risk is elevated for a {strength.profile} name — "
                        "the market is paying ahead of proven stability."
                    ),
                    materiality=76,
                )
            )
        elif canonical and canonical.revenue_growth_yoy is not None and canonical.revenue_growth_yoy < 0:
            out.append(
                _ScoredBullet(
                    text=(
                        "Contracting revenue raises execution risk: the stock must prove "
                        "stabilization before multiple expansion is realistic."
                    ),
                    materiality=72,
                )
            )

        if company_ctx.archetype.value == "biotech" and canonical:
            if canonical.free_cash_flow_latest is not None and canonical.free_cash_flow_latest < 0:
                out.append(
                    _ScoredBullet(
                        text=(
                            "Biotech cash burn creates execution and funding risk — "
                            "valuation hinges on pipeline milestones, not current earnings."
                        ),
                        materiality=75,
                    )
                )

        return out

    def _valuation_conclusion(
        self,
        *,
        snapshot: ResearchSnapshot | None,
        canonical: CanonicalFinancialMetrics | None,
        strength: FinancialStrength | None,
        street: StreetAnalysisSnapshot | None,
        metrics: list[FundamentalMetric],
        bulls: list[_ScoredBullet],
        bears: list[_ScoredBullet],
    ) -> str:
        trailing_pe = self._parse_multiple(self._metric_value(metrics, "P/E (trailing)"))
        targets = street.price_targets if street else None
        upside = targets.upside_to_mean_pct if targets else None
        rg = canonical.revenue_growth_yoy if canonical else None
        nm = canonical.net_margin_pct if canonical else None

        premium_multiple = trailing_pe is not None and trailing_pe >= 28
        hypergrowth = rg is not None and rg >= 25
        weak_profit = nm is not None and nm < 5
        above_target = upside is not None and upside <= -8
        below_target = upside is not None and upside >= 10
        strong_fundamentals = strength is not None and strength.score >= 62 and not weak_profit

        if premium_multiple and hypergrowth and weak_profit:
            return (
                "The stock trades at a premium valuation that assumes continued hypergrowth "
                "and improving profitability. Current fundamentals alone do not fully justify "
                "the market price."
            )
        if premium_multiple and hypergrowth:
            return (
                "A rich multiple prices in sustained hypergrowth — investors are paying for "
                "future margin expansion, not just today's revenue curve."
            )
        if premium_multiple and above_target:
            return (
                "The shares trade above both typical earnings multiples and the Street mean target, "
                "so expectations for growth and execution are already ambitious."
            )
        if premium_multiple and not hypergrowth:
            return (
                "The market assigns a premium multiple without matching revenue momentum — "
                "either estimates must rise or the multiple needs to normalize for investors to earn a return."
            )
        if below_target and strong_fundamentals:
            return (
                "Fundamentals are solid relative to the price, and the stock trades below the "
                "mean analyst target — returns may come from earnings delivery rather than multiple expansion."
            )
        if weak_profit and (premium_multiple or above_target):
            return (
                "Weak profitability alongside a demanding price means investors are betting on "
                "a sharp earnings inflection, not what the business earns today."
            )
        if bears and not bulls:
            return (
                "Valuation and balance-sheet risks dominate — the price already reflects optimism "
                "that must materialize in earnings and cash flow."
            )
        if bulls and not bears:
            return (
                "Operating fundamentals support the narrative at today's price, though returns still "
                "require growth and margins to hold as the market expects."
            )
        if hypergrowth:
            return (
                "Hypergrowth is largely priced in; investors need sustained revenue gains and "
                "margin progress to earn a return from current levels."
            )
        return (
            "What happens next quarter — growth, margins, and cash conversion — determines whether "
            "today's price already captures the upside or leaves room for investors."
        )

    def _valuation_summary(
        self,
        *,
        snapshot: ResearchSnapshot | None,
        canonical: CanonicalFinancialMetrics | None,
        metrics: list[FundamentalMetric],
        street: StreetAnalysisSnapshot | None,
    ) -> str:
        parts: list[str] = []
        trailing_pe = self._parse_multiple(self._metric_value(metrics, "P/E (trailing)"))
        forward_pe = self._parse_multiple(self._metric_value(metrics, "P/E (forward)"))

        if trailing_pe is not None:
            parts.append(f"Shares trade near {trailing_pe:.1f}x trailing earnings.")
        if forward_pe is not None and trailing_pe is not None and forward_pe < trailing_pe * 0.9:
            parts.append(
                f"Forward P/E of {forward_pe:.1f}x implies the market expects earnings to improve."
            )
        if canonical and canonical.revenue_growth_yoy is not None:
            parts.append(
                f"Revenue is {canonical.format_revenue_growth()} YoY — that pace sets how much "
                "growth is already embedded in the price."
            )
        targets = street.price_targets if street else None
        if targets and targets.mean is not None and targets.current is not None:
            parts.append(
                f"Versus a mean target of {self._fmt_price(targets.mean)}, "
                f"the last price of {self._fmt_price(targets.current)} "
                f"is {self._format_target_gap(targets.current, targets.mean, targets.upside_to_mean_pct).lower()}."
            )
        elif snapshot:
            parts.append(f"Last price near {self._fmt_price(snapshot.price)}.")

        return " ".join(parts[:4])

    @staticmethod
    def _street_context(street: StreetAnalysisSnapshot | None) -> str:
        if not street:
            return ""
        parts: list[str] = []
        if street.consensus_label:
            parts.append(f"Consensus rating: {street.consensus_label}")
        targets = street.price_targets
        if targets and targets.mean is not None and targets.upside_to_mean_pct is not None:
            parts.append(
                f"mean target implies {targets.upside_to_mean_pct:+.0f}% from the last price"
            )
        if street.estimate_revision_headline:
            parts.append(street.estimate_revision_headline.rstrip("."))
        if not parts:
            return ""
        return (
            "Wall Street (supporting context): "
            + "; ".join(parts)
            + ". Use estimates as a cross-check, not the core thesis."
        )

    @staticmethod
    def _format_target_gap(
        current: float,
        mean: float,
        upside_pct: float | None,
    ) -> str:
        if upside_pct is not None:
            if upside_pct >= 0.5:
                return f"{upside_pct:.1f}% below mean target"
            if upside_pct <= -0.5:
                return f"{abs(upside_pct):.1f}% above mean target"
        pct = ((current - mean) / mean) * 100 if mean else 0
        if pct <= -0.5:
            return f"{abs(pct):.1f}% below mean target"
        if pct >= 0.5:
            return f"{pct:.1f}% above mean target"
        return "At mean target"

    @staticmethod
    def _top_bullets(bullets: list[_ScoredBullet], limit: int) -> list[str]:
        ranked = sorted(bullets, key=lambda item: item.materiality, reverse=True)
        picked = [item.text for item in ranked[:limit]]
        if picked:
            return picked
        return ["Operating trends are mixed — see valuation signals for the key inputs."]

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
    def _next_period_estimate(
        street: StreetAnalysisSnapshot | None,
        period_key: str,
        *,
        revenue: bool = False,
    ):
        if not street:
            return None
        rows = street.revenue_estimates if revenue else street.eps_estimates
        for row in rows:
            if row.period_key == period_key:
                return row
        return None
