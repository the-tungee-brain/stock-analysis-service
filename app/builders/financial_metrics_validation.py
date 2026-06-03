from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics
from app.builders.financial_sector_context import FinancialCompanyContext
from app.models.company_research_models import FinancialStrength, FundamentalMetric

if TYPE_CHECKING:
    from app.builders.financial_overview_generator import FinancialOverviewResult

_PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_RATIO_RE = re.compile(r"(-?\d+(?:\.\d+)?)x", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"-?\$\d+(?:\.\d+)?[BMKT]?|-?\$[\d,]+")


class FinancialMetricsConsistencyError(AssertionError):
    pass


def validate_strength_matches_canonical(
    strength: FinancialStrength,
    canonical: CanonicalFinancialMetrics,
    *,
    ctx: FinancialCompanyContext | None = None,
) -> None:
    key_by_label = {metric.label.lower(): metric for metric in strength.key_metrics}
    canonical_rows = {row.label.lower(): row for row in canonical.to_key_metrics()}

    for label, row in canonical_rows.items():
        displayed = key_by_label.get(label)
        if displayed is None:
            raise FinancialMetricsConsistencyError(
                f"Missing key metric on strength payload: {row.label}",
            )
        if displayed.value != row.value:
            raise FinancialMetricsConsistencyError(
                f"Key metric value mismatch for {row.label}: "
                f"display={displayed.value!r} canonical={row.value!r}",
            )

    context = ctx or FinancialCompanyContext(symbol="", sector=None, industry=None)
    texts = [
        strength.financial_verdict,
        strength.headline,
        *strength.strengths,
        *strength.risks,
        *strength.highlights,
    ]
    _assert_narrative_uses_canonical_values(texts, canonical)
    _assert_strengths_risks_use_key_metrics(strength, context)
    _assert_profile_matches_breakdown(strength.profile, strength.score_breakdown)
    _assert_verdict_is_single_sentence(strength.financial_verdict)


def validate_key_metrics_match_canonical(
    metrics: list[FundamentalMetric],
    canonical: CanonicalFinancialMetrics,
) -> None:
    canonical_rows = {row.label.lower(): row for row in canonical.to_key_metrics()}
    for label, row in canonical_rows.items():
        displayed = next(
            (metric for metric in metrics if metric.label.lower() == label),
            None,
        )
        if displayed is None:
            raise FinancialMetricsConsistencyError(
                f"Metrics list missing canonical row: {row.label}",
            )
        if displayed.value != row.value:
            raise FinancialMetricsConsistencyError(
                f"Metrics list mismatch for {row.label}: "
                f"display={displayed.value!r} canonical={row.value!r}",
            )


def _assert_narrative_uses_canonical_values(
    texts: list[str],
    canonical: CanonicalFinancialMetrics,
) -> None:
    joined = " ".join(text for text in texts if text).lower()

    if canonical.revenue_growth_yoy is not None:
        expected = f"{canonical.revenue_growth_yoy:.0f}"
        rounded = f"{canonical.revenue_growth_yoy:.1f}"
        if expected not in joined and rounded not in joined:
            if _pct_re.search(joined):
                raise FinancialMetricsConsistencyError(
                    "Narrative cites revenue growth % that does not match canonical value.",
                )

    if canonical.net_margin_pct is not None:
        expected = f"{canonical.net_margin_pct:.0f}"
        if "margin" in joined and expected not in joined:
            alt = f"{canonical.net_margin_pct:.1f}"
            if alt not in joined:
                raise FinancialMetricsConsistencyError(
                    "Narrative net margin % does not match canonical value.",
                )

    if canonical.debt_to_equity is not None and "debt" in joined:
        expected = canonical.format_debt_equity()
        if expected and expected.lower() not in joined:
            raise FinancialMetricsConsistencyError(
                f"Narrative debt/equity does not match canonical display {expected!r}.",
            )

    if canonical.free_cash_flow_latest is not None and "cash flow" in joined:
        expected = canonical.format_free_cash_flow()
        if expected and expected.lower() not in joined:
            compact = expected.replace(",", "").lower()
            if compact not in joined.replace(",", ""):
                raise FinancialMetricsConsistencyError(
                    f"Narrative FCF does not match canonical display {expected!r}.",
                )


_KEY_METRIC_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("revenue", re.compile(r"\brevenue\b|\bgrowth\b", re.I)),
    ("gross margin", re.compile(r"\bgross margin\b", re.I)),
    ("net margin", re.compile(r"\bnet margin\b|\bmargin\b|\bprofitab", re.I)),
    ("debt", re.compile(r"\bdebt\b|\bleverage\b", re.I)),
    ("liquidity", re.compile(r"\bliquidity\b|\bcurrent ratio\b", re.I)),
    ("cash flow", re.compile(r"\bcash flow\b|\bfcf\b|\bfree cash\b", re.I)),
    ("roe", re.compile(r"\breturn on equity\b|\broe\b", re.I)),
    ("payout", re.compile(r"\bpayout\b|\bdividend\b", re.I)),
)

_ALLOWED_OFF_TOPIC = frozenset(
    {
        "execution",
        "scaling",
        "adoption",
        "commercial",
        "capital",
        "regulated",
        "rate base",
        "loan-book",
        "refinancing",
        "property",
        "pipeline",
        "runway",
        "funding",
        "reinvestment",
        "asset-heavy",
        "spread",
        "credit",
        "occupancy",
        "operations",
        "turnaround",
        "financing",
        "capex",
        "balance-sheet",
        "shareholder",
    }
)


def validate_overview_result(
    result: FinancialOverviewResult,
    canonical: CanonicalFinancialMetrics,
    ctx: FinancialCompanyContext,
) -> None:
    texts = [
        result.financial_verdict,
        result.headline,
        *result.strengths,
        *result.risks,
        *result.highlights,
    ]
    _assert_narrative_uses_canonical_values(texts, canonical)
    _assert_no_conflicting_metric_values(texts)
    _assert_profile_matches_breakdown(result.profile, result.score_breakdown)
    _assert_verdict_is_single_sentence(result.financial_verdict)

    pseudo_strength = FinancialStrength(
        profile=result.profile,
        score=result.score,
        financial_verdict=result.financial_verdict,
        score_explanation=result.financial_verdict,
        business_context=result.business_context,
        score_breakdown=result.score_breakdown,
        rating=result.rating,
        headline=result.headline,
        strengths=result.strengths,
        risks=result.risks,
        highlights=result.highlights,
        key_metrics=canonical.to_key_metrics(),
    )
    _assert_strengths_risks_use_key_metrics(pseudo_strength, ctx)


def _assert_verdict_is_single_sentence(verdict: str) -> None:
    cleaned = verdict.strip()
    if not cleaned:
        raise FinancialMetricsConsistencyError("Financial verdict is empty.")
    if cleaned.count("!") > 0 or cleaned.count("?") > 0:
        raise FinancialMetricsConsistencyError(
            "Financial verdict must be a single sentence.",
        )
    if not cleaned.endswith("."):
        raise FinancialMetricsConsistencyError(
            "Financial verdict must end with a period.",
        )
    without_decimals = re.sub(r"\d+\.\d+", lambda match: match.group(0).replace(".", ""), cleaned)
    if without_decimals.count(".") > 1:
        raise FinancialMetricsConsistencyError(
            "Financial verdict must be a single sentence.",
        )


def _assert_profile_matches_breakdown(profile: str, breakdown) -> None:
    scores = (
        breakdown.growth.score,
        breakdown.profitability.score,
        breakdown.cash_flow.score,
        breakdown.balance_sheet.score,
    )
    if profile == "Financially Strong" and max(scores) < 50:
        raise FinancialMetricsConsistencyError(
            "Profile 'Financially Strong' conflicts with weak category scores.",
        )
    if profile == "High Growth / High Risk" and breakdown.growth.score < 45:
        raise FinancialMetricsConsistencyError(
            "Profile 'High Growth / High Risk' conflicts with low growth score.",
        )
    if profile == "Leveraged Turnaround" and (
        breakdown.balance_sheet.score >= 72 and breakdown.profitability.score >= 72
    ):
        raise FinancialMetricsConsistencyError(
            "Profile 'Leveraged Turnaround' conflicts with strong balance sheet and profits.",
        )
    if profile == "Mature Stable Business" and breakdown.growth.score >= 85:
        raise FinancialMetricsConsistencyError(
            "Profile 'Mature Stable Business' conflicts with hypergrowth score.",
        )


def _assert_strengths_risks_use_key_metrics(
    strength: FinancialStrength,
    ctx: FinancialCompanyContext,
) -> None:
    key_labels = {metric.label.lower() for metric in strength.key_metrics}
    allowed_topics: set[str] = set()
    if any("revenue" in label for label in key_labels):
        allowed_topics.add("revenue")
    if any("gross" in label for label in key_labels):
        allowed_topics.add("gross margin")
    if any("net margin" in label for label in key_labels):
        allowed_topics.add("net margin")
    if any("debt" in label for label in key_labels):
        allowed_topics.add("debt")
    if any("current" in label for label in key_labels):
        allowed_topics.add("liquidity")
    if any("cash flow" in label or "fcf" in label for label in key_labels):
        allowed_topics.add("cash flow")
    if any("roe" in label or "return on equity" in label for label in key_labels):
        allowed_topics.add("roe")
    if ctx.allows_dividend_commentary:
        allowed_topics.add("payout")

    for line in [*strength.strengths, *strength.risks]:
        lowered = line.lower()
        if not ctx.allows_dividend_commentary and (
            "payout" in lowered or "dividend" in lowered
        ):
            raise FinancialMetricsConsistencyError(
                "Strengths/risks mention dividends outside dividend-focused sectors.",
            )
        for topic, pattern in _KEY_METRIC_TOPIC_PATTERNS:
            if not pattern.search(line):
                continue
            if topic == "payout" and not ctx.allows_dividend_commentary:
                raise FinancialMetricsConsistencyError(
                    "Strengths/risks mention payout without dividend sector context.",
                )
            if topic in allowed_topics or topic == "net margin":
                continue
            if any(token in lowered for token in _ALLOWED_OFF_TOPIC):
                continue
            if topic == "revenue" and "growth" in allowed_topics:
                continue
            raise FinancialMetricsConsistencyError(
                f"Strength/risk references '{topic}' not present in key metrics: {line!r}",
            )


def _assert_no_conflicting_metric_values(texts: list[str]) -> None:
    joined = " ".join(text for text in texts if text)
    pct_values = [float(match.group(1)) for match in _PCT_RE.finditer(joined)]
    if len(pct_values) >= 2:
        unique = {round(value, 1) for value in pct_values}
        if len(unique) > 4:
            raise FinancialMetricsConsistencyError(
                "Narrative cites too many distinct percentage values — possible conflict.",
            )
