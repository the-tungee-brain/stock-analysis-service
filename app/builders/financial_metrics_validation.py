from __future__ import annotations

import re

from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics
from app.models.company_research_models import FinancialStrength, FundamentalMetric

_PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_RATIO_RE = re.compile(r"(-?\d+(?:\.\d+)?)x", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"-?\$\d+(?:\.\d+)?[BMKT]?|-?\$[\d,]+")


class FinancialMetricsConsistencyError(AssertionError):
    pass


def validate_strength_matches_canonical(
    strength: FinancialStrength,
    canonical: CanonicalFinancialMetrics,
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

    texts = [strength.headline, *strength.strengths, *strength.risks, *strength.highlights]
    _assert_narrative_uses_canonical_values(texts, canonical)


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
