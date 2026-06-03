from app.models.company_research_models import (
    FinancialCategoryScore,
    FinancialScoreBreakdown,
    FinancialStrength,
    FundamentalMetric,
)


def _cat(score: int) -> FinancialCategoryScore:
    from app.builders.financial_score_percentiles import rank_label_for_score

    return FinancialCategoryScore(score=score, rank_label=rank_label_for_score(score))


def make_financial_strength(
    *,
    profile: str = "Financially Strong",
    score: int = 70,
    financial_verdict: str = (
        "Solid growth and cash generation support a high-quality financial profile."
    ),
    score_explanation: str | None = None,
    score_breakdown: FinancialScoreBreakdown | None = None,
    rating: str = "solid",
    headline: str = "TEST — snapshot",
    strengths: list[str] | None = None,
    risks: list[str] | None = None,
    highlights: list[str] | None = None,
    key_metrics: list[FundamentalMetric] | None = None,
) -> FinancialStrength:
    verdict = score_explanation or financial_verdict
    return FinancialStrength(
        profile=profile,
        score=score,
        financial_verdict=verdict,
        score_explanation=verdict,
        score_breakdown=score_breakdown
        or FinancialScoreBreakdown(
            growth=_cat(70),
            profitability=_cat(70),
            balance_sheet=_cat(70),
            cash_flow=_cat(70),
        ),
        rating=rating,
        headline=headline,
        strengths=strengths or [],
        risks=risks or [],
        highlights=highlights or [],
        key_metrics=key_metrics or [],
    )
