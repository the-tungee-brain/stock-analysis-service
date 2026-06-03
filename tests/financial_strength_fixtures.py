from app.models.company_research_models import (
    FinancialScoreBreakdown,
    FinancialStrength,
    FundamentalMetric,
)


def make_financial_strength(
    *,
    profile: str = "Financially Strong",
    score: int = 70,
    score_explanation: str = "Solid growth and cash generation support the overall financial score.",
    score_breakdown: FinancialScoreBreakdown | None = None,
    rating: str = "solid",
    headline: str = "TEST — snapshot",
    strengths: list[str] | None = None,
    risks: list[str] | None = None,
    highlights: list[str] | None = None,
    key_metrics: list[FundamentalMetric] | None = None,
) -> FinancialStrength:
    return FinancialStrength(
        profile=profile,
        score=score,
        score_explanation=score_explanation,
        score_breakdown=score_breakdown
        or FinancialScoreBreakdown(
            growth=70,
            profitability=70,
            balance_sheet=70,
            cash_flow=70,
        ),
        rating=rating,
        headline=headline,
        strengths=strengths or [],
        risks=risks or [],
        highlights=highlights or [],
        key_metrics=key_metrics or [],
    )
