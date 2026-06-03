from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics
from app.builders.financial_overview_generator import (
    BALANCE_SHEET_WEIGHT,
    CASH_FLOW_WEIGHT,
    GROWTH_WEIGHT,
    PROFITABILITY_WEIGHT,
    FinancialOverviewGenerator,
)


def test_overall_score_matches_weighted_breakdown():
    canonical = CanonicalFinancialMetrics(
        revenue_growth_yoy=35.0,
        gross_margin_pct=65.0,
        net_margin_pct=18.0,
        debt_to_equity=0.4,
        current_ratio=1.8,
        free_cash_flow_latest=3_000_000_000,
        free_cash_flow_yoy_pct=8.0,
    )
    result = FinancialOverviewGenerator().generate("TEST", canonical)
    expected = round(
        GROWTH_WEIGHT * result.score_breakdown.growth
        + PROFITABILITY_WEIGHT * result.score_breakdown.profitability
        + CASH_FLOW_WEIGHT * result.score_breakdown.cash_flow
        + BALANCE_SHEET_WEIGHT * result.score_breakdown.balance_sheet
    )
    assert result.score == expected
    assert result.profile
    assert result.score_explanation.endswith(".")


def test_high_score_not_labeled_turnaround():
    canonical = CanonicalFinancialMetrics(
        revenue_growth_yoy=40.0,
        net_margin_pct=20.0,
        debt_to_equity=0.3,
        free_cash_flow_latest=5_000_000_000,
        current_ratio=2.0,
    )
    result = FinancialOverviewGenerator().generate("TEST", canonical)
    assert result.score >= 60
    assert result.profile not in {"Leveraged Turnaround", "Speculative Growth"}
