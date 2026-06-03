from app.builders.canonical_financial_metrics import (
    CanonicalFinancialMetrics,
    apply_canonical_key_metrics,
    build_canonical_metrics,
    merge_key_metrics_into_list,
)
from app.builders.financial_metrics_validation import (
    FinancialMetricsConsistencyError,
    validate_key_metrics_match_canonical,
    validate_strength_matches_canonical,
)
from app.builders.financial_overview_generator import FinancialOverviewGenerator
from app.models.company_research_models import FinancialStrength, FundamentalMetric


def _canonical(**kwargs) -> CanonicalFinancialMetrics:
    return CanonicalFinancialMetrics(**kwargs)


def test_key_metrics_and_narrative_share_revenue_growth_display():
    canonical = _canonical(
        revenue_growth_yoy=35.2,
        gross_margin_pct=78.0,
        net_margin_pct=8.1,
        debt_to_equity=0.22,
        current_ratio=2.4,
        free_cash_flow_latest=-2_000_000_000,
        free_cash_flow_yoy_pct=-12.0,
    )
    overview = FinancialOverviewGenerator().generate("PLTR", canonical)
    strength = FinancialStrength(
        rating=overview.rating,
        score=overview.score,
        headline=overview.headline,
        strengths=overview.strengths,
        risks=overview.risks,
        highlights=overview.highlights,
        key_metrics=canonical.to_key_metrics(),
    )

    validate_strength_matches_canonical(strength, canonical)

    assert canonical.format_revenue_growth() in strength.strengths[0]
    fcf_display = (canonical.format_free_cash_flow() or "").lower()
    assert fcf_display in " ".join(strength.risks).lower()


def test_merge_key_metrics_replaces_stale_profit_margin_row():
    canonical = _canonical(
        revenue_growth_yoy=12.5,
        net_margin_pct=18.3,
        gross_margin_pct=55.0,
        debt_to_equity=0.4,
        current_ratio=1.6,
        free_cash_flow_latest=4_000_000_000,
    )
    stale = [
        FundamentalMetric(label="Profit margin", value="9.9%", note="stale"),
        FundamentalMetric(label="Revenue growth", value="1.0%", note="stale"),
        FundamentalMetric(label="P/E (trailing)", value="25.0x", note="keep"),
    ]
    merged = merge_key_metrics_into_list(stale, canonical.to_key_metrics())
    validate_key_metrics_match_canonical(merged, canonical)

    labels = {metric.label for metric in merged}
    assert "Profit margin" not in labels
    assert "Net margin" in labels
    assert merged[0].value == canonical.format_revenue_growth()


def test_distressed_profile_debt_display_matches_key_metric():
    canonical = _canonical(
        revenue_growth_yoy=-8.0,
        net_margin_pct=-62.0,
        debt_to_equity=5.65,
        free_cash_flow_latest=-500_000_000,
        current_ratio=0.7,
    )
    overview = FinancialOverviewGenerator().generate("DIST", canonical)
    strength = FinancialStrength(
        rating=overview.rating,
        score=overview.score,
        headline=overview.headline,
        strengths=overview.strengths,
        risks=overview.risks,
        highlights=overview.highlights,
        key_metrics=canonical.to_key_metrics(),
    )
    validate_strength_matches_canonical(strength, canonical)

    debt_display = canonical.format_debt_equity()
    assert debt_display is not None
    assert debt_display in strength.headline.lower() or any(
        debt_display.lower() in risk.lower() for risk in strength.risks
    )


def test_verdict_uses_weighted_categories_not_balance_sheet_alone():
    canonical = _canonical(
        revenue_growth_yoy=45.0,
        net_margin_pct=22.0,
        debt_to_equity=4.8,
        free_cash_flow_latest=6_000_000_000,
        current_ratio=0.9,
    )
    verdict = FinancialOverviewGenerator()._derive_verdict_weighted(canonical)
    assert verdict in {
        "Profitable Compounder",
        "Financially Strong",
        "High Growth / High Risk",
    }
    assert verdict != "Leveraged Turnaround"


def test_mismatching_key_metric_values_fail_validation():
    canonical = _canonical(revenue_growth_yoy=20.0, net_margin_pct=12.0)
    strength = FinancialStrength(
        rating="solid",
        score=60,
        headline="Test",
        key_metrics=[
            FundamentalMetric(label="Revenue growth", value="99.9%", note=None),
        ],
    )
    try:
        validate_strength_matches_canonical(strength, canonical)
    except FinancialMetricsConsistencyError:
        return
    raise AssertionError("Expected consistency validation to fail")


def test_apply_canonical_key_metrics_is_idempotent():
    canonical = _canonical(
        revenue_growth_yoy=5.5,
        net_margin_pct=14.0,
        free_cash_flow_latest=1_000_000,
    )
    base = [FundamentalMetric(label="Beta", value="1.1", note=None)]
    once = apply_canonical_key_metrics(base, canonical)
    twice = apply_canonical_key_metrics(once, canonical)
    assert once == twice
