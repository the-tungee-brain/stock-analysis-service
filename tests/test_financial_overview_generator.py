from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics
from app.builders.financial_overview_generator import FinancialOverviewGenerator


def _generate(metrics: CanonicalFinancialMetrics, symbol: str = "TEST") -> tuple:
    result = FinancialOverviewGenerator().generate(symbol, metrics)
    return result.headline, result.strengths, result.risks, result.profile, result.score


def test_hypergrowth_ai_profile_differs_from_mature_utility():
    hyper, hyper_strengths, hyper_risks, hyper_profile, hyper_score = _generate(
        CanonicalFinancialMetrics(
            revenue_growth_yoy=120,
            gross_margin_pct=78,
            net_margin_pct=8,
            debt_to_equity=0.2,
            current_ratio=2.8,
            free_cash_flow_latest=-2_000_000_000,
            free_cash_flow_yoy_pct=-15,
        ),
        symbol="PLTR",
    )
    mature, mature_strengths, mature_risks, mature_profile, mature_score = _generate(
        CanonicalFinancialMetrics(
            revenue_growth_yoy=3,
            gross_margin_pct=42,
            net_margin_pct=22,
            debt_to_equity=0.35,
            current_ratio=1.4,
            free_cash_flow_latest=8_500_000_000,
            free_cash_flow_yoy_pct=6,
            payout_ratio_pct=65,
            fcf_dividend_coverage=1.8,
        ),
        symbol="NEE",
    )

    assert hyper_profile in {"Speculative Growth", "High Growth / High Risk"}
    assert any("exceptional" in s or "strong" in s.lower() for s in hyper_strengths)
    assert any("negative" in r.lower() or "cash" in r.lower() for r in hyper_risks)
    assert mature_profile in {
        "Mature Stable Business",
        "Financially Strong",
        "Cash-Generating Value",
        "Profitable Compounder",
    }
    assert mature_score >= 55
    assert not any("speculative" in s.lower() for s in mature_strengths)
    assert hyper != mature


def test_distressed_profile_prioritizes_margin_and_leverage():
    headline, strengths, risks, profile, score = _generate(
        CanonicalFinancialMetrics(
            revenue_growth_yoy=-5,
            gross_margin_pct=22,
            net_margin_pct=-62,
            debt_to_equity=5.65,
            current_ratio=0.7,
            free_cash_flow_latest=-500_000_000,
            payout_ratio_pct=12,
        ),
        symbol="DIST",
    )

    assert profile in {"Leveraged Turnaround", "High Growth / High Risk"}
    assert score < 45
    assert "margin" in headline.lower() or "debt" in headline.lower()
    assert len(risks) <= 3
    assert risks[0].lower().find("margin") >= 0 or risks[0].lower().find("loss") >= 0
    assert any("leverage" in r.lower() or "balance-sheet" in r.lower() for r in risks)
    assert not any("revenue growth remains strong" in s for s in strengths)
    assert not any("payout" in r.lower() for r in risks[:1])


def test_revenue_growth_conditional_bands():
    exceptional = FinancialOverviewGenerator().generate(
        "X", CanonicalFinancialMetrics(revenue_growth_yoy=150, net_margin_pct=10)
    )
    strong = FinancialOverviewGenerator().generate(
        "X", CanonicalFinancialMetrics(revenue_growth_yoy=35, net_margin_pct=10)
    )
    modest = FinancialOverviewGenerator().generate(
        "X", CanonicalFinancialMetrics(revenue_growth_yoy=8, net_margin_pct=10)
    )
    contracting = FinancialOverviewGenerator().generate(
        "X", CanonicalFinancialMetrics(revenue_growth_yoy=-12, net_margin_pct=10)
    )

    assert any("exceptional" in s for s in exceptional.strengths)
    assert any("strong" in s.lower() for s in strong.strengths)
    assert any("modest" in s for s in modest.strengths)
    assert any("contracting" in r for r in contracting.risks)


def test_low_leverage_skips_leverage_risk():
    _, strengths, risks, _, _ = _generate(
        CanonicalFinancialMetrics(
            debt_to_equity=0.25,
            net_margin_pct=18,
            free_cash_flow_latest=1_000_000_000,
        )
    )
    assert any("conservative" in s for s in strengths)
    assert not any("elevated" in r.lower() or "balance-sheet risk" in r.lower() for r in risks)


def test_no_generic_fallback_copy():
    _, strengths, risks, _, _ = _generate(CanonicalFinancialMetrics())
    banned = (
        "investors should monitor",
        "review the statement tables",
        "risk tolerance",
        "the company faces challenges",
    )
    for text in strengths + risks:
        lowered = text.lower()
        assert not any(phrase in lowered for phrase in banned)
