from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics
from app.builders.fundamentals_valuation_generator import FundamentalsValuationGenerator
from app.models.company_research_models import (
    FinancialStrength,
    FundamentalMetric,
    ResearchSnapshot,
)
from app.models.yfinance_analysis_models import (
    AnalystPriceTargets,
    StreetAnalysisSnapshot,
)
from tests.financial_strength_fixtures import make_financial_strength


def _snapshot(price: float = 100.0) -> ResearchSnapshot:
    return ResearchSnapshot(
        symbol="TEST",
        name="Test Co",
        sector="Technology",
        country="US",
        price=price,
        changePct=1.0,
        marketCap="$10B",
        weburl="https://example.com",
    )


def test_valuation_overview_has_thesis_and_summary():
    street = StreetAnalysisSnapshot(
        price_targets=AnalystPriceTargets(
            current=100.0,
            mean=120.0,
            upside_to_mean_pct=20.0,
        ),
        consensus_label="Buy",
    )
    result = FundamentalsValuationGenerator().generate(
        symbol="TEST",
        snapshot=_snapshot(),
        canonical=CanonicalFinancialMetrics(
            revenue_growth_yoy=25.0,
            net_margin_pct=18.0,
            free_cash_flow_latest=1_000_000_000,
        ),
        strength=make_financial_strength(profile="Profitable Compounder", score=72),
        street=street,
        metrics=[
            FundamentalMetric(label="P/E (trailing)", value="22.5x", note=None),
            FundamentalMetric(label="P/E (forward)", value="18.0x", note=None),
        ],
        sector="Technology",
    )

    assert result.valuation_summary
    assert len(result.investment_thesis.bull_case) <= 3
    assert len(result.investment_thesis.bear_case) <= 3
    assert any("target" in bullet.lower() for bullet in result.investment_thesis.bull_case)
    assert "atAGlance" not in result.model_dump(by_alias=True)


def test_premium_to_target_surfaces_bear_case():
    street = StreetAnalysisSnapshot(
        price_targets=AnalystPriceTargets(
            current=150.0,
            mean=120.0,
            upside_to_mean_pct=-25.0,
        ),
    )
    result = FundamentalsValuationGenerator().generate(
        symbol="TEST",
        snapshot=_snapshot(price=150.0),
        canonical=CanonicalFinancialMetrics(revenue_growth_yoy=-3.0),
        strength=make_financial_strength(profile="High Growth / High Risk", score=45),
        street=street,
        metrics=[FundamentalMetric(label="P/E (trailing)", value="45.0x", note=None)],
    )
    assert any(
        "above" in bullet.lower() or "target" in bullet.lower()
        for bullet in result.investment_thesis.bear_case
    )
