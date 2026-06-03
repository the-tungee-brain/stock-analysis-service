from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics
from app.builders.fundamentals_valuation_generator import FundamentalsValuationGenerator
from app.models.company_research_models import (
    FinancialStrength,
    FundamentalMetric,
    ResearchSnapshot,
)
from app.models.yfinance_analysis_models import (
    AnalystPriceTargets,
    PeriodEstimate,
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


def test_bull_case_uses_fundamentals_not_analyst_ratings():
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
            gross_margin_pct=65.0,
            net_margin_pct=18.0,
            free_cash_flow_latest=1_000_000_000,
            free_cash_flow_yoy_pct=12.0,
        ),
        strength=make_financial_strength(profile="Profitable Compounder", score=72),
        street=street,
        metrics=[
            FundamentalMetric(label="P/E (trailing)", value="22.5x", note=None),
            FundamentalMetric(label="Price / book", value="8.2x", note=None),
            FundamentalMetric(label="EPS (trailing)", value="$4.50", note=None),
        ],
        sector="Technology",
    )

    assert result.valuation_conclusion
    assert result.valuation_signals
    assert any(signal.label == "Revenue growth" for signal in result.valuation_signals)

    bull_text = " ".join(result.investment_thesis.bull_case).lower()
    assert "consensus" not in bull_text
    assert "wall street" not in bull_text
    assert "analyst target" not in bull_text
    assert any(
        token in bull_text
        for token in ("revenue", "margin", "cash flow", "momentum", "demand")
    )


def test_bear_case_includes_target_gap_and_multiples():
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
        canonical=CanonicalFinancialMetrics(
            revenue_growth_yoy=-3.0,
            net_margin_pct=-5.0,
            debt_to_equity=3.5,
        ),
        strength=make_financial_strength(profile="High Growth / High Risk", score=45),
        street=street,
        metrics=[FundamentalMetric(label="P/E (trailing)", value="45.0x", note=None)],
    )
    bear_text = " ".join(result.investment_thesis.bear_case).lower()
    assert any(
        token in bear_text
        for token in ("target", "p/e", "margin", "execution", "debt")
    )
    assert "sell" not in bear_text or "strong sell" not in bear_text


def test_premium_hypergrowth_conclusion():
    result = FundamentalsValuationGenerator().generate(
        symbol="NVDA",
        snapshot=_snapshot(price=200.0),
        canonical=CanonicalFinancialMetrics(
            revenue_growth_yoy=80.0,
            net_margin_pct=3.0,
        ),
        strength=make_financial_strength(profile="High Growth / High Risk", score=55),
        street=StreetAnalysisSnapshot(
            price_targets=AnalystPriceTargets(current=200.0, mean=190.0, upside_to_mean_pct=-5.0)
        ),
        metrics=[FundamentalMetric(label="P/E (trailing)", value="55.0x", note=None)],
    )
    assert "premium" in result.valuation_conclusion.lower() or "hypergrowth" in result.valuation_conclusion.lower()


def test_street_context_is_supporting_not_core():
    street = StreetAnalysisSnapshot(
        consensus_label="Buy",
        price_targets=AnalystPriceTargets(current=100.0, mean=110.0, upside_to_mean_pct=9.0),
    )
    result = FundamentalsValuationGenerator().generate(
        symbol="TEST",
        snapshot=_snapshot(),
        canonical=CanonicalFinancialMetrics(revenue_growth_yoy=10.0, net_margin_pct=15.0),
        strength=None,
        street=street,
        metrics=[],
    )
    assert "supporting" in result.street_context.lower() or "cross-check" in result.street_context.lower()
