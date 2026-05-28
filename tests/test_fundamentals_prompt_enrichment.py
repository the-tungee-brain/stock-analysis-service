from app.models.yfinance_analysis_models import (
    AnalystPriceTargets,
    OwnershipSnapshot,
    RecommendationBreakdown,
    StreetAnalysisSnapshot,
)
from app.models.yfinance_funds_models import EtfFundsSnapshot, FundTopHolding, FundWeighting
from app.services.prompt_enrichment_service import PromptEnrichmentService


def test_format_street_analysis_block_includes_consensus_and_ownership():
    street = StreetAnalysisSnapshot(
        consensus_label="Mostly Buy",
        price_targets=AnalystPriceTargets(mean=110.0, low=90.0, high=120.0, upside_to_mean_pct=8.5),
        recommendation=RecommendationBreakdown(
            strong_buy=5, buy=10, hold=3, sell=1, strong_sell=0
        ),
        estimate_drift_headline="Next-quarter EPS consensus drifted up vs 30 days ago.",
        ownership=OwnershipSnapshot(
            insiders_pct_held=0.08,
            institutions_pct_held=65.0,
        ),
    )
    block = PromptEnrichmentService._format_street_analysis_block(street)

    assert "Wall Street consensus" in block
    assert "Mostly Buy" in block
    assert "mean $110.00" in block
    assert "Insiders: 0.08%" in block
    assert "Institutions: 65.00%" in block


def test_format_etf_funds_block_includes_composition():
    funds = EtfFundsSnapshot(
        category="Large Blend",
        expense_ratio_pct=0.09,
        category_expense_ratio_pct=0.11,
        sector_weightings=[
            FundWeighting(label="Technology", weight_pct=32.0),
            FundWeighting(label="Financials", weight_pct=14.0),
        ],
        top_holdings=[
            FundTopHolding(symbol="NVDA", name="NVIDIA", weight_pct=7.1),
        ],
    )
    block = PromptEnrichmentService._format_etf_funds_block(funds)

    assert "ETF fund profile" in block
    assert "Large Blend" in block
    assert "Technology 32.0%" in block
    assert "NVDA" in block
