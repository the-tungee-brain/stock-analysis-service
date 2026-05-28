from app.models.company_research_models import (
    FinancialStatementsSnapshot,
    FinancialStrength,
    FinancialsPackage,
    FundamentalMetric,
    ResearchContext,
)
from app.services.prompt_enrichment_service import PromptEnrichmentService


def test_format_dividend_payout_section_includes_payout_and_fcf_coverage():
    ctx = ResearchContext(
        symbol="KO",
        fundamentals=[
            FundamentalMetric(label="Payout ratio", value="75.0%"),
            FundamentalMetric(label="Dividend yield", value="3.10%"),
            FundamentalMetric(label="Annual dividend per share", value="$1.84"),
        ],
        yfinance_financials=FinancialsPackage(
            strength=FinancialStrength(
                rating="solid",
                score=70,
                headline="Solid for KO.",
                highlights=[
                    "Payout ratio about 75%.",
                    "Free cash flow covers dividends about 2.6x (38% of FCF paid out).",
                ],
            ),
            annual=FinancialStatementsSnapshot(periods=["2025-12-31"]),
        ),
    )

    block = PromptEnrichmentService._format_dividend_payout_section(ctx)

    assert block is not None
    assert "## Dividend & payout" in block
    assert "Payout ratio: 75.0%" in block
    assert "Dividend yield: 3.10%" in block
    assert "covers dividends about 2.6x" in block


def test_cached_context_is_stale_without_yfinance_financials():
    from app.services.company_research_service import CompanyResearchService

    cached = ResearchContext(symbol="KO", asset_type="STOCK")
    assert CompanyResearchService._cached_context_is_stale(cached) is True


def test_cached_context_is_stale_without_payout_when_yield_present():
    from app.services.company_research_service import CompanyResearchService

    cached = ResearchContext(
        symbol="KO",
        asset_type="STOCK",
        fundamentals=[
            FundamentalMetric(label="Dividend yield", value="3.10%"),
        ],
        yfinance_financials=FinancialsPackage(
            strength=FinancialStrength(
                rating="solid",
                score=70,
                headline="Solid for KO.",
            ),
        ),
    )
    assert CompanyResearchService._cached_context_is_stale(cached) is True
