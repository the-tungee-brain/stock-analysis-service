from app.models.company_research_models import (
    ResearchContext,
    ResearchSnapshot,
    SecRatioTrendPoint,
)
from app.models.sec_research_models import RatioSnapshot
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.sec_research_service import SecResearchService


def test_ratio_snapshot_to_trend_point_formats_values():
    snapshot = RatioSnapshot(
        end="2024-09-28",
        fiscal_period="FY",
        fiscal_year=2024,
        gross_margin=0.45,
        operating_margin=0.30,
        net_margin=0.25,
        roe=1.50,
        fcf_margin=0.28,
        revenue_growth_yoy=0.08,
    )

    trend = SecResearchService._ratio_snapshot_to_trend_point(snapshot)

    assert trend is not None
    assert trend.period_end == "2024-09-28"
    assert trend.fiscal_year == 2024
    assert trend.net_margin == "25.0%"
    assert trend.operating_margin == "30.0%"
    assert trend.roe == "150.0%"
    assert trend.revenue_growth_yoy == "8.0%"
    assert trend.fcf_margin == "28.0%"


def test_ratio_snapshot_to_trend_point_returns_none_when_empty():
    snapshot = RatioSnapshot(
        end="2024-09-28",
        fiscal_period="FY",
        fiscal_year=2024,
    )

    assert SecResearchService._ratio_snapshot_to_trend_point(snapshot) is None


def test_research_context_block_includes_peers_trends_and_gaps():
    ctx = ResearchContext(
        symbol="AAPL",
        snapshot=ResearchSnapshot(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            country="US",
            price=200.0,
            changePct=1.2,
            marketCap="3.0T",
            range52w="$170 – $220",
            weburl="https://apple.com",
            logo="https://example.com/logo.png",
        ),
        peers=["MSFT", "GOOGL", "META"],
        sec_ratio_trends=[
            SecRatioTrendPoint(
                period_end="2024-09-28",
                fiscal_year=2024,
                net_margin="25.0%",
                operating_margin="30.0%",
                roe="150.0%",
                revenue_growth_yoy="8.0%",
                fcf_margin="28.0%",
            )
        ],
        data_gaps=["news"],
    )

    block = PromptEnrichmentService()._format_research_context_block(ctx)

    assert "Peer companies" in block
    assert "MSFT, GOOGL, META" in block
    assert "SEC filed financial trends" in block
    assert "2024-09-28" in block
    assert "25.0%" in block
    assert "news" in block
    assert "Do not invent figures for missing sources" in block
