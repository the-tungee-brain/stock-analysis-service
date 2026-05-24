from app.adapters.cache.llm_output_cache import LLMOutputCache
from app.core.llm_routes import LLMRoute
from app.models.company_research_models import (
    EarningsContext,
    NewsHeadline,
    ResearchContext,
    ResearchSnapshot,
    SecRatioTrendPoint,
)
from app.services.prompt_enrichment_service import PromptEnrichmentService


def test_llm_output_cache_roundtrip():
    from unittest.mock import MagicMock

    redis_client = MagicMock()
    stored: dict[str, str] = {}

    redis_client.setex = lambda key, ttl, value: stored.update({key: value})
    redis_client.get = lambda key: stored.get(key)

    cache = LLMOutputCache(redis_client=redis_client, ttl_seconds=3600)
    cache.put(
        route=LLMRoute.SUMMARY,
        symbol="AAPL",
        fingerprint="abc123",
        payload='{"short":"hello"}',
    )

    assert cache.get(route=LLMRoute.SUMMARY, symbol="AAPL", fingerprint="abc123") == '{"short":"hello"}'


def test_compact_context_is_shorter_than_full():
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
        news=[
            NewsHeadline(
                headline=f"Headline {idx}",
                summary=f"Summary {idx}",
                source="Reuters",
                datetime="2026-05-20",
            )
            for idx in range(10)
        ],
        sec_ratio_trends=[
            SecRatioTrendPoint(
                period_end=f"202{idx}-09-28",
                net_margin="25.0%",
            )
            for idx in range(5)
        ],
        earnings=EarningsContext(
            upcoming_report_date="2026-07-30",
            upcoming_fiscal_period="Q3 2026",
            upcoming_timing="amc",
            last_report_date="2026-04-25",
            last_fiscal_period="Q2 2026",
            last_beat_label="beat",
            last_eps_surprise_pct="+4.2%",
        ),
    )

    service = PromptEnrichmentService()
    full = service.format_research_context_block(ctx=ctx, compact=False)
    compact = service.format_research_context_block(ctx=ctx, compact=True)

    assert len(compact) < len(full)
    assert "Earnings calendar" in full
    assert "Earnings calendar" in compact
    assert "Headline 9" in full
    assert "Headline 9" not in compact


def test_tax_angle_context_omits_news():
    ctx = ResearchContext(
        symbol="AAPL",
        news=[
            NewsHeadline(
                headline="Big news",
                summary="Details",
                source="CNBC",
                datetime="2026-05-20",
            )
        ],
        earnings=EarningsContext(upcoming_report_date="2026-07-30"),
    )

    from app.core.prompts import AnalysisAction

    block = PromptEnrichmentService().format_research_context_block(
        ctx=ctx,
        compact=True,
        action=AnalysisAction.TAX_ANGLE,
    )

    assert "Earnings calendar" in block
    assert "Big news" not in block
