from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock
from app.core.prompts import AnalysisAction
from app.models.company_research_models import (
    EarningsContext,
    EnrichedNewsSummary,
    NewsHeadline,
    ResearchContext,
    ResearchSnapshot,
)
from app.models.intelligence_models import (
    IntelligenceSignal,
    PeerComparison,
    PeerMetric,
    SymbolIntelligence,
)
from app.models.schwab_option_chain_models import OptionChain, OptionContract
from app.services.intelligence.event_timeline_builder import EventTimelineBuilder
from app.services.intelligence.options_scoring_service import OptionsScoringService
from app.services.intelligence.signal_engine import SignalEngine
from app.services.prompt_enrichment_service import PromptEnrichmentService
from tests.test_order_activity import _make_option_order
from tests.test_position_prompt_metrics import _make_account, _make_position


def _research_context(**overrides) -> ResearchContext:
    base = ResearchContext(
        symbol="AAPL",
        snapshot=ResearchSnapshot(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            country="US",
            price=200.0,
            changePct=1.0,
            marketCap="3.0T",
            range52w="$170 – $220",
            weburl="https://apple.com",
            logo="https://example.com/logo.png",
        ),
        earnings=EarningsContext(
            upcoming_report_date=(date.today()).strftime("%Y-%m-%d"),
            upcoming_fiscal_period="Q2",
            upcoming_timing="amc",
        ),
        enriched_news=EnrichedNewsSummary(
            overall_sentiment="bearish",
            summary="Mixed demand signals.",
            insights=["Services growth slowed"],
            risks=["China headwinds"],
            dominant_driver="iPhone demand",
            actionability_score=3,
            investor_takeaway="Watch China sales.",
        ),
    )
    return base.model_copy(update=overrides)


def test_signal_engine_flags_earnings_and_concentration():
    ctx = _research_context()
    account = _make_account(liquidation_value=100_000)
    position = _make_position(symbol="AAPL", market_value=35_000)

    signals = SignalEngine.build_symbol_signals(
        research=ctx,
        positions=[position],
        account=account,
        symbol="AAPL",
    )

    kinds = {signal.kind for signal in signals}
    assert "earnings" in kinds
    assert "position_size" in kinds or "concentration" in kinds


def test_event_timeline_skips_news_when_enriched_summary_present():
    ctx = _research_context(
        news=[
            NewsHeadline(
                headline="Apple unveils new AI features",
                summary="Product launch event.",
                source="Reuters",
                datetime=datetime.now(timezone.utc).isoformat(),
            )
        ]
    )

    timeline = EventTimelineBuilder.build(research=ctx, orders=[])

    assert any(entry.kind == "earnings" for entry in timeline)
    assert not any(entry.kind == "news" for entry in timeline)


def test_event_timeline_includes_news_url_when_enriched_summary_missing():
    ctx = _research_context(
        enriched_news=None,
        news=[
            NewsHeadline(
                headline="NVIDIA raises guidance",
                summary="Strong AI demand.",
                source="Reuters",
                datetime=datetime.now(timezone.utc).isoformat(),
                url="https://example.com/nvda-news",
            )
        ],
    )

    timeline = EventTimelineBuilder.build(research=ctx, orders=[])

    news_entries = [entry for entry in timeline if entry.kind == "news"]
    assert news_entries
    assert news_entries[0].url == "https://example.com/nvda-news"


def test_event_timeline_includes_trade_fill_price_from_execution_legs():
    ctx = _research_context()
    order = _make_option_order(underlying="NVDA")

    timeline = EventTimelineBuilder.build(research=ctx, orders=[order])

    trade_entries = [entry for entry in timeline if entry.kind == "trade"]
    assert trade_entries
    assert trade_entries[0].detail is not None
    assert "@ $2.50" in trade_entries[0].detail


def test_options_scoring_ranks_liquid_strikes():
    chain = OptionChain(
        symbol="AAPL",
        underlyingPrice=200.0,
        callExpDateMap={
            "2026-06-20:10": {
                "210.0": [
                    OptionContract(
                        putCall="CALL",
                        symbol="AAPL",
                        strikePrice=210.0,
                        expirationDate="2026-06-20",
                        daysToExpiration=30,
                        delta=0.25,
                        openInterest=1200,
                        bidPrice=2.5,
                        askPrice=2.7,
                        volatility=22.0,
                    )
                ]
            }
        },
        putExpDateMap={
            "2026-06-20:10": {
                "190.0": [
                    OptionContract(
                        putCall="PUT",
                        symbol="AAPL",
                        strikePrice=190.0,
                        expirationDate="2026-06-20",
                        daysToExpiration=30,
                        delta=-0.22,
                        openInterest=900,
                        bidPrice=2.1,
                        askPrice=2.3,
                        volatility=24.0,
                    )
                ]
            }
        },
    )

    scorecard = OptionsScoringService.build_scorecard(chain)

    assert scorecard is not None
    assert scorecard.covered_call_candidates
    assert scorecard.csp_candidates
    assert scorecard.covered_call_candidates[0].strike == 210.0


def test_symbol_intelligence_serializes_camel_case_aliases():
    intelligence = SymbolIntelligence(
        symbol="AAPL",
        signals=[
            IntelligenceSignal(
                kind="earnings",
                severity="watch",
                message="Earnings next week",
                symbol="AAPL",
            )
        ],
        peer_comparison=PeerComparison(
            target_symbol="AAPL",
            target_one_year_return="+12.0%",
            peers=[],
        ),
        data_gaps=["news"],
        partial=False,
    )

    payload = intelligence.model_dump(mode="json", by_alias=True)

    assert "peerComparison" in payload
    assert "eventTimeline" in payload
    assert "optionsScorecard" in payload
    assert "optionChainPreview" in payload
    assert "cachedResearch" in payload
    assert "dataGaps" in payload
    assert payload["peerComparison"]["targetSymbol"] == "AAPL"
    assert "peer_comparison" not in payload


def test_format_intelligence_block_renders_peer_table():
    intelligence = SymbolIntelligence(
        symbol="AAPL",
        peer_comparison=PeerComparison(
            target_symbol="AAPL",
            target_one_year_return="+12.0%",
            target_pe_trailing="28.0x",
            peers=[
                PeerMetric(
                    symbol="MSFT",
                    one_year_return="+15.0%",
                    pe_trailing="35.0x",
                    sector="Technology",
                )
            ],
            summary="AAPL underperformed peer median 1Y return by 3.0pp.",
        ),
    )

    block = PromptEnrichmentService.format_intelligence_block(intelligence)

    assert "Peer comparison" in block
    assert "MSFT" in block
    assert "underperformed" in block


def test_research_context_block_includes_enriched_news():
    ctx = _research_context()
    block = PromptEnrichmentService()._format_research_context_block(
        ctx, action=AnalysisAction.DAILY_SUMMARY
    )

    assert "AI news analysis" in block
    assert "iPhone demand" in block
    assert "Recent news headlines" not in block


def test_research_context_block_uses_raw_news_without_enriched_summary():
    ctx = _research_context(
        enriched_news=None,
        news=[
            NewsHeadline(
                headline="Apple supplier update",
                summary="Supply chain note.",
                source="Reuters",
                datetime="2026-05-20",
            )
        ],
    )
    block = PromptEnrichmentService()._format_research_context_block(
        ctx, action=AnalysisAction.DAILY_SUMMARY
    )

    assert "Recent news headlines" in block
    assert "Apple supplier update" in block
    assert "AI news analysis" not in block


def test_build_symbol_prompt_includes_intelligence_section():
    from app.core.prompts import SymbolContext, build_symbol_prompt

    ctx = SymbolContext(
        symbol="AAPL",
        account=_make_account(),
        positions=[_make_position(symbol="AAPL")],
        intelligence_block="## Precomputed signals\n- [WARNING] Earnings tomorrow",
    )

    prompt = build_symbol_prompt(ctx)

    assert "PRECOMPUTED INTELLIGENCE" in prompt
    assert "Earnings tomorrow" in prompt
