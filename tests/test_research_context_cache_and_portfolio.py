import json
from unittest.mock import MagicMock

from app.adapters.cache.research_context_cache import ResearchContextCache
from app.core.prompts import (
    AnalysisAction,
    PortfolioContext,
    SymbolContext,
    build_portfolio_prompt,
    build_symbol_prompt,
    is_follow_up_affirmation,
)
from app.models.company_research_models import ResearchContext
from app.services.prompt_enrichment_service import PromptEnrichmentService
from tests.test_position_prompt_metrics import _make_account, _make_position


def test_research_context_cache_uses_lookback_in_key():
    redis_client = MagicMock()
    stored: dict[str, str] = {}

    def setex(key, ttl, value):
        stored[key] = value

    def get(key):
        return stored.get(key)

    redis_client.setex = setex
    redis_client.get = get

    cache = ResearchContextCache(redis_client=redis_client, ttl_seconds=900)
    context = ResearchContext(symbol="AAPL", peers=["MSFT"])

    cache.put(symbol="AAPL", context=context, lookback_days=14)
    loaded = cache.get(symbol="AAPL", lookback_days=14)

    assert loaded is not None
    assert loaded.symbol == "AAPL"
    assert "research:context:v3:AAPL:14:news0:press0" in stored
    assert cache.get(symbol="AAPL", lookback_days=7) is None


def test_research_context_cache_roundtrip():
    redis_client = MagicMock()
    stored: dict[str, str] = {}

    def setex(key, ttl, value):
        stored[key] = value

    def get(key):
        return stored.get(key)

    redis_client.setex = setex
    redis_client.get = get

    cache = ResearchContextCache(redis_client=redis_client, ttl_seconds=900)
    context = ResearchContext(
        symbol="AAPL",
        peers=["MSFT", "GOOGL"],
        data_gaps=["news"],
    )

    cache.put(symbol="AAPL", context=context)
    loaded = cache.get(symbol="AAPL")

    assert loaded is not None
    assert loaded.symbol == "AAPL"
    assert loaded.peers == ["MSFT", "GOOGL"]
    assert loaded.data_gaps == ["news"]
    assert json.loads(stored["research:context:v3:AAPL:news0:press0"])["symbol"] == "AAPL"


def test_research_context_cache_separates_news_variants():
    redis_client = MagicMock()
    stored: dict[str, str] = {}

    def setex(key, ttl, value):
        stored[key] = value

    def get(key):
        return stored.get(key)

    redis_client.setex = setex
    redis_client.get = get

    cache = ResearchContextCache(redis_client=redis_client, ttl_seconds=900)
    no_news = ResearchContext(symbol="AAPL", news=[])
    with_news = ResearchContext(symbol="AAPL", news=[])

    cache.put(symbol="AAPL", context=no_news)
    cache.put(symbol="AAPL", context=with_news, include_news=True)

    assert "research:context:v3:AAPL:news0:press0" in stored
    assert "research:context:v3:AAPL:news1:press0" in stored
    assert cache.get(symbol="AAPL", include_news=True) is not None
    assert cache.get(symbol="AAPL", include_press_releases=True) is None


def test_research_chat_followup_message_omits_context_block():
    ctx = ResearchContext(symbol="AAPL")
    message = PromptEnrichmentService().build_research_chat_user_message(
        ctx=ctx,
        user_prompt="What are the main risks?",
        include_context=False,
    )

    assert "RESEARCH DATA FOR AAPL" not in message["content"]
    assert "What are the main risks?" in message["content"]
    assert "earlier in this conversation" in message["content"]


def test_build_symbol_prompt_includes_research_context():
    ctx = SymbolContext(
        symbol="NVDA",
        account=_make_account(),
        positions=[_make_position(symbol="NVDA")],
        research_context="Symbol: NVDA\n## Peer companies\nAMD, INTC",
        action=AnalysisAction.FREE_FORM,
    )

    prompt = build_symbol_prompt(ctx=ctx)

    assert "EQUITY RESEARCH (FUNDAMENTALS, NEWS, SEC)" in prompt
    assert "AMD, INTC" in prompt


def test_is_follow_up_affirmation():
    assert is_follow_up_affirmation("let's do that")
    assert is_follow_up_affirmation("Yes please!")
    assert is_follow_up_affirmation("sure.")
    assert not is_follow_up_affirmation("Should I trim NVDA?")
    assert not is_follow_up_affirmation(None)


def test_symbol_followup_prompt_omits_context_block():
    ctx = SymbolContext(
        symbol="NVDA",
        account=_make_account(),
        positions=[_make_position(symbol="NVDA")],
        market_snapshot="NVDA last: $120",
        action=AnalysisAction.FREE_FORM,
        user_prompt="let's do that",
    )

    prompt = build_symbol_prompt(ctx=ctx, include_context=False)

    assert "ACCOUNT CONTEXT" not in prompt
    assert "let's do that" in prompt
    assert "accepted a follow-up" in prompt
    assert "earlier in this conversation" in prompt


def test_should_include_portfolio_context_without_assistant_history():
    from app.services.chat_service import ChatService

    assert ChatService.should_include_portfolio_context(
        is_first_chat=False,
        action=AnalysisAction.FREE_FORM,
        recent_messages=[{"role": "user", "content": "Should I trim?"}],
    )


def test_should_omit_portfolio_context_on_valid_followup():
    from app.services.chat_service import ChatService

    assert not ChatService.should_include_portfolio_context(
        is_first_chat=False,
        action=AnalysisAction.FREE_FORM,
        recent_messages=[
            {"role": "user", "content": "Should I trim?"},
            {"role": "assistant", "content": "I'd trim 30%. Want redeploy ideas?"},
        ],
        user_prompt="let's do that",
    )


def test_portfolio_followup_prompt_omits_context_block():
    ctx = PortfolioContext(
        account=_make_account(),
        positions=[_make_position(symbol="NVDA")],
        user_prompt="let's do that",
    )

    prompt = build_portfolio_prompt(ctx=ctx, include_context=False)

    assert "PORTFOLIO POSITIONS" not in prompt
    assert "accepted a follow-up" in prompt
    assert "earlier in this conversation" in prompt
