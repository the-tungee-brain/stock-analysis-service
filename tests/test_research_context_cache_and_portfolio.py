import json
from unittest.mock import MagicMock

from app.adapters.cache.research_context_cache import ResearchContextCache
from app.core.prompts import SymbolContext, build_symbol_prompt, AnalysisAction
from app.models.company_research_models import ResearchContext
from app.services.prompt_enrichment_service import PromptEnrichmentService
from tests.test_position_prompt_metrics import _make_account, _make_position


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
    assert json.loads(stored["research:context:AAPL"])["symbol"] == "AAPL"


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
