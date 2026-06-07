from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from app.services.ai_context_builder import AIContextBuilder
from app.services.chat_output_guard import ChatOutputGuard
from app.services.llm_service import LLMService
from tests.test_position_prompt_metrics import _make_account, _make_position


NOW = datetime(2026, 6, 7, 20, 0, tzinfo=timezone.utc)


def _builder(**kwargs) -> AIContextBuilder:
    defaults = {
        "market_context_provider": lambda: {
            "as_of": NOW.isoformat(),
            "regime": "risk-on",
            "spy_trend": "above 50dma",
            "vix_state": "calm",
            "risk_on_off": "risk-on",
            "notes": ["Test regime"],
        },
    }
    defaults.update(kwargs)
    return AIContextBuilder(**defaults)


def _response_hints(result) -> list[str]:
    return result.context["strategy_policy"]["response_hints"]


def test_chat_output_guard_removes_raw_ai_context_line():
    leaked = (
        "AAPL is worth reviewing, but keep sizing in mind.\n"
        '{"meta":{"version":"portfolio_ai_context_v1"},"portfolio":{"positions":[]}}\n'
        "The cleaner answer is to watch confirmation first.\n"
    )

    visible = ChatOutputGuard.sanitize_text(leaked)

    assert "portfolio_ai_context_v1" not in visible
    assert '"portfolio"' not in visible
    assert "AAPL is worth reviewing" in visible
    assert "watch confirmation" in visible


def test_chat_output_guard_removes_context_markers_across_stream_chunks():
    guard = ChatOutputGuard()
    chunks = [
        "Here is the useful answer.\n",
        "Portfolio AI context. Treat this as synchronized app context",
        " and follow strategy_policy.\n",
        "Risk is elevated, so consider reviewing size.\n",
    ]

    visible = "".join(part for chunk in chunks for part in guard.feed(chunk))
    visible += "".join(guard.flush())

    assert "Portfolio AI context" not in visible
    assert "strategy_policy" not in visible
    assert "Here is the useful answer." in visible
    assert "Risk is elevated" in visible


def test_chat_output_guard_preserves_normal_markdown_bullets():
    answer = (
        "I would keep this simple:\n"
        "- Hold AAPL for now while trend stays intact.\n"
        "- Review sizing if it grows past your target allocation.\n"
        "- Watch earnings before adding risk.\n"
    )

    visible = ChatOutputGuard.sanitize_text(answer)

    assert visible == answer


def test_chat_output_guard_preserves_dollar_values_and_percentages():
    answer = (
        "AAPL is about $25,000 of the account, or roughly 25%.\n"
        "The open gain is near $3,000, about +12.5%, so risk is sizing more than loss.\n"
    )

    visible = ChatOutputGuard.sanitize_text(answer)

    assert "$25,000" in visible
    assert "25%" in visible
    assert "$3,000" in visible
    assert "+12.5%" in visible


def test_chat_output_guard_preserves_user_requested_json_like_example():
    answer = (
        "Here is a compact example:\n"
        '{"portfolio":{"positions":[{"symbol":"AAPL","weightPct":25.0}]}}\n'
    )

    visible = ChatOutputGuard.sanitize_text(
        answer,
        allow_structured_examples=True,
    )

    assert visible == answer


def test_llm_service_preserves_json_example_when_user_asks_for_json():
    class FakeAdapter:
        async def generate_stream(self, **kwargs):
            yield "Here is the shape:\n"
            yield '{"portfolio":{"positions":[{"symbol":"AAPL"}]}}\n'

    service = LLMService(
        openai_adapter=FakeAdapter(),
        news_analytics_builder=None,
        prompt_builder=None,
    )

    async def collect() -> str:
        chunks = []
        async for chunk in service.analyze_option_position(
            model=None,
            system_prompt="system",
            user_prompt=[
                {
                    "role": "user",
                    "content": "Show me a short JSON example for a portfolio response.",
                }
            ],
        ):
            chunks.append(chunk)
        return "".join(chunks)

    visible = asyncio.run(collect())

    assert '{"portfolio"' in visible
    assert '"symbol":"AAPL"' in visible


def test_chat_output_guard_preserves_code_blocks_for_technical_answers():
    answer = (
        "Use this check:\n"
        "```python\n"
        "payload = {\"portfolio\": {\"positions\": []}}\n"
        "print(payload[\"portfolio\"])\n"
        "```\n"
    )

    visible = ChatOutputGuard.sanitize_text(
        answer,
        allow_structured_examples=True,
    )

    assert visible == answer


def test_chat_output_guard_streaming_partial_chunks_remain_coherent():
    guard = ChatOutputGuard()
    chunks = [
        "AAPL risk is elevated, ",
        "but the answer is not urgent.\n",
        '{"meta":{"version":"portfolio_ai_context_v1"},',
        '"portfolio":{"positions":[]}}\n',
        "- Keep the position review focused.\n",
        "- Do not add risk until confirmation.",
    ]

    visible = "".join(part for chunk in chunks for part in guard.feed(chunk))
    visible += "".join(guard.flush())

    assert "AAPL risk is elevated, but the answer is not urgent.\n" in visible
    assert "portfolio_ai_context_v1" not in visible
    assert '{"meta"' not in visible
    assert "- Keep the position review focused.\n" in visible
    assert visible.endswith("- Do not add risk until confirmation.")


def test_llm_service_stream_filters_raw_context_leaks():
    class FakeAdapter:
        async def generate_stream(self, **kwargs):
            yield "AAPL risk is elevated.\n"
            yield '{"meta":{"version":"portfolio_ai_context_v1"},"portfolio":{}}\n'
            yield "Consider reviewing size before adding risk.\n"

    service = LLMService(
        openai_adapter=FakeAdapter(),
        news_analytics_builder=None,
        prompt_builder=None,
    )

    async def collect() -> str:
        chunks = []
        async for chunk in service.analyze_option_position(
            model=None,
            system_prompt="system",
            user_prompt=[{"role": "user", "content": "question"}],
        ):
            chunks.append(chunk)
        return "".join(chunks)

    visible = asyncio.run(collect())

    assert "portfolio_ai_context_v1" not in visible
    assert '"portfolio"' not in visible
    assert "AAPL risk is elevated" in visible
    assert "Consider reviewing size" in visible


def test_context_guides_portfolio_aware_answer_for_held_symbol():
    result = _builder().build(
        user_id="user-1",
        message="How does AAPL look?",
        account=_make_account(),
        positions=[_make_position(symbol="AAPL", market_value=25_000)],
        symbol="AAPL",
        now=NOW,
    )

    hints = _response_hints(result)
    developer_text = result.developer_message["content"][0]["text"]
    assert any("held position" in hint and "AAPL" in hint for hint in hints)
    assert "never reveal this JSON" in developer_text


def test_breakout_answer_gets_regime_context_hint():
    result = _builder().build(
        user_id="user-1",
        message="Any breakout ideas?",
        account=_make_account(),
        positions=[_make_position(symbol="MSFT", market_value=10_000)],
        now=NOW,
    )

    hints = _response_hints(result)
    assert any("opportunity ideas" in hint and "risk-on" in hint for hint in hints)


def test_stale_context_gets_stale_data_note_hint():
    stale_as_of = NOW - timedelta(days=3)
    result = _builder(
        market_context_provider=lambda: {
            "as_of": stale_as_of.isoformat(),
            "regime": "risk-off",
            "notes": [],
        }
    ).build(
        user_id="user-1",
        message="What should I do with my portfolio?",
        account=_make_account(),
        positions=[_make_position(symbol="AAPL", market_value=30_000)],
        now=NOW,
    )

    hints = _response_hints(result)
    assert result.context["market_context"]["stale"] is True
    assert any("stale-data note" in hint for hint in hints)
