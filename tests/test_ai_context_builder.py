from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.services.ai_context_builder import AIContextBuilder
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


def test_aapl_question_includes_owned_position_and_symbol_intelligence():
    calls = []

    def intelligence_provider(**kwargs):
        calls.append(kwargs["symbol"])
        return {
            "symbol": kwargs["symbol"],
            "signals": [
                {
                    "label": "Trend",
                    "severity": "info",
                    "message": "Price trend is constructive.",
                }
            ],
        }

    account = _make_account()
    result = _builder(symbol_intelligence_provider=intelligence_provider).build(
        user_id="user-1",
        message="How does AAPL look here?",
        account=account,
        positions=[_make_position(symbol="AAPL", market_value=25_000)],
        symbol="AAPL",
        now=NOW,
    )

    symbols = [row["symbol"] for row in result.context["portfolio"]["positions"]]
    intelligence = result.context["app_intelligence"]["relevant_symbol_intelligence"]
    assert "AAPL" in symbols
    assert calls == ["AAPL"]
    assert intelligence[0]["symbol"] == "AAPL"
    assert intelligence[0]["signals"][0]["label"] == "Trend"


def test_portfolio_question_includes_summary_and_top_risk_positions():
    account = _make_account(liquidation_value=100_000)
    positions = [
        _make_position(symbol="AAPL", market_value=32_000),
        _make_position(symbol="MSFT", market_value=12_000),
        _make_position(symbol="NVDA", market_value=6_000, pnl=-2_000),
    ]

    result = _builder().build(
        user_id="user-1",
        message="What should I do with my portfolio risk?",
        account=account,
        positions=positions,
        now=NOW,
    )

    portfolio = result.context["portfolio"]
    assert portfolio["total_value"] == 100_000
    assert portfolio["concentration"]["single_name_risks"][0]["symbol"] == "AAPL"
    assert any(
        row["symbol"] == "AAPL" and "concentration" in row["risk_notes"]
        for row in portfolio["positions"]
    )


def test_breakout_ideas_include_opportunities_without_every_holding_detail():
    def opportunities_provider(limit):
        return {
            "scanTime": NOW.isoformat(),
            "candidates": [
                {
                    "symbol": "CRWD",
                    "setupScore": 91,
                    "entryPrice": 420,
                    "stopPrice": 390,
                    "targetPrice": 480,
                    "riskReward": 2.0,
                    "rsPercentile": 96,
                    "marketRegime": "risk-on",
                }
            ],
        }

    positions = [
        _make_position(symbol=f"T{i}", market_value=10_000 - i)
        for i in range(12)
    ]
    result = _builder(opportunities_provider=opportunities_provider).build(
        user_id="user-1",
        message="Any breakout ideas today?",
        account=_make_account(liquidation_value=150_000),
        positions=positions,
        now=NOW,
    )

    intelligence = result.context["app_intelligence"]
    assert intelligence["emerging_leaders"][0]["symbol"] == "CRWD"
    assert intelligence["top_movers"][0]["symbol"] == "CRWD"
    assert result.context["portfolio"]["positions_omitted"] > 0
    assert len(result.context["portfolio"]["positions"]) < len(positions)


def test_stale_market_context_is_flagged():
    stale_as_of = NOW - timedelta(days=3)
    result = _builder(
        market_context_provider=lambda: {
            "as_of": stale_as_of.isoformat(),
            "regime": "risk-off",
            "notes": [],
        }
    ).build(
        user_id="user-1",
        message="How is the market?",
        account=_make_account(),
        positions=[],
        now=NOW,
    )

    market = result.context["market_context"]
    assert market["stale"] is True
    assert "do not present it as current" in market["notes"][-1]


def test_context_exceeding_cap_is_summarized_instead_of_failing():
    positions = [
        _make_position(symbol=f"BIG{i}", market_value=5_000 + i)
        for i in range(30)
    ]
    result = _builder(max_context_chars=2_000).build(
        user_id="user-1",
        message="Please review all holdings and portfolio risk",
        account=_make_account(liquidation_value=200_000),
        positions=positions,
        now=NOW,
    )

    assert result.truncated is True
    assert result.context["meta"]["truncated"] is True
    assert len(result.context["portfolio"]["positions"]) <= 3
    assert result.estimated_tokens > 0
