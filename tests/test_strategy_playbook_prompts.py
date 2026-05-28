from app.models.strategy_models import InvestmentStrategy, StrategyNextAction
from app.services.strategy.strategy_playbook_prompts import (
    build_playbook_ask_prompt,
    playbook_ask_display_message,
    playbook_ask_prefers_research_chat,
)


def test_playbook_ask_prefers_research_chat_for_csp_research():
    action = StrategyNextAction(
        type="research",
        title="Research SBUX before selling a put",
        reason="Confirm ownership comfort.",
        symbol="SBUX",
    )
    assert playbook_ask_prefers_research_chat(action) is True


def test_playbook_ask_prefers_portfolio_chat_for_monitor():
    action = StrategyNextAction(
        type="monitor",
        title="Monitor your SBUX short put",
        reason="Watch delta and DTE.",
        symbol="SBUX",
    )
    assert playbook_ask_prefers_research_chat(action) is False


def test_build_playbook_ask_prompt_is_concise_hold_verdict():
    action = StrategyNextAction(
        type="research",
        title="Research SBUX before selling a put",
        reason="Confirm ownership comfort.",
        symbol="SBUX",
    )
    prompt = build_playbook_ask_prompt(action, InvestmentStrategy.WHEEL)
    assert "SBUX" in prompt
    assert "**Verdict:**" in prompt
    assert "**Business:**" in prompt
    assert "**Financials:**" in prompt
    assert "**News:**" in prompt
    assert "220" in prompt or "320" in prompt
    assert "yfinance" in prompt.lower()
    assert "what/why parenthetical" in prompt.lower()
    assert "Business model" not in prompt
    assert "Put zone" in prompt
    assert "Confirm ownership comfort" not in prompt


def test_playbook_ask_display_message_is_user_facing_not_secret():
    action = StrategyNextAction(
        type="research",
        title="Research SBUX before selling a put",
        reason="Confirm ownership comfort.",
        symbol="SBUX",
    )
    display = playbook_ask_display_message(action, strategy=InvestmentStrategy.WHEEL)
    assert "comfortable owning SBUX" in display
    assert "Verdict" not in display
    assert "SEC financial" not in display
