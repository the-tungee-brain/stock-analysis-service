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


def test_build_playbook_ask_prompt_includes_long_term_sections():
    action = StrategyNextAction(
        type="research",
        title="Research SBUX before selling a put",
        reason="Confirm ownership comfort.",
        symbol="SBUX",
    )
    prompt = build_playbook_ask_prompt(action, InvestmentStrategy.WHEEL)
    assert "SBUX" in prompt
    assert "Business model" in prompt
    assert "SEC financial statements" in prompt
    assert "Strategy fit" in prompt
    assert "Confirm ownership comfort" not in prompt


def test_playbook_ask_display_message_is_user_facing_not_secret():
    action = StrategyNextAction(
        type="research",
        title="Research SBUX before selling a put",
        reason="Confirm ownership comfort.",
        symbol="SBUX",
    )
    display = playbook_ask_display_message(action, strategy=InvestmentStrategy.WHEEL)
    assert display == "Long-term research on SBUX before selling a put"
    assert "SEC financial" not in display
