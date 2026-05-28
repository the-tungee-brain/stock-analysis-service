from app.models.strategy_models import InvestmentStrategy, StrategyNextAction
from app.services.strategy.strategy_playbook_prompts import (
    build_playbook_ask_prompt,
    playbook_ask_display_message,
    playbook_ask_prefers_research_chat,
    playbook_research_system_message,
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
    assert "wheel strategy playbook" in prompt
    assert "verdict format" in prompt.lower()
    assert "assignment comfort" in prompt.lower() or "assigned on a put" in prompt.lower()
    assert "dividend & payout" in prompt.lower()
    assert "Put zone" in prompt
    assert "Confirm ownership comfort" not in prompt
    assert "**Verdict:**" not in prompt


def test_playbook_research_system_message_includes_format_and_strategy_focus():
    message = playbook_research_system_message(strategy=InvestmentStrategy.DIVIDEND)
    lowered = message.lower()
    assert "**Verdict:**" in message
    assert "dividend investing playbook" in message
    assert "payout ratio" in lowered
    assert "fcf dividend coverage" in lowered
    assert "data i need" in lowered
    assert "never claim payout" in lowered


def test_playbook_research_user_message_uses_verdict_instruction():
    from app.models.company_research_models import ResearchContext
    from app.services.prompt_enrichment_service import PromptEnrichmentService

    ctx = ResearchContext(symbol="HOOD")
    message = PromptEnrichmentService().build_playbook_research_user_message(
        ctx=ctx,
        user_prompt="Should I hold HOOD?",
    )

    assert "verdict format" in message["content"].lower()
    assert "Acknowledge any gaps instead of guessing" not in message["content"]


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
