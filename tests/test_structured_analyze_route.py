from app.core.prompts import AnalysisAction
from app.services.chat_service import ChatService


def test_structured_analyze_always_includes_portfolio_context():
    assert ChatService.should_include_portfolio_context(
        is_first_chat=False,
        action=AnalysisAction.FREE_FORM,
        recent_messages=[{"role": "assistant", "content": "prior natural reply"}],
        user_prompt=None,
    )


def test_followup_free_form_can_omit_context_after_assistant_history():
    assert not ChatService.should_include_portfolio_context(
        is_first_chat=False,
        action=AnalysisAction.FREE_FORM,
        recent_messages=[{"role": "assistant", "content": "prior natural reply"}],
        user_prompt="let's do that",
    )
