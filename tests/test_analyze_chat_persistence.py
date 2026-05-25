from unittest.mock import MagicMock

from app.core.prompts import AnalysisAction, uses_structured_system_message


def test_structured_analyze_is_detected_without_prompt():
    assert uses_structured_system_message(None, action=AnalysisAction.FREE_FORM)
    assert uses_structured_system_message("", action=AnalysisAction.FREE_FORM)


def test_free_form_chat_prompt_is_not_structured_analyze():
    assert not uses_structured_system_message(
        "What are the odds my call hits +30%?",
        action=AnalysisAction.FREE_FORM,
    )


def test_structured_analyze_route_skips_chat_service_when_structured():
    chat_service = MagicMock()
    chat_service.get_portfolio_analysis_session_id.return_value = ("session-id", False)
    chat_service.get_chat_messages_by_session.return_value = []

    structured = uses_structured_system_message(None, action=AnalysisAction.FREE_FORM)
    session_id = None
    recent_messages = []

    if not structured:
        session_prompt = chat_service.user_message_for_storage(
            prompt="Analyze my NVDA position.",
            action=AnalysisAction.FREE_FORM,
        )
        resolved_session_id, _ = chat_service.get_portfolio_analysis_session_id(
            user_id="user-1",
            symbol="NVDA",
            prompt=session_prompt,
            model="gpt-4.1-mini",
        )
        session_id = str(resolved_session_id) if resolved_session_id else None
        recent_messages = chat_service.get_chat_messages_by_session(
            session_id=session_id
        )

    assert structured
    assert session_id is None
    assert recent_messages == []
    chat_service.get_portfolio_analysis_session_id.assert_not_called()
    chat_service.get_chat_messages_by_session.assert_not_called()
    chat_service.create_message.assert_not_called()
