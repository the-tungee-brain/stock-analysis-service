from datetime import datetime, timezone
from uuid import uuid4

from app.models.chat_sessions_models import ChatMessage, ChatSession
from app.services.chat_service import ChatService


class _FakeSessionsBuilder:
    def __init__(self, sessions: list[ChatSession]):
        self.sessions = sessions

    def get_sessions_by_user_id(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        title_prefix: str | None = None,
    ) -> list[ChatSession]:
        filtered = [session for session in self.sessions if session.user_id == user_id]
        if title_prefix:
            filtered = [
                session
                for session in filtered
                if session.title and session.title.startswith(title_prefix)
            ]
        filtered.sort(key=lambda session: session.updated_at, reverse=True)
        return filtered[offset : offset + limit]

    def get_session_by_id(self, session_id):
        for session in self.sessions:
            if session.id == session_id:
                return session
        return None


class _FakeMessagesBuilder:
    def __init__(self, messages: dict[str, list[ChatMessage]]):
        self.messages = messages

    def list_messages_by_session(
        self,
        session_id: str,
        limit=None,
        offset: int = 0,
        order: str = "desc",
    ) -> list[ChatMessage]:
        messages = self.messages.get(session_id, [])
        if order == "asc":
            messages = sorted(messages, key=lambda message: message.created_at)
        else:
            messages = sorted(
                messages,
                key=lambda message: message.created_at,
                reverse=True,
            )
        if limit is not None:
            messages = messages[:limit]
        return messages


def _session(
    *,
    user_id: str,
    title: str,
) -> ChatSession:
    now = datetime.now(timezone.utc)
    return ChatSession(
        id=uuid4(),
        user_id=user_id,
        title=title,
        model="gpt-4.1-mini",
        created_at=now,
        updated_at=now,
    )


def test_list_user_sessions_filters_by_kind():
    user_id = "user-1"
    sessions = [
        _session(user_id=user_id, title="Portfolio: daily review"),
        _session(user_id=user_id, title="Symbol:AAPL: risk check"),
        _session(user_id=user_id, title="Research:MSFT: moat question"),
    ]
    service = ChatService(
        chat_sessions_builder=_FakeSessionsBuilder(sessions),
        chat_messages_builder=_FakeMessagesBuilder({}),
    )

    portfolio = service.list_user_sessions(user_id=user_id, kind="portfolio")
    research = service.list_user_sessions(user_id=user_id, kind="research")

    assert len(portfolio) == 2
    assert {item["kind"] for item in portfolio} == {"portfolio"}
    assert len(research) == 1
    assert research[0]["kind"] == "research"


def test_get_session_messages_for_user_enforces_ownership():
    user_id = "user-1"
    session = _session(user_id=user_id, title="Research:NVDA: thesis")
    messages = [
        ChatMessage(
            id=1,
            session_id=session.id,
            role="user",
            content="Summarize the bull case.",
        ),
        ChatMessage(
            id=2,
            session_id=session.id,
            role="assistant",
            content="Here is the bull case.",
        ),
    ]
    service = ChatService(
        chat_sessions_builder=_FakeSessionsBuilder([session]),
        chat_messages_builder=_FakeMessagesBuilder({str(session.id): messages}),
    )

    result = service.get_session_messages_for_user(
        user_id=user_id,
        session_id=session.id,
    )
    assert result is not None
    assert len(result) == 2
    assert result[0]["role"] == "user"

    denied = service.get_session_messages_for_user(
        user_id="other-user",
        session_id=session.id,
    )
    assert denied is None
