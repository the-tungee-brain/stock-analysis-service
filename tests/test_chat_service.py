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

    def get_latest_session_by_user_id_and_title_prefix(
        self,
        user_id: str,
        title_prefix: str,
    ):
        matches = [
            session
            for session in self.sessions
            if session.user_id == user_id
            and session.title
            and session.title.startswith(title_prefix)
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda session: session.updated_at, reverse=True)[0]

    def create_session(self, session: ChatSession) -> ChatSession:
        if session.id is None:
            session.id = uuid4()
        self.sessions.insert(0, session)
        return session


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


class _DeletingSessionsBuilder(_FakeSessionsBuilder):
    def __init__(self, sessions: list[ChatSession]):
        super().__init__(sessions)
        self.deleted_ids: list = []

    def delete_session(self, session_id) -> bool:
        self.deleted_ids.append(session_id)
        self.sessions = [
            session for session in self.sessions if session.id != session_id
        ]
        return True


class _DeletingMessagesBuilder(_FakeMessagesBuilder):
    def __init__(self, messages: dict[str, list[ChatMessage]]):
        super().__init__(messages)
        self.deleted_session_ids: list = []

    def delete_messages_by_session(self, session_id) -> int:
        key = str(session_id)
        self.deleted_session_ids.append(session_id)
        deleted = len(self.messages.get(key, []))
        self.messages[key] = []
        return deleted


def test_delete_session_for_user_removes_messages_then_session():
    user_id = "user-1"
    session = _session(user_id=user_id, title="Portfolio: review")
    sessions_builder = _DeletingSessionsBuilder([session])
    messages_builder = _DeletingMessagesBuilder(
        {str(session.id): [ChatMessage(id=1, session_id=session.id, role="user", content="hi")]}
    )
    service = ChatService(
        chat_sessions_builder=sessions_builder,
        chat_messages_builder=messages_builder,
    )

    assert service.delete_session_for_user(user_id=user_id, session_id=session.id) is True
    assert messages_builder.deleted_session_ids == [session.id]
    assert sessions_builder.deleted_ids == [session.id]


def test_resolve_prefixed_chat_session_creates_new_when_requested():
    user_id = "user-1"
    existing = _session(user_id=user_id, title="Portfolio: old thread")
    sessions_builder = _FakeSessionsBuilder([existing])
    service = ChatService(
        chat_sessions_builder=sessions_builder,
        chat_messages_builder=_FakeMessagesBuilder({}),
    )

    session_id, is_first = service.get_portfolio_analysis_session_id(
        user_id=user_id,
        symbol=None,
        prompt="Start a fresh portfolio review",
        model="gpt-4.1-mini",
        new_chat_session=True,
    )

    assert session_id is not None
    assert session_id != existing.id
    assert is_first is True
    assert len(sessions_builder.sessions) == 2


def test_resolve_prefixed_chat_session_resumes_explicit_session():
    user_id = "user-1"
    session = _session(user_id=user_id, title="Symbol:AAPL: earlier thread")
    messages = [
        ChatMessage(
            id=1,
            session_id=session.id,
            role="user",
            content="How concentrated is AAPL?",
        ),
    ]
    service = ChatService(
        chat_sessions_builder=_FakeSessionsBuilder([session]),
        chat_messages_builder=_FakeMessagesBuilder({str(session.id): messages}),
    )

    session_id, is_first = service.get_portfolio_analysis_session_id(
        user_id=user_id,
        symbol="AAPL",
        prompt="Follow up on trim plan",
        model="gpt-4.1-mini",
        chat_session_id=str(session.id),
    )

    assert session_id == session.id
    assert is_first is False
    user_id = "user-1"
    sessions = [
        _session(user_id=user_id, title="Symbol:AAPL: old"),
        _session(user_id=user_id, title="Symbol:AAPL: new"),
        _session(user_id=user_id, title="Portfolio: review"),
    ]
    sessions_builder = _DeletingSessionsBuilder(sessions)
    messages_builder = _DeletingMessagesBuilder({str(s.id): [] for s in sessions})
    service = ChatService(
        chat_sessions_builder=sessions_builder,
        chat_messages_builder=messages_builder,
    )

    deleted = service.clear_sessions_for_title_prefix(
        user_id=user_id,
        title_prefix="Symbol:AAPL:",
    )

    assert deleted == 2
    assert len(sessions_builder.deleted_ids) == 2
