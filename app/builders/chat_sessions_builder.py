from app.adapters.chat.chat_sessions_adapter import ChatSessionsAdapter
from app.models.chat_sessions_models import ChatSession
from typing import Optional


class ChatSessionsBuilder:
    def __init__(self, chat_sessions_adapter: ChatSessionsAdapter):
        self.chat_sessions_adapter = chat_sessions_adapter

    def create_session(self, session: ChatSession) -> ChatSession:
        return self.chat_sessions_adapter.create_session(session=session)

    def get_session_by_id(self, session_id: str) -> ChatSession:
        return self.chat_sessions_adapter.get_session_by_id(session_id=session_id)

    def get_latest_session_by_user_id(self, user_id: str) -> Optional[ChatSession]:
        sessions = self.chat_sessions_adapter.get_sessions_by_user_id(
            user_id=user_id,
            limit=1,
            offset=0,
        )
        return sessions[0] if sessions else None
