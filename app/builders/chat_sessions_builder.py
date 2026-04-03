from app.adapters.chat.chat_sessions_adapter import ChatSessionsAdapter
from app.models.chat_sessions_models import ChatSession


class ChatSessionsBuilder:
    def __init__(self, chat_sessions_adapter: ChatSessionsAdapter):
        self.chat_sessions_adapter = chat_sessions_adapter

    async def create_session(self, session: ChatSession) -> ChatSession:
        return await self.chat_sessions_adapter.create_session(session=session)

    async def get_session_by_id(self, session_id: str) -> ChatSession:
        return await self.chat_sessions_adapter.get_session_by_id(session_id=session_id)
