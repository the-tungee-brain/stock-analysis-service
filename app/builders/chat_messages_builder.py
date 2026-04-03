from typing import List, Optional
from app.adapters.chat.chat_messages_adapter import ChatMessagesAdapter
from app.models.chat_sessions_models import ChatMessage


class ChatMessagesBuilder:
    def __init__(self, chat_messages_adapter: ChatMessagesAdapter):
        self.chat_messages_adapter = chat_messages_adapter

    def create_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> ChatMessage:
        message = ChatMessage(session_id=session_id, role=role, content=content)
        return self.chat_messages_adapter.create(message=message)

    async def list_messages_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "desc",
    ) -> List[ChatMessage]:
        return await self.chat_messages_adapter.list_by_session(
            session_id=session_id,
            limit=limit,
            offset=offset,
            order=order,
        )
