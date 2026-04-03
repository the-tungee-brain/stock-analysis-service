from typing import Optional, List, Dict, Any
from uuid import UUID

from app.builders.chat_sessions_builder import ChatSessionsBuilder
from app.builders.chat_messages_builder import ChatMessagesBuilder
from app.models.chat_sessions_models import ChatSession, ChatMessage
from app.core.prompts import SYSTEM_NATURAL_MESSAGE
from app.core.prompts import (
    SYSTEM_NATURAL_MESSAGE,
)


class ChatService:
    def __init__(
        self,
        chat_sessions_builder: ChatSessionsBuilder,
        chat_messages_builder: ChatMessagesBuilder,
    ):
        self.chat_sessions_builder = chat_sessions_builder
        self.chat_messages_builder = chat_messages_builder

    async def get_chat_session_id(
        self,
        user_id: str,
        session_id: Optional[str],
        prompt: Optional[str],
    ) -> Optional[str]:
        if not prompt:
            return None

        if not session_id:
            new_session = ChatSession(
                id="",
                user_id=user_id,
                title=prompt,
                system_prompt=SYSTEM_NATURAL_MESSAGE,
            )
            created_session = await self.chat_sessions_builder.create_session(
                session=new_session
            )
            return created_session.id

        return session_id

    async def create_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> ChatMessage:
        message = ChatMessage(
            id=None,
            session_id=UUID(session_id),
            role=role,
            content=content,
            metadata=metadata,
            created_at=None,
        )
        return await self.chat_messages_builder.create_message(message=message)

    async def get_chat_messages_by_session(
        self,
        session_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not session_id:
            return []

        messages = await self.chat_messages_builder.list_messages_by_session(
            session_id=session_id,
            limit=10,
            order="desc",
        )
        messages = list(reversed(messages))

        return self.chat_messages_to_openai_format(messages=messages)

    def chat_messages_to_openai_format(
        self,
        messages: List[ChatMessage],
    ) -> List[Dict[str, Any]]:
        openai_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.role.lower()
            if role not in ("user", "assistant"):
                continue

            openai_messages.append(
                {
                    "role": role,
                    "content": msg.content,
                }
            )

        return openai_messages
