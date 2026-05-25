from typing import Literal, Optional, List, Dict, Any
from uuid import UUID

from app.builders.chat_sessions_builder import ChatSessionsBuilder
from app.builders.chat_messages_builder import ChatMessagesBuilder
from app.models.chat_sessions_models import ChatSession, ChatMessage
from app.core.prompts import SYSTEM_NATURAL_MESSAGE, AnalysisAction, should_use_natural_response
from app.services.prompt_enrichment_service import RESEARCH_CHAT_SYSTEM_MESSAGE
from openai.types.shared import ResponsesModel


class ChatService:
    PORTFOLIO_TITLE_PREFIXES = ("Portfolio:", "Symbol:")
    RESEARCH_TITLE_PREFIX = "Research:"

    def __init__(
        self,
        chat_sessions_builder: ChatSessionsBuilder,
        chat_messages_builder: ChatMessagesBuilder,
    ):
        self.chat_sessions_builder = chat_sessions_builder
        self.chat_messages_builder = chat_messages_builder

    @staticmethod
    def user_message_for_storage(
        prompt: Optional[str],
        action: AnalysisAction,
    ) -> str:
        text = (prompt or "").strip()
        if text:
            return text
        return action.label

    @staticmethod
    def should_include_portfolio_context(
        *,
        is_first_chat: bool,
        action: AnalysisAction,
        recent_messages: List[Dict[str, Any]],
        user_prompt: Optional[str] = None,
    ) -> bool:
        if not should_use_natural_response(user_prompt, action=action):
            return True

        has_assistant_history = any(
            message["role"] == "assistant" for message in recent_messages
        )
        return (
            is_first_chat
            or action is not AnalysisAction.FREE_FORM
            or not has_assistant_history
        )

    def get_portfolio_analysis_session_id(
        self,
        user_id: str,
        symbol: Optional[str],
        prompt: Optional[str],
        model: ResponsesModel,
    ) -> tuple[Optional[UUID], bool]:
        if not prompt:
            return None, True

        prefix = self._portfolio_session_title_prefix(symbol=symbol)
        session = (
            self.chat_sessions_builder.get_latest_session_by_user_id_and_title_prefix(
                user_id=user_id,
                title_prefix=prefix,
            )
        )
        is_first_chat = not session

        if is_first_chat:
            new_session = ChatSession(
                user_id=user_id,
                title=f"{prefix} {prompt[:200]}",
                model=model,
                system_prompt=SYSTEM_NATURAL_MESSAGE,
            )
            created_session = self.chat_sessions_builder.create_session(
                session=new_session
            )
            return created_session.id, is_first_chat

        return session.id, is_first_chat

    @staticmethod
    def _portfolio_session_title_prefix(symbol: Optional[str]) -> str:
        if symbol:
            return f"Symbol:{symbol.strip().upper()}:"
        return "Portfolio:"

    def get_chat_session_id(
        self,
        user_id: str,
        prompt: Optional[str],
        model: ResponsesModel,
    ) -> tuple[Optional[UUID], bool]:
        if not prompt:
            return None, True

        session = self.chat_sessions_builder.get_latest_session_by_user_id(
            user_id=user_id
        )
        is_first_chat = not session

        if is_first_chat:
            new_session = ChatSession(
                user_id=user_id,
                title=prompt,
                model=model,
                system_prompt=SYSTEM_NATURAL_MESSAGE,
            )
            created_session = self.chat_sessions_builder.create_session(
                session=new_session
            )
            return created_session.id, is_first_chat

        return session.id, is_first_chat

    @staticmethod
    def _research_session_title_prefix(symbol: str) -> str:
        return f"Research:{symbol.strip().upper()}:"

    def get_research_chat_session_id(
        self,
        user_id: str,
        symbol: str,
        prompt: Optional[str],
        model: ResponsesModel,
    ) -> tuple[Optional[UUID], bool]:
        if not prompt:
            return None, True

        prefix = self._research_session_title_prefix(symbol=symbol)
        session = (
            self.chat_sessions_builder.get_latest_session_by_user_id_and_title_prefix(
                user_id=user_id,
                title_prefix=prefix,
            )
        )
        is_first_chat = not session

        if is_first_chat:
            new_session = ChatSession(
                user_id=user_id,
                title=f"{prefix} {prompt[:200]}",
                model=model,
                system_prompt=RESEARCH_CHAT_SYSTEM_MESSAGE,
            )
            created_session = self.chat_sessions_builder.create_session(
                session=new_session
            )
            return created_session.id, is_first_chat

        return session.id, is_first_chat

    def create_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> ChatMessage:
        return self.chat_messages_builder.create_message(
            session_id=session_id, role=role, content=content
        )

    def get_chat_messages_by_session(
        self,
        session_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not session_id:
            return []

        messages = self.chat_messages_builder.list_messages_by_session(
            session_id=session_id,
            limit=20,
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

    @classmethod
    def _title_prefix_for_kind(cls, kind: str) -> Optional[str]:
        if kind == "portfolio":
            return cls.PORTFOLIO_TITLE_PREFIXES[0]
        if kind == "research":
            return cls.RESEARCH_TITLE_PREFIX
        return None

    @staticmethod
    def _session_kind(title: Optional[str]) -> str:
        if not title:
            return "other"
        if title.startswith("Research:"):
            return "research"
        if title.startswith("Portfolio:") or title.startswith("Symbol:"):
            return "portfolio"
        return "other"

    def list_user_sessions(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        kind: Literal["all", "portfolio", "research"] = "all",
    ) -> List[Dict[str, Any]]:
        if kind == "all":
            sessions = self.chat_sessions_builder.get_sessions_by_user_id(
                user_id=user_id,
                limit=limit,
                offset=offset,
            )
        else:
            prefix = self._title_prefix_for_kind(kind)
            sessions = self.chat_sessions_builder.get_sessions_by_user_id(
                user_id=user_id,
                limit=limit,
                offset=offset,
                title_prefix=prefix,
            )
            if kind == "portfolio":
                symbol_sessions = (
                    self.chat_sessions_builder.get_sessions_by_user_id(
                        user_id=user_id,
                        limit=limit,
                        offset=offset,
                        title_prefix="Symbol:",
                    )
                )
                combined = {session.id: session for session in sessions + symbol_sessions}
                sessions = sorted(
                    combined.values(),
                    key=lambda session: session.updated_at,
                    reverse=True,
                )[:limit]

        return [
            {
                "id": str(session.id),
                "title": session.title,
                "model": session.model,
                "kind": self._session_kind(session.title),
                "createdAt": session.created_at.isoformat(),
                "updatedAt": session.updated_at.isoformat(),
            }
            for session in sessions
            if session.id is not None
        ]

    def get_session_for_user(
        self,
        user_id: str,
        session_id: UUID,
    ) -> Optional[ChatSession]:
        session = self.chat_sessions_builder.get_session_by_id(session_id)
        if session is None or session.user_id != user_id:
            return None
        return session

    def get_session_messages_for_user(
        self,
        user_id: str,
        session_id: UUID,
        *,
        limit: int = 100,
    ) -> Optional[List[Dict[str, Any]]]:
        session = self.get_session_for_user(user_id=user_id, session_id=session_id)
        if session is None:
            return None

        messages = self.chat_messages_builder.list_messages_by_session(
            session_id=str(session_id),
            limit=limit,
            order="asc",
        )
        return [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "createdAt": message.created_at.isoformat(),
            }
            for message in messages
        ]

    def delete_session_for_user(self, user_id: str, session_id: UUID) -> bool:
        session = self.get_session_for_user(user_id=user_id, session_id=session_id)
        if session is None or session.id is None:
            return False

        self.chat_messages_builder.delete_messages_by_session(session.id)
        return self.chat_sessions_builder.delete_session(session.id)

    def clear_sessions_for_title_prefix(
        self,
        user_id: str,
        title_prefix: str,
    ) -> int:
        sessions = self.chat_sessions_builder.get_sessions_by_user_id(
            user_id=user_id,
            limit=100,
            offset=0,
            title_prefix=title_prefix,
        )

        deleted = 0
        for session in sessions:
            if session.id is None or session.user_id != user_id:
                continue
            self.chat_messages_builder.delete_messages_by_session(session.id)
            if self.chat_sessions_builder.delete_session(session.id):
                deleted += 1

        return deleted
