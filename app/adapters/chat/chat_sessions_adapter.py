from __future__ import annotations

from datetime import timezone
from typing import Any, Optional
from uuid import UUID

import oracledb

from app.models.chat_sessions_models import ChatSession


class ChatSessionsAdapter:
    """CRUD adapter for CHAT_SESSIONS table using oracledb."""

    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "CHAT_SESSIONS"

    async def create_session(self, session: ChatSession) -> ChatSession:
        """Insert a new chat session and return it with DB-generated defaults."""
        sql = f"""
            INSERT INTO {self.table_name} (
                id, user_id, title, model, system_prompt, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :user_id, :title, :model, :system_prompt, :metadata,
                :created_at, :updated_at
            )
            RETURNING id, created_at, updated_at INTO :out_id, :out_created, :out_updated
        """

        async with self.client.acquire() as conn:
            async with conn.cursor() as cur:
                out_id = cur.var(oracledb.DB_TYPE_RAW)
                out_created = cur.var(oracledb.DB_TYPE_TIMESTAMP_TZ)
                out_updated = cur.var(oracledb.DB_TYPE_TIMESTAMP_TZ)

                await cur.execute(
                    sql,
                    id=session.id.bytes,
                    user_id=session.user_id.bytes,
                    title=session.title,
                    model=session.model,
                    system_prompt=session.system_prompt,
                    metadata=session.model_dump_json() if session.metadata else None,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    out_id=out_id,
                    out_created=out_created,
                    out_updated=out_updated,
                )
                await conn.commit()

                return session.model_copy(
                    update={
                        "created_at": out_created.getvalue().replace(
                            tzinfo=timezone.utc
                        ),
                        "updated_at": out_updated.getvalue().replace(
                            tzinfo=timezone.utc
                        ),
                    }
                )

    async def get_session_by_id(self, session_id: UUID) -> Optional[ChatSession]:
        """Fetch a single session by its ID."""
        sql = f"""
            SELECT id, user_id, title, model, system_prompt, metadata,
                   created_at, updated_at
            FROM {self.table_name}
            WHERE id = :id
        """

        async with self.client.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, id=session_id.bytes)
                row = await cur.fetchone()
                if not row:
                    return None
                return self._row_to_session(row)

    async def get_sessions_by_user_id(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChatSession]:
        """Fetch sessions for a user, ordered by updated_at DESC."""
        sql = f"""
            SELECT id, user_id, title, model, system_prompt, metadata,
                   created_at, updated_at
            FROM {self.table_name}
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
            OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
        """

        async with self.client.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql, user_id=user_id.bytes, offset=offset, limit=limit
                )
                rows = await cur.fetchall()
                return [self._row_to_session(r) for r in rows]

    async def update_session(
        self,
        session_id: UUID,
        *,
        title: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[ChatSession]:
        updates = []
        params: dict[str, Any] = {"id": session_id.bytes}

        if title is not None:
            updates.append("title = :title")
            params["title"] = title
        if model is not None:
            updates.append("model = :model")
            params["model"] = model
        if system_prompt is not None:
            updates.append("system_prompt = :system_prompt")
            params["system_prompt"] = system_prompt
        if metadata is not None:
            updates.append("metadata = :metadata")
            params["metadata"] = ChatSession(metadata=metadata).model_dump_json()

        if not updates:
            return await self.get_session_by_id(session_id)

        updates.append("updated_at = SYSTIMESTAMP")
        sql = f"""
            UPDATE {self.table_name}
            SET {", ".join(updates)}
            WHERE id = :id
        """

        async with self.client.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, **params)
                await conn.commit()
                if cur.rowcount == 0:
                    return None
                return await self.get_session_by_id(session_id)

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session by ID. Returns True if deleted, False if not found."""
        sql = f"DELETE FROM {self.table_name} WHERE id = :id"

        async with self.client.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, id=session_id.bytes)
                await conn.commit()
                return cur.rowcount > 0

    async def delete_sessions_by_user_id(self, user_id: UUID) -> int:
        """Delete all sessions for a user. Returns number of rows deleted."""
        sql = f"DELETE FROM {self.table_name} WHERE user_id = :user_id"

        async with self.client.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, user_id=user_id.bytes)
                await conn.commit()
                return cur.rowcount

    @staticmethod
    def _row_to_session(row: tuple) -> ChatSession:
        """Convert a DB row to a ChatSession Pydantic model."""
        (
            raw_id,
            raw_user_id,
            title,
            model,
            system_prompt,
            metadata_json,
            created_at,
            updated_at,
        ) = row
        return ChatSession(
            id=UUID(bytes=raw_id),
            user_id=UUID(bytes=raw_user_id),
            title=title,
            model=model,
            system_prompt=system_prompt,
            metadata=(
                ChatSession.model_validate_json(metadata_json).metadata
                if metadata_json
                else None
            ),
            created_at=created_at.replace(tzinfo=timezone.utc) if created_at else None,
            updated_at=updated_at.replace(tzinfo=timezone.utc) if updated_at else None,
        )
