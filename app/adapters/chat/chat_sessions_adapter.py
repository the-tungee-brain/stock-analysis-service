from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Dict
from uuid import UUID, uuid4

import oracledb

from app.models.chat_sessions_models import ChatSession


class ChatSessionsAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "CHAT_SESSIONS"

    def _row_to_session(self, row: tuple) -> ChatSession:
        (
            raw_id,
            user_id,
            title,
            model,
            system_prompt,
            metadata_json,
            created_at,
            updated_at,
        ) = row

        return ChatSession(
            id=UUID(bytes=raw_id) if raw_id else None,
            user_id=user_id,
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

    def create_session(self, session: ChatSession) -> ChatSession:
        sql = f"""
            INSERT INTO {self.table_name} (
                id, user_id, title, model, system_prompt, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :user_id, :title, :model, :system_prompt, :metadata,
                :created_at, :updated_at
            )
            RETURNING created_at, updated_at
            INTO :out_created, :out_updated
        """

        session_id = session.id or uuid4()
        created_at = session.created_at or datetime.now(timezone.utc)
        updated_at = session.updated_at or created_at

        con = self.client.acquire()
        try:
            cur = con.cursor()

            out_created = cur.var(oracledb.DB_TYPE_TIMESTAMP_TZ)
            out_updated = cur.var(oracledb.DB_TYPE_TIMESTAMP_TZ)

            cur.execute(
                sql,
                id=session_id.bytes,
                user_id=session.user_id,
                title=session.title,
                model=session.model,
                system_prompt=session.system_prompt or "",
                metadata=session.model_dump_json() if session.metadata else None,
                created_at=created_at,
                updated_at=updated_at,
                out_created=out_created,
                out_updated=out_updated,
            )
            con.commit()

            created_val = out_created.getvalue()
            updated_val = out_updated.getvalue()

            # If DB returns naive datetimes, optionally attach UTC once, but guard None
            # if created_val and created_val.tzinfo is None:
            #     created_val = created_val.replace(tzinfo=timezone.utc)
            # if updated_val and updated_val.tzinfo is None:
            #     updated_val = updated_val.replace(tzinfo=timezone.utc)

            return session.model_copy(
                update={
                    "id": session_id,
                    "created_at": created_val,
                    "updated_at": updated_val,
                }
            )
        finally:
            con.close()

    def get_session_by_id(self, session_id: UUID) -> Optional[ChatSession]:
        sql = f"""
            SELECT id, user_id, title, model, system_prompt, metadata,
                   created_at, updated_at
            FROM {self.table_name}
            WHERE id = :id
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"id": session_id.bytes})
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_session(row)
        finally:
            con.close()

    def get_sessions_by_user_id(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChatSession]:
        sql = f"""
            SELECT id, user_id, title, model, system_prompt, metadata,
                   created_at, updated_at
            FROM {self.table_name}
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
            OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                sql,
                {
                    "user_id": user_id,
                    "offset": offset,
                    "limit": limit,
                },
            )
            rows = cur.fetchall()
            return [self._row_to_session(r) for r in rows]
        finally:
            con.close()

    def update_session(
        self,
        session_id: UUID,
        *,
        title: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[ChatSession]:
        updates = []
        params: Dict[str, Any] = {"id": session_id.bytes}

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
            return self.get_session_by_id(session_id)

        updates.append("updated_at = SYSTIMESTAMP")
        sql = f"""
            UPDATE {self.table_name}
            SET {", ".join(updates)}
            WHERE id = :id
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, params)
            con.commit()
            if cur.rowcount == 0:
                return None
            return self.get_session_by_id(session_id)
        finally:
            con.close()

    def delete_session(self, session_id: UUID) -> bool:
        sql = f"DELETE FROM {self.table_name} WHERE id = :id"

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"id": session_id.bytes})
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()

    def delete_sessions_by_user_id(self, user_id: str) -> int:
        sql = f"DELETE FROM {self.table_name} WHERE user_id = :user_id"

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"user_id": user_id})
            con.commit()
            return cur.rowcount
        finally:
            con.close()
