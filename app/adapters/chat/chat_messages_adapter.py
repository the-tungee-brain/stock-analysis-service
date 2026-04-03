from __future__ import annotations

import json
from typing import Any, Optional
from uuid import UUID

import oracledb

from app.models.chat_sessions_models import ChatMessage


class ChatMessagesAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "CHAT_MESSAGES"

    # ---------- Helpers ----------

    def _uuid_to_raw(self, uuid_val: UUID | str) -> bytes:
        if isinstance(uuid_val, str):
            uuid_val = UUID(uuid_val)
        return uuid_val.bytes

    def _raw_to_uuid(self, raw_val: bytes) -> UUID:
        return UUID(bytes=raw_val)

    def _json_to_clob(self, data: Optional[dict[str, Any]]) -> Optional[str]:
        if data is None:
            return None
        return json.dumps(data, ensure_ascii=False)

    def _clob_to_json(self, clob_val: Optional[str]) -> Optional[dict[str, Any]]:
        if clob_val is None:
            return None
        return json.loads(clob_val)

    # ---------- CRUD ----------

    def create(self, message: ChatMessage) -> ChatMessage:
        sql = f"""
            INSERT INTO {self.table_name} (
                session_id,
                role,
                content,
                metadata
            ) VALUES (
                :session_id,
                :role,
                :content,
                :metadata
            )
            RETURNING id, created_at INTO :id, :created_at
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()

            id_var = cur.var(oracledb.DB_TYPE_NUMBER)
            created_at_var = cur.var(oracledb.DB_TYPE_TIMESTAMP_TZ)

            cur.execute(
                sql,
                session_id=self._uuid_to_raw(message.session_id),
                role=message.role,
                content=message.content,
                metadata=self._json_to_clob(message.metadata),
                id=id_var,
                created_at=created_at_var,
            )

            con.commit()

            id_val = id_var.getvalue()
            if isinstance(id_val, list):
                id_val = id_val[0]
            new_id = int(id_val)

            created_at = created_at_var.getvalue()
            if isinstance(created_at, list):
                created_at = created_at[0]

            return ChatMessage(
                id=new_id,
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                metadata=message.metadata,
                created_at=created_at,
            )
        finally:
            con.close()

    def get_by_id(self, message_id: int) -> Optional[ChatMessage]:
        sql = f"""
            SELECT id, session_id, role, content, metadata, created_at
            FROM {self.table_name}
            WHERE id = :id
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, id=message_id)
            row = cur.fetchone()
            if row is None:
                return None

            id_val, session_id_raw, role, content, metadata_clob, created_at = row

            return ChatMessage(
                id=id_val,
                session_id=self._raw_to_uuid(session_id_raw),
                role=role,
                content=content.read() if hasattr(content, "read") else content,
                metadata=self._clob_to_json(
                    metadata_clob.read()
                    if hasattr(metadata_clob, "read")
                    else metadata_clob
                ),
                created_at=created_at,
            )
        finally:
            con.close()

    def list_by_session(
        self,
        session_id: UUID | str,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "asc",
    ) -> list[ChatMessage]:
        if isinstance(session_id, str):
            session_id = UUID(session_id)

        order_clause = "DESC" if order.lower() == "desc" else "ASC"
        sql = f"""
            SELECT id, session_id, role, content, metadata, created_at
            FROM {self.table_name}
            WHERE session_id = :session_id
            ORDER BY created_at {order_clause}
        """

        params: dict[str, Any] = {"session_id": self._uuid_to_raw(session_id)}

        if limit is not None:
            sql += " OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY"
            params["offset"] = offset
            params["limit"] = limit

        messages: list[ChatMessage] = []

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()

            for row in rows:
                id_val, session_id_raw, role, content, metadata_clob, created_at = row
                messages.append(
                    ChatMessage(
                        id=id_val,
                        session_id=self._raw_to_uuid(session_id_raw),
                        role=role,
                        content=content.read() if hasattr(content, "read") else content,
                        metadata=self._clob_to_json(
                            metadata_clob.read()
                            if hasattr(metadata_clob, "read")
                            else metadata_clob
                        ),
                        created_at=created_at,
                    )
                )

        finally:
            con.close()

        return messages

    def update(self, message_id: int, updates: dict[str, Any]) -> Optional[ChatMessage]:
        allowed = {"role", "content", "metadata"}
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return self.get_by_id(message_id)

        set_clauses = []
        params: dict[str, Any] = {"id": message_id}

        for key, value in filtered.items():
            param_name = f"p_{key}"
            if key == "metadata":
                set_clauses.append(f"metadata = :{param_name}")
                params[param_name] = self._json_to_clob(value)
            else:
                set_clauses.append(f"{key} = :{param_name}")
                params[param_name] = value

        sql = f"""
            UPDATE {self.table_name}
            SET {", ".join(set_clauses)},
                updated_at = SYSTIMESTAMP
            WHERE id = :id
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, params)
            if cur.rowcount == 0:
                return None
            con.commit()
        finally:
            con.close()

        return self.get_by_id(message_id)

    def delete(self, message_id: int) -> bool:
        sql = f"DELETE FROM {self.table_name} WHERE id = :id"

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, id=message_id)
            deleted = cur.rowcount > 0
            con.commit()
            return deleted
        finally:
            con.close()

    def delete_by_session(self, session_id: UUID | str) -> int:
        if isinstance(session_id, str):
            session_id = UUID(session_id)

        sql = f"DELETE FROM {self.table_name} WHERE session_id = :session_id"

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, session_id=self._uuid_to_raw(session_id))
            deleted_count = cur.rowcount
            con.commit()
            return deleted_count
        finally:
            con.close()
