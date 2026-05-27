import oracledb
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from app.models.user_models import IdentityPayload
from app.models.waitlist_models import WaitlistEntryItem


class WaitlistAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "WAITLIST_ENTRY"

    def dict_to_item(self, row: Dict[str, Any]) -> WaitlistEntryItem:
        return WaitlistEntryItem(
            id=row["ID"],
            identity_sub=row["IDENTITY_SUB"],
            identity_provider=row["IDENTITY_PROVIDER"],
            email=row["EMAIL"],
            full_name=row.get("FULL_NAME"),
            avatar_url=row.get("AVATAR_URL"),
            status=row["STATUS"],
            created_at=row["CREATED_AT"],
            updated_at=row["UPDATED_AT"],
        )

    def get_by_identity_sub(self, identity_sub: str) -> Optional[WaitlistEntryItem]:
        con = self.client.acquire()
        try:
            cur = con.cursor()
            sql = f"""
                SELECT *
                FROM {self.table_name}
                WHERE identity_sub = :identity_sub
                  AND status = 'waiting'
            """
            cur.execute(sql, {"identity_sub": identity_sub})
            cols = [col[0] for col in cur.description]
            row = cur.fetchone()
            if not row:
                return None
            return self.dict_to_item(dict(zip(cols, row)))
        finally:
            con.close()

    def save_waiting(self, payload: IdentityPayload) -> WaitlistEntryItem:
        now = datetime.now(timezone.utc)
        entry_id = str(uuid4())

        sql = f"""
            MERGE INTO {self.table_name} t
            USING (
                SELECT
                    :id               AS id,
                    :identity_sub     AS identity_sub,
                    :identity_provider AS identity_provider,
                    :email            AS email,
                    :full_name        AS full_name,
                    :avatar_url       AS avatar_url,
                    :status           AS status,
                    :created_at       AS created_at,
                    :updated_at       AS updated_at
                FROM dual
            ) s
            ON (t.identity_sub = s.identity_sub)
            WHEN MATCHED THEN
                UPDATE SET
                    t.email             = s.email,
                    t.full_name         = s.full_name,
                    t.avatar_url        = s.avatar_url,
                    t.status            = s.status,
                    t.updated_at        = s.updated_at
            WHEN NOT MATCHED THEN
                INSERT (
                    id,
                    identity_sub,
                    identity_provider,
                    email,
                    full_name,
                    avatar_url,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (
                    s.id,
                    s.identity_sub,
                    s.identity_provider,
                    s.email,
                    s.full_name,
                    s.avatar_url,
                    s.status,
                    s.created_at,
                    s.updated_at
                )
        """

        bind_vars = {
            "id": entry_id,
            "identity_sub": payload.identity_sub,
            "identity_provider": payload.identity_provider,
            "email": str(payload.email),
            "full_name": payload.full_name,
            "avatar_url": payload.avatar_url,
            "status": "waiting",
            "created_at": now,
            "updated_at": now,
        }

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, bind_vars)
            con.commit()
        finally:
            con.close()

        saved = self.get_by_identity_sub(payload.identity_sub)
        if saved is None:
            raise RuntimeError("Failed to persist waitlist entry.")
        return saved

    def mark_promoted(self, identity_sub: str) -> None:
        now = datetime.now(timezone.utc)
        sql = f"""
            UPDATE {self.table_name}
            SET status = 'promoted',
                updated_at = :updated_at
            WHERE identity_sub = :identity_sub
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                sql,
                {"identity_sub": identity_sub, "updated_at": now},
            )
            con.commit()
        finally:
            con.close()

    def get_queue_position(self, identity_sub: str) -> Optional[int]:
        con = self.client.acquire()
        try:
            cur = con.cursor()
            sql = f"""
                SELECT queue_position
                FROM (
                    SELECT
                        identity_sub,
                        ROW_NUMBER() OVER (ORDER BY created_at ASC) AS queue_position
                    FROM {self.table_name}
                    WHERE status = 'waiting'
                )
                WHERE identity_sub = :identity_sub
            """
            cur.execute(sql, {"identity_sub": identity_sub})
            row = cur.fetchone()
            if not row:
                return None
            return int(row[0])
        finally:
            con.close()
