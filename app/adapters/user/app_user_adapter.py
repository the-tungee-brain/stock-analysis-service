import oracledb
from typing import Optional, Dict, Any

from app.models.user_models import AppUserItem


class AppUserAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "APP_USER"

    def item_to_dict(self, item: AppUserItem) -> Dict[str, Any]:
        return {
            "id": item.id,
            "identity_sub": item.identity_sub,
            "identity_provider": item.identity_provider,
            "email": str(item.email),
            "full_name": item.full_name,
            "avatar_url": item.avatar_url,
            "created_at": item.created_at,
            "last_login_at": item.last_login_at,
        }

    def dict_to_item(self, row: Dict[str, Any]) -> AppUserItem:
        return AppUserItem(
            id=row["ID"],
            identity_sub=row["IDENTITY_SUB"],
            identity_provider=row["IDENTITY_PROVIDER"],
            email=row["EMAIL"],
            full_name=row.get("FULL_NAME"),
            avatar_url=row.get("AVATAR_URL"),
            created_at=row["CREATED_AT"],
            last_login_at=row.get("LAST_LOGIN_AT"),
        )

    def get_by_identity_sub(self, identity_sub: str) -> Optional[AppUserItem]:
        con = self.client.acquire()
        try:
            cur = con.cursor()
            sql = f"SELECT * FROM {self.table_name} WHERE identity_sub = :identity_sub"
            cur.execute(sql, {"identity_sub": identity_sub})

            cols = [col[0] for col in cur.description]
            row = cur.fetchone()
            if not row:
                return None

            row_dict = dict(zip(cols, row))
            return self.dict_to_item(row_dict)
        finally:
            con.close()

    def save(self, item: AppUserItem) -> int:
        con = self.client.acquire()
        try:
            cur = con.cursor()

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
                    :created_at       AS created_at,
                    :last_login_at    AS last_login_at
                FROM dual
            ) s
            ON (t.identity_sub = s.identity_sub)
            WHEN MATCHED THEN
                UPDATE SET
                    t.email         = s.email,
                    t.full_name     = s.full_name,
                    t.avatar_url    = s.avatar_url,
                    t.last_login_at = s.last_login_at
            WHEN NOT MATCHED THEN
                INSERT (
                    id,
                    identity_sub,
                    identity_provider,
                    email,
                    full_name,
                    avatar_url,
                    created_at,
                    last_login_at
                )
                VALUES (
                    s.id,
                    s.identity_sub,
                    s.identity_provider,
                    s.email,
                    s.full_name,
                    s.avatar_url,
                    s.created_at,
                    s.last_login_at
                )
            """

            bind_vars = self.item_to_dict(item)
            cur.execute(sql, bind_vars)
            rowcount = cur.rowcount
            con.commit()
            return rowcount
        finally:
            con.close()
