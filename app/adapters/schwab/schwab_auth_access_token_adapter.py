import oracledb
from typing import Optional, Dict, Any
from app.models.schwab_models import SchwabAuthTokenItem


class SchwabAuthAccessTokenAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client

    def item_to_dict(self, item: SchwabAuthTokenItem) -> Dict[str, Any]:
        return {
            "user_id": item.user_id,
            "access_token": item.access_token,
            "refresh_token": item.refresh_token,
            "expires_at": item.expires_at,
        }

    def dict_to_item(self, row: Dict[str, Any]) -> SchwabAuthTokenItem:
        return SchwabAuthTokenItem(
            id=row.get("ID"),
            user_id=row["USER_ID"],
            access_token=row["ACCESS_TOKEN"],
            refresh_token=row.get("REFRESH_TOKEN"),
            expires_at=row.get("EXPIRES_AT"),
            created_at=row.get("CREATED_AT"),
            updated_at=row.get("UPDATED_AT"),
        )

    def save(self, item: SchwabAuthTokenItem) -> int:
        con = self.client.acquire()
        try:
            cur = con.cursor()

            sql = f"""
            MERGE INTO {self.table_name} t
            USING (SELECT :user_id as user_id, :access_token as access_token, 
                          :refresh_token as refresh_token, :expires_at as expires_at 
                   FROM dual) s
            ON (t.user_id = s.user_id)
            WHEN MATCHED THEN
                UPDATE SET 
                    access_token = s.access_token,
                    refresh_token = s.refresh_token,
                    expires_at = s.expires_at,
                    updated_at = SYSTIMESTAMP
            WHEN NOT MATCHED THEN
                INSERT (user_id, access_token, refresh_token, expires_at)
                VALUES (s.user_id, s.access_token, s.refresh_token, s.expires_at)
            """

            bind_vars = self.item_to_dict(item)
            cur.execute(sql, bind_vars)
            rowcount = cur.rowcount
            con.commit()
            return rowcount
        finally:
            con.close()

    def get_by_user_id(self, user_id: str) -> Optional[SchwabAuthTokenItem]:
        con = self.client.acquire()
        try:
            cur = con.cursor()
            sql = f"SELECT * FROM {self.table_name} WHERE user_id = :user_id"
            cur.execute(sql, {"user_id": user_id})

            cols = [col[0] for col in cur.description]
            row = cur.fetchone()
            if not row:
                return None

            row_dict = dict(zip(cols, row))
            return self.dict_to_item(row_dict)
        finally:
            con.close()
