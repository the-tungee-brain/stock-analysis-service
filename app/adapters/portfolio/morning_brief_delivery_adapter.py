from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional
from uuid import uuid4

import oracledb


class MorningBriefDeliveryAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "MORNING_BRIEF_DELIVERY"

    def was_delivered_today(self, user_id: str, delivery_date: date | None = None) -> bool:
        target_date = delivery_date or date.today()
        sql = f"""
            SELECT 1
            FROM {self.table_name}
            WHERE user_id = :user_id
              AND delivery_date = :delivery_date
              AND status = 'sent'
            FETCH FIRST 1 ROWS ONLY
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                sql,
                {"user_id": user_id, "delivery_date": target_date},
            )
            return cur.fetchone() is not None
        finally:
            con.close()

    def record_delivery(
        self,
        *,
        user_id: str,
        email: str,
        status: str = "sent",
        error_message: str | None = None,
        delivery_date: date | None = None,
    ) -> None:
        target_date = delivery_date or date.today()
        sql = f"""
            MERGE INTO {self.table_name} t
            USING (
                SELECT
                    :id              AS id,
                    :user_id         AS user_id,
                    :delivery_date   AS delivery_date,
                    :email           AS email,
                    :status          AS status,
                    :error_message   AS error_message
                FROM dual
            ) s
            ON (t.user_id = s.user_id AND t.delivery_date = s.delivery_date)
            WHEN MATCHED THEN
                UPDATE SET
                    t.email         = s.email,
                    t.status        = s.status,
                    t.error_message = s.error_message
            WHEN NOT MATCHED THEN
                INSERT (
                    id,
                    user_id,
                    delivery_date,
                    email,
                    status,
                    error_message
                )
                VALUES (
                    s.id,
                    s.user_id,
                    s.delivery_date,
                    s.email,
                    s.status,
                    s.error_message
                )
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                sql,
                {
                    "id": str(uuid4()),
                    "user_id": user_id,
                    "delivery_date": target_date,
                    "email": email,
                    "status": status,
                    "error_message": error_message,
                },
            )
            con.commit()
        finally:
            con.close()

    def count_sent_on(self, delivery_date: date | None = None) -> int:
        target_date = delivery_date or date.today()
        sql = f"""
            SELECT COUNT(*)
            FROM {self.table_name}
            WHERE delivery_date = :delivery_date
              AND status = 'sent'
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"delivery_date": target_date})
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()

    def delete_by_user_id(self, user_id: str) -> int:
        sql = f"DELETE FROM {self.table_name} WHERE user_id = :user_id"
        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"user_id": user_id})
            rowcount = cur.rowcount
            con.commit()
            return rowcount
        finally:
            con.close()
