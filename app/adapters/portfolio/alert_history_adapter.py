from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import oracledb

from app.core.prompts import AnalysisAction
from app.models.portfolio_memory_models import AlertHistoryItem, AlertStatus


class AlertHistoryAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "ALERT_HISTORY"

    @staticmethod
    def _row_to_item(row: tuple) -> AlertHistoryItem:
        (
            record_id,
            _user_id,
            fingerprint,
            action,
            symbol,
            reason,
            priority,
            status,
            first_seen_at,
            last_seen_at,
            _resolved_at,
        ) = row

        first_seen = first_seen_at.replace(tzinfo=timezone.utc)
        last_seen = last_seen_at.replace(tzinfo=timezone.utc)
        days_active = max(0, (last_seen.date() - first_seen.date()).days)

        return AlertHistoryItem(
            id=record_id,
            fingerprint=fingerprint,
            action=AnalysisAction.parse(action),
            label=AnalysisAction.parse(action).label,
            symbol=symbol,
            reason=reason,
            priority=int(priority),
            status=status,
            first_seen_at=first_seen,
            last_seen_at=last_seen,
            days_active=days_active,
        )

    def list_active(self, user_id: str) -> list[AlertHistoryItem]:
        sql = f"""
            SELECT id, user_id, fingerprint, action, symbol, reason,
                   priority, status, first_seen_at, last_seen_at, resolved_at
            FROM {self.table_name}
            WHERE user_id = :user_id
              AND status = 'active'
            ORDER BY priority ASC, last_seen_at DESC
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"user_id": user_id})
            return [self._row_to_item(row) for row in cur.fetchall()]
        finally:
            con.close()

    def list_recent(
        self, user_id: str, *, days: int = 30, limit: int = 100
    ) -> list[AlertHistoryItem]:
        sql = f"""
            SELECT id, user_id, fingerprint, action, symbol, reason,
                   priority, status, first_seen_at, last_seen_at, resolved_at
            FROM {self.table_name}
            WHERE user_id = :user_id
              AND last_seen_at >= systimestamp - :days
            ORDER BY last_seen_at DESC
            FETCH FIRST :limit ROWS ONLY
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"user_id": user_id, "days": days, "limit": limit})
            return [self._row_to_item(row) for row in cur.fetchall()]
        finally:
            con.close()

    def upsert_active(
        self,
        *,
        user_id: str,
        fingerprint: str,
        action: str,
        symbol: str | None,
        reason: str,
        priority: int,
    ) -> None:
        sql = f"""
            MERGE INTO {self.table_name} t
            USING (
                SELECT
                    :id           AS id,
                    :user_id      AS user_id,
                    :fingerprint  AS fingerprint,
                    :action       AS action,
                    :symbol       AS symbol,
                    :reason       AS reason,
                    :priority     AS priority
                FROM dual
            ) s
            ON (
                t.user_id = s.user_id
                AND t.fingerprint = s.fingerprint
                AND t.status = 'active'
            )
            WHEN MATCHED THEN
                UPDATE SET
                    t.reason       = s.reason,
                    t.priority     = s.priority,
                    t.last_seen_at = systimestamp
            WHEN NOT MATCHED THEN
                INSERT (
                    id,
                    user_id,
                    fingerprint,
                    action,
                    symbol,
                    reason,
                    priority,
                    status
                )
                VALUES (
                    s.id,
                    s.user_id,
                    s.fingerprint,
                    s.action,
                    s.symbol,
                    s.reason,
                    s.priority,
                    'active'
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
                    "fingerprint": fingerprint,
                    "action": action,
                    "symbol": symbol,
                    "reason": reason,
                    "priority": priority,
                },
            )
            con.commit()
        finally:
            con.close()

    def resolve_missing(self, user_id: str, active_fingerprints: set[str]) -> int:
        if not active_fingerprints:
            sql = f"""
                UPDATE {self.table_name}
                SET status = 'resolved',
                    resolved_at = systimestamp,
                    last_seen_at = systimestamp
                WHERE user_id = :user_id
                  AND status = 'active'
            """
            params: dict[str, object] = {"user_id": user_id}
        else:
            placeholders = ", ".join(
                f":fp_{index}" for index in range(len(active_fingerprints))
            )
            sql = f"""
                UPDATE {self.table_name}
                SET status = 'resolved',
                    resolved_at = systimestamp,
                    last_seen_at = systimestamp
                WHERE user_id = :user_id
                  AND status = 'active'
                  AND fingerprint NOT IN ({placeholders})
            """
            params = {"user_id": user_id}
            for index, fingerprint in enumerate(sorted(active_fingerprints)):
                params[f"fp_{index}"] = fingerprint

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, params)
            con.commit()
            return cur.rowcount
        finally:
            con.close()

    def dismiss(self, user_id: str, alert_id: str) -> bool:
        sql = f"""
            UPDATE {self.table_name}
            SET status = 'dismissed',
                resolved_at = systimestamp,
                last_seen_at = systimestamp
            WHERE user_id = :user_id
              AND id = :alert_id
              AND status = 'active'
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"user_id": user_id, "alert_id": alert_id})
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()

    def get_by_id(self, user_id: str, alert_id: str) -> Optional[AlertHistoryItem]:
        sql = f"""
            SELECT id, user_id, fingerprint, action, symbol, reason,
                   priority, status, first_seen_at, last_seen_at, resolved_at
            FROM {self.table_name}
            WHERE user_id = :user_id
              AND id = :alert_id
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"user_id": user_id, "alert_id": alert_id})
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_item(row)
        finally:
            con.close()
