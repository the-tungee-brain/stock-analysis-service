from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Optional
from uuid import uuid4

import oracledb

from app.models.portfolio_memory_models import (
    PortfolioSnapshotRecord,
    PortfolioSnapshotSummary,
    SnapshotPosition,
)


class PortfolioSnapshotAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "PORTFOLIO_SNAPSHOT"

    def _row_to_record(self, row: tuple) -> PortfolioSnapshotRecord:
        (
            record_id,
            user_id,
            snapshot_date,
            account_number,
            liquidation_value,
            cash_balance,
            positions_json,
            summary_json,
            created_at,
        ) = row

        positions = [
            SnapshotPosition.model_validate(item)
            for item in json.loads(positions_json or "[]")
        ]
        summary = (
            PortfolioSnapshotSummary.model_validate_json(summary_json)
            if summary_json
            else None
        )

        return PortfolioSnapshotRecord(
            id=record_id,
            user_id=user_id,
            snapshot_date=snapshot_date,
            account_number=account_number,
            liquidation_value=liquidation_value,
            cash_balance=cash_balance,
            positions=positions,
            summary=summary,
            created_at=created_at.replace(tzinfo=timezone.utc) if created_at else None,
        )

    def get_by_user_and_date(
        self, user_id: str, snapshot_date: date
    ) -> Optional[PortfolioSnapshotRecord]:
        sql = f"""
            SELECT id, user_id, snapshot_date, account_number,
                   liquidation_value, cash_balance, positions_json,
                   summary_json, created_at
            FROM {self.table_name}
            WHERE user_id = :user_id
              AND snapshot_date = :snapshot_date
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                sql,
                {"user_id": user_id, "snapshot_date": snapshot_date},
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_record(row)
        finally:
            con.close()

    def list_recent(
        self, user_id: str, *, limit: int = 30
    ) -> list[PortfolioSnapshotRecord]:
        sql = f"""
            SELECT id, user_id, snapshot_date, account_number,
                   liquidation_value, cash_balance, positions_json,
                   summary_json, created_at
            FROM {self.table_name}
            WHERE user_id = :user_id
            ORDER BY snapshot_date DESC
            FETCH FIRST :limit ROWS ONLY
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"user_id": user_id, "limit": limit})
            return [self._row_to_record(row) for row in cur.fetchall()]
        finally:
            con.close()

    def upsert(self, record: PortfolioSnapshotRecord) -> PortfolioSnapshotRecord:
        record_id = record.id or str(uuid4())
        positions_json = json.dumps(
            [position.model_dump(mode="json") for position in record.positions]
        )
        summary_json = (
            record.summary.model_dump_json(by_alias=True) if record.summary else None
        )

        sql = f"""
            MERGE INTO {self.table_name} t
            USING (
                SELECT
                    :id                 AS id,
                    :user_id            AS user_id,
                    :snapshot_date      AS snapshot_date,
                    :account_number     AS account_number,
                    :liquidation_value  AS liquidation_value,
                    :cash_balance       AS cash_balance,
                    :positions_json     AS positions_json,
                    :summary_json       AS summary_json
                FROM dual
            ) s
            ON (t.user_id = s.user_id AND t.snapshot_date = s.snapshot_date)
            WHEN MATCHED THEN
                UPDATE SET
                    t.account_number    = s.account_number,
                    t.liquidation_value = s.liquidation_value,
                    t.cash_balance      = s.cash_balance,
                    t.positions_json    = s.positions_json,
                    t.summary_json      = s.summary_json
            WHEN NOT MATCHED THEN
                INSERT (
                    id,
                    user_id,
                    snapshot_date,
                    account_number,
                    liquidation_value,
                    cash_balance,
                    positions_json,
                    summary_json
                )
                VALUES (
                    s.id,
                    s.user_id,
                    s.snapshot_date,
                    s.account_number,
                    s.liquidation_value,
                    s.cash_balance,
                    s.positions_json,
                    s.summary_json
                )
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                sql,
                {
                    "id": record_id,
                    "user_id": record.user_id,
                    "snapshot_date": record.snapshot_date,
                    "account_number": record.account_number,
                    "liquidation_value": record.liquidation_value,
                    "cash_balance": record.cash_balance,
                    "positions_json": positions_json,
                    "summary_json": summary_json,
                },
            )
            con.commit()
        finally:
            con.close()

        saved = self.get_by_user_and_date(record.user_id, record.snapshot_date)
        return saved or record.model_copy(update={"id": record_id})

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
