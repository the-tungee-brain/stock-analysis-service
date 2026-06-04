"""Oracle-backed paper-trading performance store."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import oracledb

from trade_planner.alerts.paper_trade_models import PaperTradePerformanceRecord
from app.adapters.strategy.paper_trade_performance_mappers import (
    record_from_row,
    record_to_row,
)

_TABLE = "MOMENTUM_BREAKOUT_PAPER_TRADE"


class OraclePaperTradePerformanceStore:
    def __init__(self, client: oracledb.ConnectionPool) -> None:
        self._client = client

    def get(self, user_id: str, alert_id: str) -> PaperTradePerformanceRecord | None:
        sql = f"""
            SELECT alert_id, user_id, symbol, setup_name, signal_date,
                   entry_triggered_at, entry_price, stop_price, target_price,
                   exit_at, exit_price, status, outcome_return_pct, holding_days,
                   risk_gate_action, market_regime, volume_ratio, rs_percentile,
                   created_at
            FROM {_TABLE}
            WHERE user_id = :user_id AND alert_id = :alert_id
        """
        row = self._fetchone(sql, {"user_id": user_id, "alert_id": alert_id})
        if row is None:
            return None
        return record_from_row(self._row_dict(row))

    def save(self, user_id: str, record: PaperTradePerformanceRecord) -> None:
        payload = record_to_row(record)
        payload["user_id"] = user_id
        sql = f"""
            MERGE INTO {_TABLE} t
            USING (
                SELECT
                    :alert_id AS alert_id,
                    :user_id AS user_id,
                    :symbol AS symbol,
                    :setup_name AS setup_name,
                    :signal_date AS signal_date,
                    :entry_triggered_at AS entry_triggered_at,
                    :entry_price AS entry_price,
                    :stop_price AS stop_price,
                    :target_price AS target_price,
                    :exit_at AS exit_at,
                    :exit_price AS exit_price,
                    :status AS status,
                    :outcome_return_pct AS outcome_return_pct,
                    :holding_days AS holding_days,
                    :risk_gate_action AS risk_gate_action,
                    :market_regime AS market_regime,
                    :volume_ratio AS volume_ratio,
                    :rs_percentile AS rs_percentile,
                    :created_at AS created_at
                FROM dual
            ) s
            ON (t.alert_id = s.alert_id)
            WHEN MATCHED THEN UPDATE SET
                t.symbol = s.symbol,
                t.setup_name = s.setup_name,
                t.signal_date = s.signal_date,
                t.entry_triggered_at = s.entry_triggered_at,
                t.entry_price = s.entry_price,
                t.stop_price = s.stop_price,
                t.target_price = s.target_price,
                t.exit_at = s.exit_at,
                t.exit_price = s.exit_price,
                t.status = s.status,
                t.outcome_return_pct = s.outcome_return_pct,
                t.holding_days = s.holding_days,
                t.risk_gate_action = s.risk_gate_action,
                t.market_regime = s.market_regime,
                t.volume_ratio = s.volume_ratio,
                t.rs_percentile = s.rs_percentile,
                t.updated_at = systimestamp
            WHEN NOT MATCHED THEN INSERT (
                alert_id, user_id, symbol, setup_name, signal_date,
                entry_triggered_at, entry_price, stop_price, target_price,
                exit_at, exit_price, status, outcome_return_pct, holding_days,
                risk_gate_action, market_regime, volume_ratio, rs_percentile,
                created_at
            ) VALUES (
                s.alert_id, s.user_id, s.symbol, s.setup_name, s.signal_date,
                s.entry_triggered_at, s.entry_price, s.stop_price, s.target_price,
                s.exit_at, s.exit_price, s.status, s.outcome_return_pct, s.holding_days,
                s.risk_gate_action, s.market_regime, s.volume_ratio, s.rs_percentile,
                s.created_at
            )
        """
        self._execute(sql, payload)

    def list_for_user(
        self, user_id: str, *, limit: int = 500
    ) -> tuple[PaperTradePerformanceRecord, ...]:
        sql = f"""
            SELECT alert_id, user_id, symbol, setup_name, signal_date,
                   entry_triggered_at, entry_price, stop_price, target_price,
                   exit_at, exit_price, status, outcome_return_pct, holding_days,
                   risk_gate_action, market_regime, volume_ratio, rs_percentile,
                   created_at
            FROM {_TABLE}
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            FETCH FIRST :limit ROWS ONLY
        """
        rows = self._fetchall(sql, {"user_id": user_id, "limit": limit})
        return tuple(record_from_row(self._row_dict(row)) for row in rows)

    def list_all(
        self, *, limit: int = 10_000
    ) -> tuple[PaperTradePerformanceRecord, ...]:
        sql = f"""
            SELECT alert_id, user_id, symbol, setup_name, signal_date,
                   entry_triggered_at, entry_price, stop_price, target_price,
                   exit_at, exit_price, status, outcome_return_pct, holding_days,
                   risk_gate_action, market_regime, volume_ratio, rs_percentile,
                   created_at
            FROM {_TABLE}
            ORDER BY created_at DESC
            FETCH FIRST :limit ROWS ONLY
        """
        rows = self._fetchall(sql, {"limit": limit})
        return tuple(record_from_row(self._row_dict(row)) for row in rows)

    def count_all(self) -> int:
        sql = f"SELECT COUNT(*) FROM {_TABLE}"
        row = self._fetchone(sql, {})
        return int(row[0]) if row else 0

    def latest_updated_at(self) -> datetime | None:
        sql = f"SELECT MAX(updated_at) FROM {_TABLE}"
        row = self._fetchone(sql, {})
        if row is None or row[0] is None:
            return None
        return row[0]

    def _execute(self, sql: str, params: dict[str, Any]) -> None:
        with self._client.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
            conn.commit()

    def _fetchone(self, sql: str, params: dict[str, Any]) -> tuple | None:
        with self._client.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchone()

    def _fetchall(self, sql: str, params: dict[str, Any]) -> list[tuple]:
        with self._client.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                return list(cursor.fetchall())

    @staticmethod
    def _row_dict(row: tuple) -> dict[str, Any]:
        (
            alert_id,
            user_id,
            symbol,
            setup_name,
            signal_date,
            entry_triggered_at,
            entry_price,
            stop_price,
            target_price,
            exit_at,
            exit_price,
            status,
            outcome_return_pct,
            holding_days,
            risk_gate_action,
            market_regime,
            volume_ratio,
            rs_percentile,
            created_at,
        ) = row
        return {
            "alert_id": alert_id,
            "user_id": user_id,
            "symbol": symbol,
            "setup_name": setup_name,
            "signal_date": signal_date,
            "entry_triggered_at": entry_triggered_at,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "exit_at": exit_at,
            "exit_price": exit_price,
            "status": status,
            "outcome_return_pct": outcome_return_pct,
            "holding_days": holding_days,
            "risk_gate_action": risk_gate_action,
            "market_regime": market_regime,
            "volume_ratio": volume_ratio,
            "rs_percentile": rs_percentile,
            "created_at": created_at,
        }
