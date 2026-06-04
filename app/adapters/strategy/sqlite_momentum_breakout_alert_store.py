"""SQLite Momentum Breakout alert store for tests and local dev."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from trade_planner.alerts.lifecycle_models import (
    ACTIVE_STATUSES,
    AlertLifecycleEvent,
    MomentumBreakoutAlertRecord,
)
from app.adapters.strategy.momentum_breakout_alert_mappers import (
    ACTIVE_STATUS_VALUES,
    event_from_row,
    event_to_row,
    record_from_row,
    record_to_row,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS momentum_breakout_alert (
    alert_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    setup_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    entry_is_stop INTEGER NOT NULL,
    status TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    triggered_at TEXT,
    exit_at TEXT,
    exit_price REAL,
    outcome_return_pct REAL,
    risk_gate_action TEXT,
    risk_gate_reasons TEXT,
    historical_win_rate REAL,
    historical_profit_factor REAL,
    historical_total_trades INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS momentum_breakout_alert_event (
    event_id TEXT PRIMARY KEY,
    alert_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    price REAL,
    recorded_at TEXT NOT NULL,
    message TEXT
);

CREATE INDEX IF NOT EXISTS idx_mb_alert_user_status
    ON momentum_breakout_alert (user_id, status, created_at);

CREATE INDEX IF NOT EXISTS idx_mb_alert_event_alert
    ON momentum_breakout_alert_event (alert_id, recorded_at);
"""


def _dt_iso(value: Any) -> str:
    from datetime import date, datetime

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class SqliteMomentumBreakoutAlertStore:
    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def get(self, user_id: str, alert_id: str) -> MomentumBreakoutAlertRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM momentum_breakout_alert WHERE user_id = ? AND alert_id = ?",
                (user_id, alert_id),
            ).fetchone()
        if row is None:
            return None
        return record_from_row(dict(row))

    def save(self, user_id: str, record: MomentumBreakoutAlertRecord) -> None:
        payload = record_to_row(record)
        payload["created_at"] = _dt_iso(payload["created_at"])
        payload["signal_date"] = _dt_iso(payload["signal_date"])
        payload["expires_at"] = _dt_iso(payload["expires_at"])
        payload["triggered_at"] = _dt_iso(payload["triggered_at"]) if payload["triggered_at"] else None
        payload["exit_at"] = _dt_iso(payload["exit_at"]) if payload["exit_at"] else None
        from datetime import datetime, timezone

        payload["updated_at"] = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO momentum_breakout_alert (
                    alert_id, user_id, symbol, setup_name, created_at, signal_date,
                    entry_price, stop_price, target_price, entry_is_stop, status,
                    expires_at, triggered_at, exit_at, exit_price, outcome_return_pct,
                    risk_gate_action, risk_gate_reasons, historical_win_rate,
                    historical_profit_factor, historical_total_trades, updated_at
                ) VALUES (
                    :alert_id, :user_id, :symbol, :setup_name, :created_at, :signal_date,
                    :entry_price, :stop_price, :target_price, :entry_is_stop, :status,
                    :expires_at, :triggered_at, :exit_at, :exit_price, :outcome_return_pct,
                    :risk_gate_action, :risk_gate_reasons, :historical_win_rate,
                    :historical_profit_factor, :historical_total_trades, :updated_at
                )
                ON CONFLICT(alert_id) DO UPDATE SET
                    symbol = excluded.symbol,
                    setup_name = excluded.setup_name,
                    entry_price = excluded.entry_price,
                    stop_price = excluded.stop_price,
                    target_price = excluded.target_price,
                    entry_is_stop = excluded.entry_is_stop,
                    status = excluded.status,
                    expires_at = excluded.expires_at,
                    triggered_at = excluded.triggered_at,
                    exit_at = excluded.exit_at,
                    exit_price = excluded.exit_price,
                    outcome_return_pct = excluded.outcome_return_pct,
                    risk_gate_action = excluded.risk_gate_action,
                    risk_gate_reasons = excluded.risk_gate_reasons,
                    historical_win_rate = excluded.historical_win_rate,
                    historical_profit_factor = excluded.historical_profit_factor,
                    historical_total_trades = excluded.historical_total_trades,
                    updated_at = excluded.updated_at
                """,
                payload,
            )
            conn.commit()

    def has_active_for_symbol(
        self, user_id: str, symbol: str, setup_name: str
    ) -> bool:
        placeholders = ",".join("?" for _ in ACTIVE_STATUS_VALUES)
        params: list[Any] = [user_id, symbol.upper(), setup_name, *ACTIVE_STATUS_VALUES]
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT 1 FROM momentum_breakout_alert
                WHERE user_id = ? AND symbol = ? AND setup_name = ?
                  AND status IN ({placeholders})
                LIMIT 1
                """,
                params,
            ).fetchone()
        return row is not None

    def list_active(self, user_id: str) -> tuple[MomentumBreakoutAlertRecord, ...]:
        placeholders = ",".join("?" for _ in ACTIVE_STATUS_VALUES)
        params: list[Any] = [user_id, *ACTIVE_STATUS_VALUES]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM momentum_breakout_alert
                WHERE user_id = ? AND status IN ({placeholders})
                ORDER BY created_at DESC
                """,
                params,
            ).fetchall()
        return tuple(record_from_row(dict(row)) for row in rows)

    def list_all_active(self) -> tuple[MomentumBreakoutAlertRecord, ...]:
        placeholders = ",".join("?" for _ in ACTIVE_STATUS_VALUES)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM momentum_breakout_alert
                WHERE status IN ({placeholders})
                ORDER BY created_at DESC
                """,
                list(ACTIVE_STATUS_VALUES),
            ).fetchall()
        return tuple(record_from_row(dict(row)) for row in rows)

    def list_history(
        self, user_id: str, *, limit: int = 100
    ) -> tuple[MomentumBreakoutAlertRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM momentum_breakout_alert
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return tuple(record_from_row(dict(row)) for row in rows)

    def append_event(self, user_id: str, event: AlertLifecycleEvent) -> None:
        payload = event_to_row(user_id, event)
        payload["recorded_at"] = _dt_iso(payload["recorded_at"])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO momentum_breakout_alert_event (
                    event_id, alert_id, user_id, event_type, from_status, to_status,
                    price, recorded_at, message
                ) VALUES (
                    :event_id, :alert_id, :user_id, :event_type, :from_status, :to_status,
                    :price, :recorded_at, :message
                )
                """,
                payload,
            )
            conn.commit()

    def list_events(
        self, user_id: str, alert_id: str
    ) -> tuple[AlertLifecycleEvent, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM momentum_breakout_alert_event
                WHERE user_id = ? AND alert_id = ?
                ORDER BY recorded_at ASC
                """,
                (user_id, alert_id),
            ).fetchall()
        return tuple(event_from_row(dict(row)) for row in rows)
