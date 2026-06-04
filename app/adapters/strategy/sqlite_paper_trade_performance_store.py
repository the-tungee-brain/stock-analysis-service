"""SQLite paper-trading performance store for tests and local dev."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from trade_planner.alerts.paper_trade_models import PaperTradePerformanceRecord
from app.adapters.strategy.paper_trade_performance_mappers import (
    record_from_row,
    record_to_row,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS momentum_breakout_paper_trade (
    alert_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    setup_name TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    entry_triggered_at TEXT,
    entry_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    exit_at TEXT,
    exit_price REAL,
    status TEXT NOT NULL,
    outcome_return_pct REAL,
    holding_days INTEGER,
    risk_gate_action TEXT,
    market_regime TEXT,
    volume_ratio REAL,
    rs_percentile REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mb_paper_trade_user_created
    ON momentum_breakout_paper_trade (user_id, created_at);
"""


def _dt_iso(value: Any) -> str:
    from datetime import date, datetime

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class SqlitePaperTradePerformanceStore:
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

    def get(self, user_id: str, alert_id: str) -> PaperTradePerformanceRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM momentum_breakout_paper_trade WHERE user_id = ? AND alert_id = ?",
                (user_id, alert_id),
            ).fetchone()
        if row is None:
            return None
        return record_from_row(dict(row))

    def save(self, user_id: str, record: PaperTradePerformanceRecord) -> None:
        payload = record_to_row(record)
        payload["user_id"] = user_id
        payload["signal_date"] = _dt_iso(payload["signal_date"])
        payload["entry_triggered_at"] = (
            _dt_iso(payload["entry_triggered_at"])
            if payload["entry_triggered_at"]
            else None
        )
        payload["exit_at"] = _dt_iso(payload["exit_at"]) if payload["exit_at"] else None
        payload["created_at"] = _dt_iso(payload["created_at"])
        from datetime import datetime, timezone

        payload["updated_at"] = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO momentum_breakout_paper_trade (
                    alert_id, user_id, symbol, setup_name, signal_date,
                    entry_triggered_at, entry_price, stop_price, target_price,
                    exit_at, exit_price, status, outcome_return_pct, holding_days,
                    risk_gate_action, market_regime, volume_ratio, rs_percentile,
                    created_at, updated_at
                ) VALUES (
                    :alert_id, :user_id, :symbol, :setup_name, :signal_date,
                    :entry_triggered_at, :entry_price, :stop_price, :target_price,
                    :exit_at, :exit_price, :status, :outcome_return_pct, :holding_days,
                    :risk_gate_action, :market_regime, :volume_ratio, :rs_percentile,
                    :created_at, :updated_at
                )
                ON CONFLICT(alert_id) DO UPDATE SET
                    symbol = excluded.symbol,
                    setup_name = excluded.setup_name,
                    signal_date = excluded.signal_date,
                    entry_triggered_at = excluded.entry_triggered_at,
                    entry_price = excluded.entry_price,
                    stop_price = excluded.stop_price,
                    target_price = excluded.target_price,
                    exit_at = excluded.exit_at,
                    exit_price = excluded.exit_price,
                    status = excluded.status,
                    outcome_return_pct = excluded.outcome_return_pct,
                    holding_days = excluded.holding_days,
                    risk_gate_action = excluded.risk_gate_action,
                    market_regime = excluded.market_regime,
                    volume_ratio = excluded.volume_ratio,
                    rs_percentile = excluded.rs_percentile,
                    updated_at = excluded.updated_at
                """,
                payload,
            )
            conn.commit()

    def list_for_user(
        self, user_id: str, *, limit: int = 500
    ) -> tuple[PaperTradePerformanceRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM momentum_breakout_paper_trade
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return tuple(record_from_row(dict(row)) for row in rows)

    def list_all(
        self, *, limit: int = 10_000
    ) -> tuple[PaperTradePerformanceRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM momentum_breakout_paper_trade
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(record_from_row(dict(row)) for row in rows)

    def count_all(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM momentum_breakout_paper_trade"
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def latest_updated_at(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(updated_at) AS latest FROM momentum_breakout_paper_trade"
            ).fetchone()
        if row is None or row["latest"] is None:
            return None
        return str(row["latest"])
