"""Detect whether Momentum Breakout DDL appears applied."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import oracledb

from app.adapters.strategy.momentum_breakout_store_config import (
    resolve_alert_sqlite_path,
    resolve_alert_store_mode,
    resolve_paper_trade_sqlite_path,
    resolve_paper_trade_store_mode,
)


@dataclass(frozen=True, slots=True)
class DdlStatus:
    required: bool
    applied: bool | None
    detail: str


def inspect_ddl_status(
    *,
    db_pool: oracledb.ConnectionPool | None = None,
) -> tuple[DdlStatus, DdlStatus]:
    alert_mode = resolve_alert_store_mode()
    paper_mode = resolve_paper_trade_store_mode()

    if alert_mode == "memory" or paper_mode == "memory":
        return (
            DdlStatus(required=False, applied=None, detail="memory store — DDL not used"),
            DdlStatus(required=False, applied=None, detail="memory store — DDL not used"),
        )

    if alert_mode == "sqlite" or paper_mode == "sqlite":
        return (
            _sqlite_alert_ddl_status(),
            _sqlite_paper_ddl_status(),
        )

    return _oracle_ddl_status(db_pool)


def _sqlite_alert_ddl_status() -> DdlStatus:
    path = Path(resolve_alert_sqlite_path())
    if not path.exists():
        return DdlStatus(
            required=True,
            applied=False,
            detail=f"SQLite alert DB not found at {path}",
        )
    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "momentum_breakout_alert" not in tables:
            return DdlStatus(
                required=True,
                applied=False,
                detail="momentum_breakout_alert table missing",
            )
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(momentum_breakout_alert)"
            ).fetchall()
        }
        missing = [
            col
            for col in ("market_regime", "volume_ratio", "rs_percentile")
            if col not in columns
        ]
        if missing:
            return DdlStatus(
                required=True,
                applied=False,
                detail=f"alert table missing columns: {', '.join(missing)}",
            )
    return DdlStatus(required=True, applied=True, detail="SQLite alert schema present")


def _sqlite_paper_ddl_status() -> DdlStatus:
    path = Path(resolve_paper_trade_sqlite_path())
    if not path.exists():
        return DdlStatus(
            required=True,
            applied=False,
            detail=f"SQLite paper DB not found at {path}",
        )
    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "momentum_breakout_paper_trade" not in tables:
            return DdlStatus(
                required=True,
                applied=False,
                detail="momentum_breakout_paper_trade table missing",
            )
    return DdlStatus(required=True, applied=True, detail="SQLite paper schema present")


def _oracle_ddl_status(
    db_pool: oracledb.ConnectionPool | None,
) -> tuple[DdlStatus, DdlStatus]:
    if db_pool is None:
        return (
            DdlStatus(required=True, applied=None, detail="Oracle pool unavailable"),
            DdlStatus(required=True, applied=None, detail="Oracle pool unavailable"),
        )
    try:
        with db_pool.acquire() as conn:
            with conn.cursor() as cur:
                alert_ok = _oracle_table_exists(cur, "MOMENTUM_BREAKOUT_ALERT")
                paper_ok = _oracle_table_exists(cur, "MOMENTUM_BREAKOUT_PAPER_TRADE")
                alert_cols_ok = True
                if alert_ok:
                    alert_cols_ok = _oracle_column_exists(
                        cur, "MOMENTUM_BREAKOUT_ALERT", "MARKET_REGIME"
                    )
        alert_status = DdlStatus(
            required=True,
            applied=alert_ok and alert_cols_ok,
            detail="Oracle alert table and context columns present"
            if alert_ok and alert_cols_ok
            else "Oracle alert DDL incomplete",
        )
        paper_status = DdlStatus(
            required=True,
            applied=paper_ok,
            detail="Oracle paper trade table present"
            if paper_ok
            else "MOMENTUM_BREAKOUT_PAPER_TRADE table missing",
        )
        return alert_status, paper_status
    except Exception as exc:  # noqa: BLE001
        msg = f"Oracle DDL check failed: {exc}"
        return (
            DdlStatus(required=True, applied=None, detail=msg),
            DdlStatus(required=True, applied=None, detail=msg),
        )


def _oracle_table_exists(cur: Any, table_name: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM user_tables
        WHERE table_name = :table_name
        FETCH FIRST 1 ROWS ONLY
        """,
        {"table_name": table_name.upper()},
    )
    return cur.fetchone() is not None


def _oracle_column_exists(cur: Any, table_name: str, column_name: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM user_tab_columns
        WHERE table_name = :table_name AND column_name = :column_name
        FETCH FIRST 1 ROWS ONLY
        """,
        {"table_name": table_name.upper(), "column_name": column_name.upper()},
    )
    return cur.fetchone() is not None
