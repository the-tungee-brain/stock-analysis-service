"""Factory for Momentum Breakout alert persistence backends."""

from __future__ import annotations

import os
from pathlib import Path

import oracledb

from trade_planner.alerts.lifecycle_store import (
    InMemoryMomentumBreakoutAlertStore,
    MomentumBreakoutAlertStore,
)
from app.adapters.strategy.oracle_momentum_breakout_alert_store import (
    OracleMomentumBreakoutAlertStore,
)
from app.adapters.strategy.sqlite_momentum_breakout_alert_store import (
    SqliteMomentumBreakoutAlertStore,
)


def build_momentum_breakout_alert_store(
    db_pool: oracledb.ConnectionPool | None = None,
) -> MomentumBreakoutAlertStore:
    mode = os.getenv("MB_ALERT_STORE", "oracle").strip().lower()
    if mode == "memory":
        return InMemoryMomentumBreakoutAlertStore()
    if mode == "sqlite":
        path = os.getenv(
            "MB_ALERT_SQLITE_PATH",
            str(Path("data") / "momentum_breakout_alerts.db"),
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteMomentumBreakoutAlertStore(path)
    if db_pool is None:
        raise ValueError("Oracle MB alert store requires a database connection pool")
    return OracleMomentumBreakoutAlertStore(db_pool)
