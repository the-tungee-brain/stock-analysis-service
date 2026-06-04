"""Factory for paper-trading performance persistence backends."""

from __future__ import annotations

import os
from pathlib import Path

import oracledb

from trade_planner.alerts.paper_trade_store import (
    InMemoryPaperTradePerformanceStore,
    PaperTradePerformanceStore,
)
from app.adapters.strategy.oracle_paper_trade_performance_store import (
    OraclePaperTradePerformanceStore,
)
from app.adapters.strategy.sqlite_paper_trade_performance_store import (
    SqlitePaperTradePerformanceStore,
)


def build_paper_trade_performance_store(
    db_pool: oracledb.ConnectionPool | None = None,
) -> PaperTradePerformanceStore:
    mode = os.getenv("MB_PAPER_TRADE_STORE", "oracle").strip().lower()
    if mode == "memory":
        return InMemoryPaperTradePerformanceStore()
    if mode == "sqlite":
        path = os.getenv(
            "MB_PAPER_TRADE_SQLITE_PATH",
            str(Path("data") / "momentum_breakout_paper_trades.db"),
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqlitePaperTradePerformanceStore(path)
    if db_pool is None:
        raise ValueError("Oracle paper trade store requires a database connection pool")
    return OraclePaperTradePerformanceStore(db_pool)
