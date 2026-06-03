"""SQLite persistence for constructed portfolios (same DB as ranking)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ranking_pipeline.portfolio.config import PortfolioConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PortfolioStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def latest_portfolio_id(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT portfolio_id FROM portfolio_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row["portfolio_id"] if row else None

    def load_previous_weights(self, before_date: str) -> dict[str, float]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT portfolio_id FROM portfolio_snapshots "
                "WHERE as_of_date < ? ORDER BY as_of_date DESC LIMIT 1",
                (before_date,),
            ).fetchone()
            if not row:
                return {}
            pid = row["portfolio_id"]
            holdings = conn.execute(
                "SELECT symbol, weight FROM portfolio_holdings WHERE portfolio_id = ?",
                (pid,),
            ).fetchall()
        return {r["symbol"]: float(r["weight"]) for r in holdings}

    def save_portfolio(
        self,
        *,
        portfolio_id: str,
        ranking_run_id: str,
        as_of_date: str,
        sizing_mode: str,
        holdings: list[dict[str, Any]],
        metrics: dict[str, Any],
        trades: list[dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio_snapshots "
                "(portfolio_id, ranking_run_id, as_of_date, sizing_mode, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (portfolio_id, ranking_run_id, as_of_date, sizing_mode, _utc_now()),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio_holdings "
                "(portfolio_id, symbol, weight, final_score, ml_probability, "
                "expected_excess_return, atr_14) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        portfolio_id,
                        h["symbol"],
                        h["weight"],
                        h.get("final_score"),
                        h.get("ml_probability"),
                        h.get("expected_excess_return"),
                        h.get("atr_14"),
                    )
                    for h in holdings
                ],
            )
            conn.execute(
                "INSERT OR REPLACE INTO portfolio_metrics "
                "(portfolio_id, expected_return_5d, expected_excess_5d, portfolio_volatility, "
                "turnover, concentration_hhi, metrics_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    portfolio_id,
                    metrics.get("expected_return_5d"),
                    metrics.get("expected_excess_5d"),
                    metrics.get("portfolio_volatility"),
                    metrics.get("turnover"),
                    metrics.get("concentration_hhi"),
                    json.dumps(metrics),
                ),
            )
            conn.execute(
                "DELETE FROM portfolio_trades WHERE portfolio_id = ?", (portfolio_id,)
            )
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio_trades "
                "(portfolio_id, symbol, side, weight_change, target_weight, previous_weight) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        portfolio_id,
                        t["symbol"],
                        t["side"],
                        t["weight_change"],
                        t["target_weight"],
                        t["previous_weight"],
                    )
                    for t in trades
                ],
            )
            conn.commit()

    def get_latest_portfolio(self) -> dict[str, Any] | None:
        pid = self.latest_portfolio_id()
        if not pid:
            return None
        return self.get_portfolio(pid)

    def get_portfolio(self, portfolio_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            snap = conn.execute(
                "SELECT * FROM portfolio_snapshots WHERE portfolio_id = ?",
                (portfolio_id,),
            ).fetchone()
            if not snap:
                return None
            holdings = conn.execute(
                "SELECT * FROM portfolio_holdings WHERE portfolio_id = ? ORDER BY weight DESC",
                (portfolio_id,),
            ).fetchall()
            met = conn.execute(
                "SELECT * FROM portfolio_metrics WHERE portfolio_id = ?",
                (portfolio_id,),
            ).fetchone()
            trades = conn.execute(
                "SELECT * FROM portfolio_trades WHERE portfolio_id = ?",
                (portfolio_id,),
            ).fetchall()
        return {
            "snapshot": dict(snap),
            "holdings": [dict(h) for h in holdings],
            "metrics": dict(met) if met else {},
            "trades": [dict(t) for t in trades],
        }

    def save_portfolio_backtest(
        self,
        *,
        portfolio_backtest_id: str,
        portfolio_id: str,
        ranking_run_id: str,
        as_of_date: str,
        metrics: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio_backtest_runs "
                "(portfolio_backtest_id, portfolio_id, ranking_run_id, as_of_date, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    portfolio_backtest_id,
                    portfolio_id,
                    ranking_run_id,
                    as_of_date,
                    _utc_now(),
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO portfolio_backtest_metrics "
                "(portfolio_backtest_id, portfolio_return, excess_vs_spy, sharpe_ratio, "
                "max_drawdown, turnover, slippage_bps, metrics_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    portfolio_backtest_id,
                    metrics.get("portfolio_return"),
                    metrics.get("excess_vs_spy"),
                    metrics.get("sharpe_ratio"),
                    metrics.get("max_drawdown"),
                    metrics.get("turnover"),
                    metrics.get("slippage_bps"),
                    json.dumps(metrics),
                ),
            )
            conn.commit()


def open_portfolio_store(config: PortfolioConfig | None = None) -> PortfolioStore:
    from ranking_pipeline.portfolio.config import default_portfolio_config

    cfg = config or default_portfolio_config()
    return PortfolioStore(cfg.db_path)
