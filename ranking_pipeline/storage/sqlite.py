"""SQLite persistence for universe snapshots and ranking runs."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterator

from data.paths import active_universe_pointer_path
from ranking_pipeline.config import RankingPipelineConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class UniverseMemberRecord:
    symbol: str
    market_cap: float | None
    avg_dollar_volume_20d: float | None
    ranking_score: float | None


@dataclass(frozen=True, slots=True)
class LatestRankingRunMeta:
    run_id: str
    as_of_date: str
    created_at: str
    universe_snapshot_id: str | None
    symbol_count: int


@dataclass(frozen=True, slots=True)
class RankingResultRecord:
    symbol: str
    rank: int
    final_score: float
    composite_score: float | None


class RankingStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        sql = schema_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(sql)
            self._migrate(conn)
            conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(ranking_runs)").fetchall()}
        if "regime_id" not in cols:
            conn.execute("ALTER TABLE ranking_runs ADD COLUMN regime_id TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_universe_snapshot(
        self,
        snapshot_id: str,
        members: list[dict[str, Any]],
    ) -> None:
        passed = [m for m in members if m.get("passed_filters")]
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO universe_snapshots (snapshot_id, created_at, symbol_count) "
                "VALUES (?, ?, ?)",
                (snapshot_id, _utc_now(), len(passed)),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO universe_members "
                "(snapshot_id, symbol, last_close, market_cap, avg_dollar_volume_20d, passed_filters) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        snapshot_id,
                        m["symbol"],
                        m.get("last_close"),
                        m.get("market_cap"),
                        m.get("avg_dollar_volume_20d"),
                        1 if m.get("passed_filters") else 0,
                    )
                    for m in members
                ],
            )
            conn.commit()
        pointer = {"snapshot_id": snapshot_id, "updated_at": _utc_now()}
        path = active_universe_pointer_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pointer), encoding="utf-8")

    def active_snapshot_id(self) -> str | None:
        path = active_universe_pointer_path()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("snapshot_id")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT snapshot_id FROM universe_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row["snapshot_id"] if row else None

    def load_universe_symbols(self, snapshot_id: str | None = None) -> list[str]:
        sid = snapshot_id or self.active_snapshot_id()
        if not sid:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol FROM universe_members "
                "WHERE snapshot_id = ? AND passed_filters = 1 ORDER BY symbol",
                (sid,),
            ).fetchall()
        return [r["symbol"] for r in rows]

    def load_passed_universe_members(
        self,
        snapshot_id: str | None = None,
        *,
        ranking_run_id: str | None = None,
    ) -> list[UniverseMemberRecord]:
        """Passed universe members with liquidity metrics and optional latest ranking score."""
        sid = snapshot_id or self.active_snapshot_id()
        if not sid:
            return []
        run_id = ranking_run_id if ranking_run_id is not None else self.latest_run_id()
        with self._connect() as conn:
            if run_id:
                rows = conn.execute(
                    "SELECT um.symbol, um.market_cap, um.avg_dollar_volume_20d, "
                    "rr.final_score AS ranking_score "
                    "FROM universe_members um "
                    "LEFT JOIN ranking_results rr "
                    "ON rr.symbol = um.symbol AND rr.run_id = ? "
                    "WHERE um.snapshot_id = ? AND um.passed_filters = 1",
                    (run_id, sid),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT symbol, market_cap, avg_dollar_volume_20d, "
                    "NULL AS ranking_score "
                    "FROM universe_members "
                    "WHERE snapshot_id = ? AND passed_filters = 1",
                    (sid,),
                ).fetchall()
        return [
            UniverseMemberRecord(
                symbol=str(r["symbol"]).strip().upper(),
                market_cap=r["market_cap"],
                avg_dollar_volume_20d=r["avg_dollar_volume_20d"],
                ranking_score=r["ranking_score"],
            )
            for r in rows
        ]

    def upsert_ohlcv_sync(self, symbol: str, last_bar_date: str, row_count: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ohlcv_sync (symbol, last_bar_date, row_count, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (symbol, last_bar_date, row_count, _utc_now()),
            )
            conn.commit()

    def save_market_regime_row(
        self,
        date: str,
        regime_id: str,
        regime_multiplier: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO market_regime_daily "
                "(date, regime_id, regime_multiplier, metadata_json) VALUES (?, ?, ?, ?)",
                (date, regime_id, regime_multiplier, json.dumps(metadata or {})),
            )
            conn.commit()

    def get_market_regime(self, as_of_date: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM market_regime_daily WHERE date <= ? ORDER BY date DESC LIMIT 1",
                (as_of_date,),
            ).fetchone()
        return dict(row) if row else None

    def load_adv_by_symbols(
        self,
        snapshot_id: str,
        symbols: list[str],
    ) -> dict[str, float | None]:
        if not symbols:
            return {}
        placeholders = ",".join("?" * len(symbols))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT symbol, avg_dollar_volume_20d FROM universe_members "
                f"WHERE snapshot_id = ? AND symbol IN ({placeholders})",
                [snapshot_id, *symbols],
            ).fetchall()
        return {r["symbol"]: r["avg_dollar_volume_20d"] for r in rows}

    def save_ranking_run(
        self,
        run_id: str,
        as_of_date: str,
        model_backend: str,
        universe_snapshot_id: str | None,
        results: list[dict[str, Any]],
        *,
        regime_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ranking_runs "
                "(run_id, as_of_date, model_backend, universe_snapshot_id, symbol_count, "
                "created_at, regime_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    as_of_date,
                    model_backend,
                    universe_snapshot_id,
                    len(results),
                    _utc_now(),
                    regime_id,
                ),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO ranking_results "
                "(run_id, symbol, rank, composite_score, ml_probability, expected_excess_return, "
                "final_score, contributions_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        r["symbol"],
                        r["rank"],
                        r.get("composite_score"),
                        r.get("ml_probability"),
                        r.get("expected_excess_return"),
                        r["final_score"],
                        json.dumps(r.get("contributions", {})),
                    )
                    for r in results
                ],
            )
            conn.commit()

    def latest_run_id(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id FROM ranking_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row["run_id"] if row else None

    def get_latest_ranking_run(self) -> LatestRankingRunMeta | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id, as_of_date, created_at, universe_snapshot_id, symbol_count "
                "FROM ranking_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return LatestRankingRunMeta(
            run_id=row["run_id"],
            as_of_date=row["as_of_date"],
            created_at=row["created_at"],
            universe_snapshot_id=row["universe_snapshot_id"],
            symbol_count=int(row["symbol_count"]),
        )

    def load_ranking_results_ordered(self, run_id: str) -> list[RankingResultRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, rank, final_score, composite_score "
                "FROM ranking_results WHERE run_id = ? ORDER BY rank ASC",
                (run_id,),
            ).fetchall()
        return [
            RankingResultRecord(
                symbol=str(r["symbol"]).strip().upper(),
                rank=int(r["rank"]),
                final_score=float(r["final_score"]),
                composite_score=r["composite_score"],
            )
            for r in rows
        ]

    def get_symbol_ranking_row(
        self,
        run_id: str,
        symbol: str,
    ) -> dict[str, Any] | None:
        sym = symbol.strip().upper()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ranking_results WHERE run_id = ? AND symbol = ?",
                (run_id, sym),
            ).fetchone()
        if not row:
            return None
        return {
            "symbol": row["symbol"],
            "rank": row["rank"],
            "composite_score": row["composite_score"],
            "ml_probability": row["ml_probability"],
            "expected_excess_return": row["expected_excess_return"],
            "final_score": row["final_score"],
            "contributions": json.loads(row["contributions_json"]),
        }

    def count_ranking_results(self, run_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM ranking_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def get_ranking_results(
        self,
        run_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ranking_results WHERE run_id = ? ORDER BY rank ASC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "symbol": row["symbol"],
                    "rank": row["rank"],
                    "composite_score": row["composite_score"],
                    "ml_probability": row["ml_probability"],
                    "expected_excess_return": row["expected_excess_return"],
                    "final_score": row["final_score"],
                    "contributions": json.loads(row["contributions_json"]),
                }
            )
        return out

    def get_run_meta(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ranking_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def save_backtest_run(
        self,
        *,
        backtest_id: str,
        ranking_run_id: str,
        as_of_date: str,
        top_n: int,
        hold_days: int,
        metrics: Any,
        cost_config: Any,
    ) -> None:
        from ranking_pipeline.backtest.metrics import BacktestMetrics

        m: BacktestMetrics = metrics
        costs = {
            "slippage_bps_per_side": cost_config.slippage_bps_per_side,
            "round_trip_sides": cost_config.round_trip_sides,
            "liquidity_penalty_bps": cost_config.liquidity_penalty_bps,
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO backtest_runs "
                "(backtest_id, ranking_run_id, as_of_date, top_n, hold_days, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (backtest_id, ranking_run_id, as_of_date, top_n, hold_days, _utc_now()),
            )
            conn.execute(
                "INSERT OR REPLACE INTO backtest_metrics "
                "(backtest_id, avg_return, avg_excess_return, hit_rate_vs_spy, sharpe_ratio, "
                "max_drawdown, slippage_bps, costs_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    backtest_id,
                    m.avg_return,
                    m.avg_excess_return,
                    m.hit_rate_vs_spy,
                    m.sharpe_ratio,
                    m.max_drawdown,
                    cost_config.slippage_bps_per_side,
                    json.dumps(costs),
                ),
            )
            conn.commit()

    def get_backtest_metrics(self, backtest_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM backtest_metrics WHERE backtest_id = ?", (backtest_id,)
            ).fetchone()
        return dict(row) if row else None

    def prune_old_runs(self, keep_days: int) -> int:
        cutoff = datetime.now(timezone.utc).timestamp() - keep_days * 86400
        with self._connect() as conn:
            rows = conn.execute("SELECT run_id, created_at FROM ranking_runs").fetchall()
            stale = [
                r["run_id"]
                for r in rows
                if datetime.fromisoformat(r["created_at"]).timestamp() < cutoff
            ]
            for run_id in stale:
                conn.execute("DELETE FROM ranking_results WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM ranking_runs WHERE run_id = ?", (run_id,))
            conn.commit()
        return len(stale)


def open_store(config: RankingPipelineConfig | None = None) -> RankingStore:
    cfg = config or RankingPipelineConfig()
    return RankingStore(cfg.db_path)
