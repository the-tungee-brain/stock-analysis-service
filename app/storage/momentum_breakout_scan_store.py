from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ranking_pipeline.config import RankingPipelineConfig, default_config


@dataclass(frozen=True, slots=True)
class MomentumBreakoutScanRunRecord:
    run_id: str
    as_of_date: str | None
    generated_at: str
    ranking_run_id: str | None
    ranking_snapshot_id: str | None
    universe_source: str | None
    selection_method: str | None
    total_ranked_symbols: int
    total_eligible_symbols: int
    symbols_scanned: int
    excluded_by_cap: int
    valid_setups_found: int
    tradable_candidates_found: int
    blocked_candidates_count: int
    status: str
    error_message: str | None
    duration_ms: int


class MomentumBreakoutScanStore:
    def __init__(self, db_path: Path | None = None) -> None:
        cfg = default_config()
        self.db_path = db_path or cfg.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "ranking_pipeline"
            / "storage"
            / "momentum_breakout_scan_schema.sql"
        )
        sql = schema_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(sql)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def start_run(self, *, run_id: str, generated_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO momentum_breakout_scan_runs
                (run_id, generated_at, status)
                VALUES (?, ?, 'running')
                """,
                (run_id, generated_at),
            )
            conn.commit()

    def complete_run(
        self,
        *,
        run_id: str,
        as_of_date: str | None,
        generated_at: str,
        ranking_run_id: str | None,
        ranking_snapshot_id: str | None,
        universe_source: str | None,
        selection_method: str | None,
        total_ranked_symbols: int,
        total_eligible_symbols: int,
        symbols_scanned: int,
        excluded_by_cap: int,
        valid_setups_found: int,
        tradable_candidates_found: int,
        blocked_candidates_count: int,
        duration_ms: int,
        results: list[dict[str, Any]],
    ) -> None:
        payload = [
            (
                run_id,
                int(row["rank"]),
                str(row["symbol"]).upper(),
                float(row["entry_price"]),
                float(row["stop_price"]),
                float(row["target_price"]),
                float(row["risk_reward"]),
                row.get("historical_win_rate"),
                row.get("historical_profit_factor"),
                row.get("historical_total_trades"),
                float(row["setup_score"]),
                float(row["stop_distance_pct"]),
                row.get("volume_ratio"),
                row.get("rs_percentile"),
                row.get("market_regime"),
                json.dumps(row["risk_gate"]),
            )
            for row in results
        ]
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE momentum_breakout_scan_runs
                SET as_of_date = ?,
                    generated_at = ?,
                    ranking_run_id = ?,
                    ranking_snapshot_id = ?,
                    universe_source = ?,
                    selection_method = ?,
                    total_ranked_symbols = ?,
                    total_eligible_symbols = ?,
                    symbols_scanned = ?,
                    excluded_by_cap = ?,
                    valid_setups_found = ?,
                    tradable_candidates_found = ?,
                    blocked_candidates_count = ?,
                    status = 'completed',
                    error_message = NULL,
                    duration_ms = ?
                WHERE run_id = ?
                """,
                (
                    as_of_date,
                    generated_at,
                    ranking_run_id,
                    ranking_snapshot_id,
                    universe_source,
                    selection_method,
                    int(total_ranked_symbols),
                    int(total_eligible_symbols),
                    int(symbols_scanned),
                    int(excluded_by_cap),
                    int(valid_setups_found),
                    int(tradable_candidates_found),
                    int(blocked_candidates_count),
                    int(duration_ms),
                    run_id,
                ),
            )
            conn.execute(
                "DELETE FROM momentum_breakout_scan_results WHERE run_id = ?",
                (run_id,),
            )
            conn.executemany(
                """
                INSERT INTO momentum_breakout_scan_results
                (run_id, rank, symbol, entry_price, stop_price, target_price,
                 risk_reward, historical_win_rate, historical_profit_factor,
                 historical_total_trades, setup_score, stop_distance_pct,
                 volume_ratio, rs_percentile, market_regime, risk_gate_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()

    def fail_run(
        self,
        *,
        run_id: str,
        error_message: str,
        duration_ms: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE momentum_breakout_scan_runs
                SET status = 'failed',
                    error_message = ?,
                    duration_ms = ?
                WHERE run_id = ?
                """,
                (error_message, int(duration_ms), run_id),
            )
            conn.commit()

    def get_run(self, run_id: str) -> MomentumBreakoutScanRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM momentum_breakout_scan_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def latest_run(self) -> MomentumBreakoutScanRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM momentum_breakout_scan_runs
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def list_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM momentum_breakout_scan_results
                WHERE run_id = ?
                ORDER BY rank
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "rank": int(row["rank"]),
                "symbol": row["symbol"],
                "entry_price": float(row["entry_price"]),
                "stop_price": float(row["stop_price"]),
                "target_price": float(row["target_price"]),
                "risk_reward": float(row["risk_reward"]),
                "historical_win_rate": row["historical_win_rate"],
                "historical_profit_factor": row["historical_profit_factor"],
                "historical_total_trades": row["historical_total_trades"],
                "setup_score": float(row["setup_score"]),
                "stop_distance_pct": float(row["stop_distance_pct"]),
                "volume_ratio": row["volume_ratio"],
                "rs_percentile": row["rs_percentile"],
                "market_regime": row["market_regime"],
                "risk_gate": json.loads(row["risk_gate_json"]),
            }
            for row in rows
        ]

    def _record_from_row(self, row: sqlite3.Row) -> MomentumBreakoutScanRunRecord:
        return MomentumBreakoutScanRunRecord(
            run_id=row["run_id"],
            as_of_date=row["as_of_date"],
            generated_at=row["generated_at"],
            ranking_run_id=row["ranking_run_id"],
            ranking_snapshot_id=row["ranking_snapshot_id"],
            universe_source=row["universe_source"],
            selection_method=row["selection_method"],
            total_ranked_symbols=int(row["total_ranked_symbols"]),
            total_eligible_symbols=int(row["total_eligible_symbols"]),
            symbols_scanned=int(row["symbols_scanned"]),
            excluded_by_cap=int(row["excluded_by_cap"]),
            valid_setups_found=int(row["valid_setups_found"]),
            tradable_candidates_found=int(row["tradable_candidates_found"]),
            blocked_candidates_count=int(row["blocked_candidates_count"]),
            status=row["status"],
            error_message=row["error_message"],
            duration_ms=int(row["duration_ms"]),
        )


def open_momentum_breakout_scan_store(
    cfg: RankingPipelineConfig | None = None,
) -> MomentumBreakoutScanStore:
    config = cfg or default_config()
    return MomentumBreakoutScanStore(config.db_path)
