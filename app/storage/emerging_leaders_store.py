from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ranking_pipeline.config import RankingPipelineConfig, default_config


@dataclass(frozen=True, slots=True)
class EmergingLeadersRunRecord:
    run_id: str
    as_of_date: str | None
    generated_at: str
    universe_snapshot_id: str | None
    ranking_run_id: str | None
    symbols_with_data: int
    candidates_scanned: int
    excluded_top_movers: int
    evaluations_computed: int
    status: str
    error_message: str | None
    duration_ms: int


class EmergingLeadersStore:
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
            / "emerging_leaders_schema.sql"
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
                INSERT INTO emerging_leaders_snapshot_runs
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
        universe_snapshot_id: str | None,
        ranking_run_id: str | None,
        symbols_with_data: int,
        candidates_scanned: int,
        excluded_top_movers: int,
        evaluations_computed: int,
        duration_ms: int,
        results: list[dict[str, Any]],
    ) -> None:
        payload = [
            (
                run_id,
                int(row["rank"]),
                str(row["symbol"]).upper(),
                int(row["setup_quality_score"]),
                str(row["setup_stage"]),
                str(row["setup_stage_label"]),
                int(row["compression_velocity"]),
                str(row["compression_velocity_label"]),
                str(row["why_it_ranks"]),
                json.dumps(row["positive_factors"]),
                json.dumps(row["missing_factors"]),
                json.dumps(row["next_confirmation"]),
                json.dumps(row["components"]),
            )
            for row in results
        ]
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE emerging_leaders_snapshot_runs
                SET as_of_date = ?,
                    generated_at = ?,
                    universe_snapshot_id = ?,
                    ranking_run_id = ?,
                    symbols_with_data = ?,
                    candidates_scanned = ?,
                    excluded_top_movers = ?,
                    evaluations_computed = ?,
                    status = 'completed',
                    error_message = NULL,
                    duration_ms = ?
                WHERE run_id = ?
                """,
                (
                    as_of_date,
                    generated_at,
                    universe_snapshot_id,
                    ranking_run_id,
                    int(symbols_with_data),
                    int(candidates_scanned),
                    int(excluded_top_movers),
                    int(evaluations_computed),
                    int(duration_ms),
                    run_id,
                ),
            )
            conn.execute(
                "DELETE FROM emerging_leaders_snapshot_results WHERE run_id = ?",
                (run_id,),
            )
            conn.executemany(
                """
                INSERT INTO emerging_leaders_snapshot_results
                (run_id, rank, symbol, setup_quality_score, setup_stage,
                 setup_stage_label, compression_velocity, compression_velocity_label,
                 why_it_ranks, positive_factors_json, missing_factors_json,
                 next_confirmation_json, components_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                UPDATE emerging_leaders_snapshot_runs
                SET status = 'failed',
                    error_message = ?,
                    duration_ms = ?
                WHERE run_id = ?
                """,
                (error_message, int(duration_ms), run_id),
            )
            conn.commit()

    def get_run(self, run_id: str) -> EmergingLeadersRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM emerging_leaders_snapshot_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def latest_run(self) -> EmergingLeadersRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM emerging_leaders_snapshot_runs
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def latest_completed_run(self) -> EmergingLeadersRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM emerging_leaders_snapshot_runs
                WHERE status = 'completed'
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def list_results(
        self,
        run_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        limit_clause = "" if limit is None else "LIMIT ?"
        params: tuple[Any, ...] = (run_id,) if limit is None else (run_id, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM emerging_leaders_snapshot_results
                WHERE run_id = ?
                ORDER BY rank
                {limit_clause}
                """,
                params,
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "rank": int(row["rank"]),
                "symbol": row["symbol"],
                "setup_quality_score": int(row["setup_quality_score"]),
                "setup_stage": row["setup_stage"],
                "setup_stage_label": row["setup_stage_label"],
                "compression_velocity": int(row["compression_velocity"]),
                "compression_velocity_label": row["compression_velocity_label"],
                "why_it_ranks": row["why_it_ranks"],
                "positive_factors": json.loads(row["positive_factors_json"]),
                "missing_factors": json.loads(row["missing_factors_json"]),
                "next_confirmation": json.loads(row["next_confirmation_json"]),
                "components": json.loads(row["components_json"]),
            }
            for row in rows
        ]

    def _record_from_row(self, row: sqlite3.Row) -> EmergingLeadersRunRecord:
        return EmergingLeadersRunRecord(
            run_id=row["run_id"],
            as_of_date=row["as_of_date"],
            generated_at=row["generated_at"],
            universe_snapshot_id=row["universe_snapshot_id"],
            ranking_run_id=row["ranking_run_id"],
            symbols_with_data=int(row["symbols_with_data"]),
            candidates_scanned=int(row["candidates_scanned"]),
            excluded_top_movers=int(row["excluded_top_movers"]),
            evaluations_computed=int(row["evaluations_computed"]),
            status=row["status"],
            error_message=row["error_message"],
            duration_ms=int(row["duration_ms"]),
        )


def open_emerging_leaders_store(
    cfg: RankingPipelineConfig | None = None,
) -> EmergingLeadersStore:
    config = cfg or default_config()
    return EmergingLeadersStore(config.db_path)
