from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ranking_pipeline.config import RankingPipelineConfig, default_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EmergingLeadersValidationStore:
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
            / "emerging_leaders_validation_schema.sql"
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

    def has_snapshot_date(self, snapshot_date: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM emerging_leaders_daily_snapshots "
                "WHERE snapshot_date = ? LIMIT 1",
                (snapshot_date,),
            ).fetchone()
            return row is not None

    def insert_snapshot_rows(
        self,
        snapshot_date: str,
        rows: list[dict[str, Any]],
    ) -> int:
        if not rows:
            return 0
        created_at = _utc_now()
        payload = [
            (
                snapshot_date,
                str(r["symbol"]).upper(),
                int(r["rank"]),
                int(r["setup_score"]),
                int(r["compression_velocity"]),
                float(r["setup_purity"]),
                str(r["stage"]),
                created_at,
            )
            for r in rows
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO emerging_leaders_daily_snapshots "
                "(snapshot_date, symbol, rank, setup_score, compression_velocity, "
                "setup_purity, stage, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                payload,
            )
            conn.commit()
        return len(payload)

    def list_snapshots_pending_returns(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.snapshot_date, s.symbol, s.rank, s.setup_score,
                       s.compression_velocity, s.setup_purity, s.stage
                FROM emerging_leaders_daily_snapshots s
                LEFT JOIN emerging_leaders_forward_outcomes o ON o.snapshot_id = s.id
                WHERE o.snapshot_id IS NULL
                ORDER BY s.snapshot_date, s.rank
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def upsert_forward_outcomes(self, outcomes: list[dict[str, Any]]) -> int:
        if not outcomes:
            return 0
        computed_at = _utc_now()
        payload = [
            (
                int(o["snapshot_id"]),
                o.get("ret_5d"),
                o.get("ret_10d"),
                o.get("ret_20d"),
                o.get("spy_ret_5d"),
                o.get("spy_ret_10d"),
                o.get("spy_ret_20d"),
                o.get("excess_ret_5d"),
                o.get("excess_ret_10d"),
                o.get("excess_ret_20d"),
                o.get("universe_pct_rank_5d"),
                o.get("universe_pct_rank_10d"),
                o.get("universe_pct_rank_20d"),
                computed_at,
            )
            for o in outcomes
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO emerging_leaders_forward_outcomes "
                "(snapshot_id, ret_5d, ret_10d, ret_20d, "
                "spy_ret_5d, spy_ret_10d, spy_ret_20d, "
                "excess_ret_5d, excess_ret_10d, excess_ret_20d, "
                "universe_pct_rank_5d, universe_pct_rank_10d, universe_pct_rank_20d, "
                "computed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                payload,
            )
            conn.commit()
        return len(payload)

    def load_labeled_rows(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.snapshot_date, s.symbol, s.rank, s.setup_score,
                       s.compression_velocity, s.setup_purity, s.stage,
                       o.ret_5d, o.ret_10d, o.ret_20d,
                       o.spy_ret_5d, o.spy_ret_10d, o.spy_ret_20d,
                       o.excess_ret_5d, o.excess_ret_10d, o.excess_ret_20d,
                       o.universe_pct_rank_5d, o.universe_pct_rank_10d,
                       o.universe_pct_rank_20d
                FROM emerging_leaders_daily_snapshots s
                INNER JOIN emerging_leaders_forward_outcomes o ON o.snapshot_id = s.id
                WHERE o.ret_20d IS NOT NULL
                ORDER BY s.snapshot_date, s.rank
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def summary_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            snap = conn.execute(
                "SELECT COUNT(*) AS c FROM emerging_leaders_daily_snapshots"
            ).fetchone()
            labeled = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM emerging_leaders_forward_outcomes
                WHERE ret_20d IS NOT NULL
                """
            ).fetchone()
            dates = conn.execute(
                "SELECT COUNT(DISTINCT snapshot_date) AS c "
                "FROM emerging_leaders_daily_snapshots"
            ).fetchone()
            return {
                "snapshot_rows": int(snap["c"] if snap else 0),
                "labeled_rows": int(labeled["c"] if labeled else 0),
                "snapshot_dates": int(dates["c"] if dates else 0),
            }


def open_validation_store(cfg: RankingPipelineConfig | None = None) -> EmergingLeadersValidationStore:
    config = cfg or default_config()
    return EmergingLeadersValidationStore(config.db_path)
