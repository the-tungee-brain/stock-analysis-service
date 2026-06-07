"""Oracle persistence for ranking universe screening runs."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import oracledb


SCREEN_RUNS_TABLE = "SCREEN_RUNS"
SCREEN_RESULTS_TABLE = "SCREEN_RESULTS"
SCREEN_ERRORS_TABLE = "SCREEN_ERRORS"
SCREEN_CHECKPOINTS_TABLE = "SCREEN_CHECKPOINTS"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ScreenRun:
    run_id: str
    snapshot_id: str
    processed_count: int
    passed_count: int


class OracleScreeningStore:
    def __init__(self, pool: oracledb.ConnectionPool) -> None:
        self._pool = pool
        self._conn: Any | None = None

    def _write_conn(self):
        if self._conn is None:
            self._conn = self._pool.acquire()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def commit(self) -> None:
        self._write_conn().commit()

    def ensure_schema(self) -> None:
        statements = [
            f"""
            CREATE TABLE {SCREEN_RUNS_TABLE} (
                run_id VARCHAR2(64) PRIMARY KEY,
                snapshot_id VARCHAR2(64) NOT NULL,
                status VARCHAR2(24) NOT NULL,
                total_count NUMBER DEFAULT 0 NOT NULL,
                processed_count NUMBER DEFAULT 0 NOT NULL,
                passed_count NUMBER DEFAULT 0 NOT NULL,
                batch_size NUMBER,
                max_workers NUMBER,
                started_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                completed_at TIMESTAMP WITH TIME ZONE
            )
            """,
            f"""
            CREATE TABLE {SCREEN_RESULTS_TABLE} (
                run_id VARCHAR2(64) NOT NULL,
                symbol VARCHAR2(32) NOT NULL,
                last_close NUMBER,
                market_cap NUMBER,
                avg_dollar_volume_20d NUMBER,
                score NUMBER,
                passed_filters NUMBER(1) DEFAULT 0 NOT NULL,
                status VARCHAR2(24) NOT NULL,
                reasons_json CLOB,
                screened_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT screen_results_pk PRIMARY KEY (run_id, symbol)
            )
            """,
            f"""
            CREATE TABLE {SCREEN_ERRORS_TABLE} (
                run_id VARCHAR2(64) NOT NULL,
                symbol VARCHAR2(32) NOT NULL,
                error_type VARCHAR2(128),
                error_message VARCHAR2(1000),
                attempts NUMBER DEFAULT 1 NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT screen_errors_pk PRIMARY KEY (run_id, symbol)
            )
            """,
            f"""
            CREATE TABLE {SCREEN_CHECKPOINTS_TABLE} (
                run_id VARCHAR2(64) PRIMARY KEY,
                processed_count NUMBER DEFAULT 0 NOT NULL,
                passed_count NUMBER DEFAULT 0 NOT NULL,
                rss_mb NUMBER,
                checkpoint_json CLOB,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
            )
            """,
        ]
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            for statement in statements:
                try:
                    cur.execute(statement)
                except oracledb.DatabaseError as exc:
                    if "ORA-00955" not in str(exc):
                        raise
            conn.commit()

    def start_or_resume_run(
        self,
        *,
        snapshot_id: str,
        total_count: int,
        batch_size: int,
        max_workers: int,
        resume: bool,
    ) -> ScreenRun:
        conn = self._write_conn()
        cur = conn.cursor()
        if resume:
            row = cur.execute(
                f"""
                SELECT run_id, snapshot_id, processed_count, passed_count
                FROM {SCREEN_RUNS_TABLE}
                WHERE status IN ('running', 'interrupted')
                ORDER BY updated_at DESC
                FETCH FIRST 1 ROWS ONLY
                """
            ).fetchone()
            if row:
                run_id = str(row[0])
                cur.execute(
                    f"""
                    UPDATE {SCREEN_RUNS_TABLE}
                    SET status = 'running',
                        total_count = :total_count,
                        batch_size = :batch_size,
                        max_workers = :max_workers,
                        updated_at = SYSTIMESTAMP
                    WHERE run_id = :run_id
                    """,
                    {
                        "run_id": run_id,
                        "total_count": total_count,
                        "batch_size": batch_size,
                        "max_workers": max_workers,
                    },
                )
                conn.commit()
                return ScreenRun(
                    run_id=run_id,
                    snapshot_id=str(row[1] or snapshot_id),
                    processed_count=int(row[2] or 0),
                    passed_count=int(row[3] or 0),
                )

        run_id = f"screen-{_utc_now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        cur.execute(
            f"""
            INSERT INTO {SCREEN_RUNS_TABLE}
                (run_id, snapshot_id, status, total_count, processed_count,
                 passed_count, batch_size, max_workers)
            VALUES
                (:run_id, :snapshot_id, 'running', :total_count, 0, 0,
                 :batch_size, :max_workers)
            """,
            {
                "run_id": run_id,
                "snapshot_id": snapshot_id,
                "total_count": total_count,
                "batch_size": batch_size,
                "max_workers": max_workers,
            },
        )
        conn.commit()
        return ScreenRun(run_id=run_id, snapshot_id=snapshot_id, processed_count=0, passed_count=0)

    def completed_symbols_for(self, run_id: str, symbols: list[str]) -> set[str]:
        if not symbols:
            return set()
        params: dict[str, Any] = {"run_id": run_id}
        placeholders: list[str] = []
        for idx, symbol in enumerate(symbols):
            key = f"symbol_{idx}"
            params[key] = symbol.strip().upper()
            placeholders.append(f":{key}")
        with self._pool.acquire() as conn:
            rows = conn.cursor().execute(
                f"""
                SELECT symbol
                FROM {SCREEN_RESULTS_TABLE}
                WHERE run_id = :run_id
                  AND symbol IN ({", ".join(placeholders)})
                """,
                params,
            ).fetchall()
        return {str(row[0]).strip().upper() for row in rows}

    def upsert_result(self, run_id: str, result: dict[str, Any]) -> None:
        payload = {
            "run_id": run_id,
            "symbol": result["symbol"],
            "last_close": result.get("last_close"),
            "market_cap": result.get("market_cap"),
            "avg_dollar_volume_20d": result.get("avg_dollar_volume_20d"),
            "score": result.get("avg_dollar_volume_20d"),
            "passed_filters": 1 if result.get("passed_filters") else 0,
            "status": "passed" if result.get("passed_filters") else "failed",
            "reasons_json": json.dumps(result.get("reasons", {})),
        }
        conn = self._write_conn()
        cur = conn.cursor()
        cur.setinputsizes(reasons_json=oracledb.DB_TYPE_CLOB)
        cur.execute(
            f"""
                MERGE INTO {SCREEN_RESULTS_TABLE} t
                USING (
                    SELECT
                        :run_id AS run_id,
                        :symbol AS symbol,
                        :last_close AS last_close,
                        :market_cap AS market_cap,
                        :avg_dollar_volume_20d AS avg_dollar_volume_20d,
                        :score AS score,
                        :passed_filters AS passed_filters,
                        :status AS status,
                        :reasons_json AS reasons_json
                    FROM dual
                ) s
                ON (t.run_id = s.run_id AND t.symbol = s.symbol)
                WHEN MATCHED THEN UPDATE SET
                    t.last_close = s.last_close,
                    t.market_cap = s.market_cap,
                    t.avg_dollar_volume_20d = s.avg_dollar_volume_20d,
                    t.score = s.score,
                    t.passed_filters = s.passed_filters,
                    t.status = s.status,
                    t.reasons_json = s.reasons_json,
                    t.updated_at = SYSTIMESTAMP
                WHEN NOT MATCHED THEN INSERT (
                    run_id, symbol, last_close, market_cap, avg_dollar_volume_20d,
                    score, passed_filters, status, reasons_json
                ) VALUES (
                    s.run_id, s.symbol, s.last_close, s.market_cap, s.avg_dollar_volume_20d,
                    s.score, s.passed_filters, s.status, s.reasons_json
                )
            """,
            payload,
        )

    def upsert_error(self, run_id: str, symbol: str, error: str) -> None:
        self._write_conn().cursor().execute(
            f"""
                MERGE INTO {SCREEN_ERRORS_TABLE} t
                USING (
                    SELECT
                        :run_id AS run_id,
                        :symbol AS symbol,
                        :error_message AS error_message
                    FROM dual
                ) s
                ON (t.run_id = s.run_id AND t.symbol = s.symbol)
                WHEN MATCHED THEN UPDATE SET
                    t.error_message = s.error_message,
                    t.attempts = t.attempts + 1,
                    t.updated_at = SYSTIMESTAMP
                WHEN NOT MATCHED THEN INSERT (
                    run_id, symbol, error_type, error_message
                ) VALUES (
                    s.run_id, s.symbol, 'screen_error', s.error_message
                )
            """,
            {"run_id": run_id, "symbol": symbol, "error_message": error[:1000]},
        )

    def update_progress(
        self,
        run_id: str,
        *,
        processed_count: int,
        passed_count: int,
        rss_mb: float,
        commit: bool,
    ) -> None:
        checkpoint = json.dumps(
            {
                "processed_count": processed_count,
                "passed_count": passed_count,
                "rss_mb": rss_mb,
                "updated_at": _utc_now().isoformat(),
            }
        )
        conn = self._write_conn()
        cur = conn.cursor()
        cur.execute(
            f"""
                UPDATE {SCREEN_RUNS_TABLE}
                SET processed_count = :processed_count,
                    passed_count = :passed_count,
                    updated_at = SYSTIMESTAMP
                WHERE run_id = :run_id
            """,
            {
                "run_id": run_id,
                "processed_count": processed_count,
                "passed_count": passed_count,
            },
        )
        cur.setinputsizes(checkpoint_json=oracledb.DB_TYPE_CLOB)
        cur.execute(
            f"""
                MERGE INTO {SCREEN_CHECKPOINTS_TABLE} t
                USING (
                    SELECT
                        :run_id AS run_id,
                        :processed_count AS processed_count,
                        :passed_count AS passed_count,
                        :rss_mb AS rss_mb,
                        :checkpoint_json AS checkpoint_json
                    FROM dual
                ) s
                ON (t.run_id = s.run_id)
                WHEN MATCHED THEN UPDATE SET
                    t.processed_count = s.processed_count,
                    t.passed_count = s.passed_count,
                    t.rss_mb = s.rss_mb,
                    t.checkpoint_json = s.checkpoint_json,
                    t.updated_at = SYSTIMESTAMP
                WHEN NOT MATCHED THEN INSERT (
                    run_id, processed_count, passed_count, rss_mb, checkpoint_json
                ) VALUES (
                    s.run_id, s.processed_count, s.passed_count, s.rss_mb, s.checkpoint_json
                )
            """,
            {
                "run_id": run_id,
                "processed_count": processed_count,
                "passed_count": passed_count,
                "rss_mb": rss_mb,
                "checkpoint_json": checkpoint,
            },
        )
        if commit:
            conn.commit()

    def mark_interrupted(self, run_id: str, *, processed_count: int, passed_count: int) -> None:
        conn = self._write_conn()
        conn.cursor().execute(
            f"""
                UPDATE {SCREEN_RUNS_TABLE}
                SET status = 'interrupted',
                    processed_count = :processed_count,
                    passed_count = :passed_count,
                    updated_at = SYSTIMESTAMP
                WHERE run_id = :run_id
            """,
            {
                "run_id": run_id,
                "processed_count": processed_count,
                "passed_count": passed_count,
            },
        )
        conn.commit()

    def finalize_run(self, run_id: str, *, processed_count: int, passed_count: int) -> None:
        conn = self._write_conn()
        conn.cursor().execute(
            f"""
                UPDATE {SCREEN_RUNS_TABLE}
                SET status = 'completed',
                    processed_count = :processed_count,
                    passed_count = :passed_count,
                    completed_at = SYSTIMESTAMP,
                    updated_at = SYSTIMESTAMP
                WHERE run_id = :run_id
            """,
            {
                "run_id": run_id,
                "processed_count": processed_count,
                "passed_count": passed_count,
            },
        )
        conn.commit()

    def latest_completed_run(self) -> ScreenRun | None:
        with self._pool.acquire() as conn:
            row = conn.cursor().execute(
                f"""
                SELECT run_id, snapshot_id, processed_count, passed_count
                FROM {SCREEN_RUNS_TABLE}
                WHERE status = 'completed'
                  AND passed_count > 0
                ORDER BY completed_at DESC NULLS LAST, updated_at DESC
                FETCH FIRST 1 ROWS ONLY
                """
            ).fetchone()
        if not row:
            return None
        return ScreenRun(
            run_id=str(row[0]),
            snapshot_id=str(row[1]),
            processed_count=int(row[2] or 0),
            passed_count=int(row[3] or 0),
        )

    def iter_results(
        self,
        run_id: str,
        *,
        page_size: int = 500,
    ) -> Iterator[list[dict[str, Any]]]:
        last_symbol = ""
        while True:
            with self._pool.acquire() as conn:
                rows = conn.cursor().execute(
                    f"""
                    SELECT symbol, last_close, market_cap, avg_dollar_volume_20d,
                           passed_filters
                    FROM {SCREEN_RESULTS_TABLE}
                    WHERE run_id = :run_id
                      AND symbol > :last_symbol
                    ORDER BY symbol
                    FETCH FIRST :page_size ROWS ONLY
                    """,
                    {
                        "run_id": run_id,
                        "last_symbol": last_symbol,
                        "page_size": page_size,
                    },
                ).fetchall()
            if not rows:
                break
            page = [
                {
                    "symbol": str(row[0]).strip().upper(),
                    "last_close": row[1],
                    "market_cap": row[2],
                    "avg_dollar_volume_20d": row[3],
                    "passed_filters": bool(row[4]),
                }
                for row in rows
            ]
            last_symbol = page[-1]["symbol"]
            yield page


def build_oracle_pool() -> oracledb.ConnectionPool:
    user = os.getenv("POWERPOCKETDB_USER")
    password = os.getenv("POWERPOCKETDB_PASSWORD")
    dsn = os.getenv("POWERPOCKETDB_TP_TNS")
    if not (user and password and dsn):
        raise RuntimeError(
            "POWERPOCKETDB_USER, POWERPOCKETDB_PASSWORD, and POWERPOCKETDB_TP_TNS "
            "must be set for Oracle universe screening."
        )
    oracledb.defaults.fetch_lobs = False
    return oracledb.create_pool(user=user, password=password, dsn=dsn, min=1, max=2)


def open_oracle_screening_store() -> OracleScreeningStore:
    store = OracleScreeningStore(build_oracle_pool())
    store.ensure_schema()
    return store
