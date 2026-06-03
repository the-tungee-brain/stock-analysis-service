"""Read-only pipeline status from ranking SQLite (no ranking/portfolio logic)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ranking_pipeline.config import default_config


@dataclass(frozen=True)
class PipelineStatusSnapshot:
    last_pipeline_run_time: str | None
    universe_size: int | None
    last_successful_ranking_run: str | None
    last_ranking_run_at: str | None
    last_successful_portfolio_run: str | None
    last_portfolio_run_at: str | None
    system_status: str
    ranking_run_id: str | None
    portfolio_id: str | None
    regime_id: str | None


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_hours(ts: str | None) -> float | None:
    dt = _parse_iso(ts)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


class PipelineStatusReader:
    """Query precomputed run metadata for health and product APIs."""

    def __init__(self, db_path: Path | None = None) -> None:
        cfg = default_config()
        self.db_path = Path(db_path or cfg.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_status(self) -> PipelineStatusSnapshot:
        ranking_run_id = None
        ranking_at = None
        regime_id = None
        portfolio_id = None
        portfolio_at = None
        universe_size = None
        last_pipeline = None

        if not self.db_path.exists():
            return PipelineStatusSnapshot(
                last_pipeline_run_time=None,
                universe_size=None,
                last_successful_ranking_run=None,
                last_ranking_run_at=None,
                last_successful_portfolio_run=None,
                last_portfolio_run_at=None,
                system_status="failing",
                ranking_run_id=None,
                portfolio_id=None,
                regime_id=None,
            )

        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id, created_at, regime_id FROM ranking_runs "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row:
                ranking_run_id = row["run_id"]
                ranking_at = row["created_at"]
                regime_id = row["regime_id"] if "regime_id" in row.keys() else None
                last_pipeline = ranking_at

            prow = conn.execute(
                "SELECT portfolio_id, created_at FROM portfolio_snapshots "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if prow:
                portfolio_id = prow["portfolio_id"]
                portfolio_at = prow["created_at"]
                if not last_pipeline or (portfolio_at and portfolio_at > (ranking_at or "")):
                    last_pipeline = portfolio_at

            snap = conn.execute(
                "SELECT symbol_count FROM universe_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if snap:
                universe_size = int(snap["symbol_count"])

        status = _derive_system_status(ranking_at, portfolio_at)

        return PipelineStatusSnapshot(
            last_pipeline_run_time=last_pipeline,
            universe_size=universe_size,
            last_successful_ranking_run=ranking_run_id,
            last_ranking_run_at=ranking_at,
            last_successful_portfolio_run=portfolio_id,
            last_portfolio_run_at=portfolio_at,
            system_status=status,
            ranking_run_id=ranking_run_id,
            portfolio_id=portfolio_id,
            regime_id=regime_id,
        )


def _derive_system_status(ranking_at: str | None, portfolio_at: str | None) -> str:
    rank_h = _age_hours(ranking_at)
    port_h = _age_hours(portfolio_at)
    if rank_h is None:
        return "failing"
    if rank_h > 72:
        return "failing"
    if rank_h > 36 or (port_h is not None and port_h > 48):
        return "degraded"
    return "ok"
