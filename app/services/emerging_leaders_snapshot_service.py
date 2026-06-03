from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.services.emerging_leaders_evaluations import (
    collect_qualifying_emerging_leader_evaluations,
)
from app.storage.emerging_leaders_validation_store import open_validation_store

logger = logging.getLogger(__name__)


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def capture_emerging_leaders_daily_snapshot(
    *,
    snapshot_date: str | None = None,
    force: bool = False,
) -> dict[str, int | str | bool]:
    """
    Persist all qualifying Emerging Leaders evaluations for a trading day.
    Does not alter live ranking or API list logic.
    """
    store = open_validation_store()
    date = snapshot_date or _today_iso()
    if store.has_snapshot_date(date) and not force:
        logger.info("Emerging leaders snapshot already exists for %s", date)
        counts = store.summary_counts()
        return {
            "snapshot_date": date,
            "skipped": True,
            "rows_written": 0,
            **counts,
        }

    evaluations, as_of_date, candidates_scanned, _with_data, _excluded = (
        collect_qualifying_emerging_leader_evaluations()
    )
    effective_date = as_of_date or date

    rows = [
        {
            "symbol": ev.symbol,
            "rank": idx,
            "setup_score": ev.setup_quality_score,
            "compression_velocity": int(ev.components.compression_velocity),
            "setup_purity": float(ev.components.setup_purity_score),
            "stage": ev.setup_stage,
        }
        for idx, ev in enumerate(evaluations, start=1)
    ]

    if effective_date != date:
        if store.has_snapshot_date(effective_date) and not force:
            date = effective_date
            counts = store.summary_counts()
            return {
                "snapshot_date": date,
                "skipped": True,
                "rows_written": 0,
                **counts,
            }
        date = effective_date

    written = store.insert_snapshot_rows(date, rows)
    counts = store.summary_counts()
    logger.info(
        "Emerging leaders snapshot %s: %d rows (candidates=%d)",
        date,
        written,
        candidates_scanned,
    )
    return {
        "snapshot_date": date,
        "skipped": False,
        "rows_written": written,
        "evaluations": len(evaluations),
        "candidates_scanned": candidates_scanned,
        **counts,
    }
