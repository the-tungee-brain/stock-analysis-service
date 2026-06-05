from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.builders.emerging_leaders_engine import (
    STAGE_LABELS,
    compression_velocity_label,
)
from app.services.emerging_leaders_evaluations import (
    score_emerging_leader_candidates,
    select_emerging_leader_candidates,
)
from app.storage.emerging_leaders_store import (
    EmergingLeadersStore,
    open_emerging_leaders_store,
)
from ranking_pipeline.config import RankingPipelineConfig, default_config
from ranking_pipeline.storage.sqlite import RankingStore, open_store

logger = logging.getLogger(__name__)

RESULT_LIMIT = 100


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(started_at: float) -> int:
    return max(0, int(round((time.perf_counter() - started_at) * 1000)))


def _emergency_cap() -> int | None:
    raw = os.environ.get("EMERGING_LEADERS_PRECOMPUTE_MAX_UNIVERSE")
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("EMERGING_LEADERS_PRECOMPUTE_MAX_UNIVERSE must be positive")
    return value


def _top_mover_exclude_count() -> int:
    return int(os.environ.get("EMERGING_LEADERS_EXCLUDE_TOP_MOVERS", "12"))


def _current_top_movers(store: RankingStore) -> tuple[set[str], str | None, str | None]:
    exclude_n = _top_mover_exclude_count()
    symbols: set[str] = set()
    as_of_date: str | None = None
    run_id = store.latest_run_id()
    if run_id:
        meta = store.get_run_meta(run_id)
        if meta:
            as_of_date = meta.get("as_of_date")
        for row in store.get_ranking_results(run_id, limit=exclude_n):
            symbols.add(str(row["symbol"]).strip().upper())
    return symbols, run_id, as_of_date


def _result_rows(evaluations: list[Any], *, limit: int = RESULT_LIMIT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, ev in enumerate(evaluations[:limit], start=1):
        rows.append(
            {
                "rank": rank,
                "symbol": ev.symbol,
                "setup_quality_score": ev.setup_quality_score,
                "setup_stage": ev.setup_stage,
                "setup_stage_label": STAGE_LABELS[ev.setup_stage],
                "compression_velocity": int(ev.components.compression_velocity),
                "compression_velocity_label": compression_velocity_label(
                    ev.components.compression_velocity
                ),
                "why_it_ranks": ev.why_it_ranks,
                "positive_factors": ev.positive_factors,
                "missing_factors": ev.missing_factors,
                "next_confirmation": ev.next_confirmation,
                "components": asdict(ev.components),
            }
        )
    return rows


def precompute_emerging_leaders_snapshot(
    *,
    cfg: RankingPipelineConfig | None = None,
    ranking_store: RankingStore | None = None,
    snapshot_store: EmergingLeadersStore | None = None,
    max_universe: int | None = None,
) -> dict[str, Any]:
    """
    Score the full qualified Emerging Leaders universe and persist top results.

    `max_universe` is an emergency/resource-control cap. When omitted, the
    environment cap is used only if explicitly configured.
    """
    config = cfg or default_config()
    rank_store = ranking_store or open_store(config)
    el_store = snapshot_store or open_emerging_leaders_store(config)
    run_id = str(uuid4())
    generated_at = _utc_timestamp()
    started_at = time.perf_counter()
    el_store.start_run(run_id=run_id, generated_at=generated_at)

    try:
        cap = max_universe if max_universe is not None else _emergency_cap()
        universe_snapshot_id = rank_store.active_snapshot_id()
        if not universe_snapshot_id:
            raise LookupError("No active ranking universe")

        top_mover_symbols, ranking_run_id, as_of_date = _current_top_movers(rank_store)
        candidates, symbols_with_data = select_emerging_leader_candidates(
            rank_store,
            max_universe=cap,
            top_mover_symbols=top_mover_symbols,
        )
        evaluations = score_emerging_leader_candidates(candidates)
        rows = _result_rows(evaluations)
        duration_ms = _duration_ms(started_at)
        el_store.complete_run(
            run_id=run_id,
            as_of_date=as_of_date,
            generated_at=generated_at,
            universe_snapshot_id=universe_snapshot_id,
            ranking_run_id=ranking_run_id,
            symbols_with_data=symbols_with_data,
            candidates_scanned=len(candidates),
            excluded_top_movers=len(top_mover_symbols),
            evaluations_computed=len(evaluations),
            duration_ms=duration_ms,
            results=rows,
        )
        logger.info(
            "Emerging Leaders precompute %s completed: candidates=%d results=%d",
            run_id,
            len(candidates),
            len(rows),
        )
        return {
            "run_id": run_id,
            "status": "completed",
            "as_of_date": as_of_date,
            "generated_at": generated_at,
            "universe_snapshot_id": universe_snapshot_id,
            "ranking_run_id": ranking_run_id,
            "symbols_with_data": symbols_with_data,
            "candidates_scanned": len(candidates),
            "excluded_top_movers": len(top_mover_symbols),
            "evaluations_computed": len(evaluations),
            "results_stored": len(rows),
            "duration_ms": duration_ms,
            "emergency_cap": cap,
        }
    except Exception as exc:
        duration_ms = _duration_ms(started_at)
        el_store.fail_run(
            run_id=run_id,
            error_message=str(exc),
            duration_ms=duration_ms,
        )
        logger.exception("Emerging Leaders precompute %s failed", run_id)
        return {
            "run_id": run_id,
            "status": "failed",
            "error_message": str(exc),
            "duration_ms": duration_ms,
        }
