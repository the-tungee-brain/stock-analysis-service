"""Shared Emerging Leaders evaluation collection (ranking logic unchanged)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.builders.emerging_leaders_engine import (
    EmergingLeaderEvaluation,
    evaluate_emerging_leader,
    passes_emerging_leader_list,
    ranking_sort_key,
)
from data.store import load_raw, raw_exists
from ranking_pipeline.config import default_config
from ranking_pipeline.storage.sqlite import open_store


def _max_universe() -> int:
    return int(os.environ.get("EMERGING_LEADERS_MAX_UNIVERSE", "500"))


def _top_mover_exclude_count() -> int:
    return int(os.environ.get("EMERGING_LEADERS_EXCLUDE_TOP_MOVERS", "12"))


def _worker_count() -> int:
    return int(os.environ.get("EMERGING_LEADERS_WORKERS", "8"))


def _score_symbol(symbol: str) -> EmergingLeaderEvaluation | None:
    sym = symbol.strip().upper()
    if not raw_exists(sym):
        return None
    try:
        raw = load_raw(sym)
        return evaluate_emerging_leader(sym, raw)
    except Exception:
        return None


def collect_qualifying_emerging_leader_evaluations(
    *,
    max_universe: int | None = None,
) -> tuple[list[EmergingLeaderEvaluation], str | None, int, int, int]:
    """
    Score universe candidates and return all names passing list filters,
    sorted by existing ranking_sort_key (same order as production list).
    """
    cfg = default_config()
    store = open_store(cfg)
    universe = store.load_universe_symbols()
    if not universe:
        raise LookupError("No active ranking universe")

    exclude_n = _top_mover_exclude_count()
    top_mover_symbols: set[str] = set()
    as_of_date: str | None = None
    run_id = store.latest_run_id()
    if run_id:
        meta = store.get_run_meta(run_id)
        if meta:
            as_of_date = meta.get("as_of_date")
        for row in store.get_ranking_results(run_id, limit=exclude_n):
            top_mover_symbols.add(str(row["symbol"]).upper())

    with_data = [s for s in universe if raw_exists(s.strip().upper())]
    cap = max_universe if max_universe is not None else _max_universe()
    candidates = [
        s for s in with_data if s.upper() not in top_mover_symbols
    ][:cap]

    evaluations: list[EmergingLeaderEvaluation] = []
    workers = max(1, min(_worker_count(), 16))
    if candidates:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_symbol, sym): sym for sym in candidates}
            for future in as_completed(futures):
                result = future.result()
                if result is not None and passes_emerging_leader_list(result):
                    evaluations.append(result)

    evaluations.sort(key=ranking_sort_key, reverse=True)
    return (
        evaluations,
        as_of_date,
        len(candidates),
        len(with_data),
        len(top_mover_symbols),
    )
