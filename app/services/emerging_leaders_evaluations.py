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
from app.services.strategy.momentum_breakout_scan_universe import (
    is_ranking_output_stale,
)
from data.store import load_raw, raw_exists
from ranking_pipeline.config import default_config
from ranking_pipeline.storage.sqlite import RankingStore, UniverseMemberRecord, open_store


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


def _metric(value: float | None) -> float:
    return float(value) if value is not None else -1.0


def _sort_by_liquidity(
    members: list[UniverseMemberRecord],
) -> list[UniverseMemberRecord]:
    return sorted(
        members,
        key=lambda member: (
            -_metric(member.avg_dollar_volume_20d),
            -_metric(member.market_cap),
            member.symbol,
        ),
    )


def select_emerging_leader_candidates(
    store: RankingStore,
    *,
    max_universe: int | None,
    top_mover_symbols: set[str],
) -> tuple[list[str], int]:
    """Return quality-ordered candidates and total passed symbols with local data."""
    snapshot_id = store.active_snapshot_id()
    members = store.load_passed_universe_members(snapshot_id)
    if not members:
        raise LookupError("No active ranking universe")

    with_data = [
        member
        for member in members
        if raw_exists(member.symbol.strip().upper())
    ]
    eligible = [
        member
        for member in with_data
        if member.symbol.strip().upper() not in top_mover_symbols
    ]

    latest_run = store.get_latest_ranking_run()
    total_ranked = (
        store.count_ranking_results(latest_run.run_id)
        if latest_run is not None
        else 0
    )
    ranking_is_fresh = not is_ranking_output_stale(
        latest_run,
        total_ranked=total_ranked,
    )

    if latest_run is not None and ranking_is_fresh:
        eligible_by_symbol = {
            member.symbol.strip().upper(): member for member in eligible
        }
        ranked_symbols: list[str] = []
        seen: set[str] = set()
        for row in store.load_ranking_results_ordered(latest_run.run_id):
            sym = row.symbol.strip().upper()
            if sym in seen or sym not in eligible_by_symbol:
                continue
            seen.add(sym)
            ranked_symbols.append(sym)

        tail = [
            member
            for member in eligible
            if member.symbol.strip().upper() not in seen
        ]
        ranked_symbols.extend(
            member.symbol.strip().upper() for member in _sort_by_liquidity(tail)
        )
        if max_universe is None:
            return ranked_symbols, len(with_data)
        return ranked_symbols[:max_universe], len(with_data)

    liquidity_ordered = [
        member.symbol.strip().upper() for member in _sort_by_liquidity(eligible)
    ]
    if max_universe is None:
        return liquidity_ordered, len(with_data)
    return liquidity_ordered[:max_universe], len(with_data)


def score_emerging_leader_candidates(
    candidates: list[str],
) -> list[EmergingLeaderEvaluation]:
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
    return evaluations


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
    snapshot_id = store.active_snapshot_id()
    if not snapshot_id:
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

    cap = max_universe if max_universe is not None else _max_universe()
    candidates, symbols_with_data = select_emerging_leader_candidates(
        store,
        max_universe=cap,
        top_mover_symbols=top_mover_symbols,
    )

    evaluations = score_emerging_leader_candidates(candidates)
    return (
        evaluations,
        as_of_date,
        len(candidates),
        symbols_with_data,
        len(top_mover_symbols),
    )
