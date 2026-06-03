from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from app.builders.emerging_leaders_engine import (
    STAGE_LABELS,
    EmergingLeaderEvaluation,
    evaluate_emerging_leader,
    passes_emerging_leader_list,
    ranking_sort_key,
)
from app.models.emerging_leaders_models import EmergingLeaderItem, EmergingLeadersResponse
from data.store import load_raw, raw_exists
from ranking_pipeline.config import default_config
from ranking_pipeline.storage.sqlite import open_store

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    except Exception as exc:
        logger.debug("Emerging leaders skip %s: %s", sym, exc)
        return None


def build_emerging_leaders(*, limit: int = 20) -> EmergingLeadersResponse:
    cfg = default_config()
    store = open_store(cfg)
    universe = store.load_universe_symbols()
    if not universe:
        raise LookupError("No active ranking universe")

    exclude_n = _top_mover_exclude_count()
    top_mover_symbols: set[str] = set()
    run_id = store.latest_run_id()
    as_of_date: str | None = None
    if run_id:
        meta = store.get_run_meta(run_id)
        if meta:
            as_of_date = meta.get("as_of_date")
        for row in store.get_ranking_results(run_id, limit=exclude_n):
            top_mover_symbols.add(str(row["symbol"]).upper())

    with_data = [s for s in universe if raw_exists(s.strip().upper())]
    candidates = [
        s
        for s in with_data
        if s.upper() not in top_mover_symbols
    ][: _max_universe()]

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
    trimmed = evaluations[: max(1, min(limit, 50))]

    items = [
        EmergingLeaderItem(
            rank=idx,
            symbol=ev.symbol,
            setup_quality_score=ev.setup_quality_score,
            setup_stage=ev.setup_stage,
            setup_stage_label=STAGE_LABELS[ev.setup_stage],
            why_it_ranks=ev.why_it_ranks,
            positive_factors=ev.positive_factors,
            missing_factors=ev.missing_factors,
            next_confirmation=ev.next_confirmation,
        )
        for idx, ev in enumerate(trimmed, start=1)
    ]

    if not items and with_data:
        logger.warning(
            "Emerging leaders: %d symbols with OHLCV but 0 evaluations "
            "(candidates=%d, excluded_top=%d)",
            len(with_data),
            len(candidates),
            len(top_mover_symbols),
        )

    return EmergingLeadersResponse(
        as_of_date=as_of_date,
        timestamp=_utc_timestamp(),
        universe_scanned=len(candidates),
        symbols_with_data=len(with_data),
        evaluations_computed=len(evaluations),
        excluded_top_movers=len(top_mover_symbols),
        items=items,
    )
