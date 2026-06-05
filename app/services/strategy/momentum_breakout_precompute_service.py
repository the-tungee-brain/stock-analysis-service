from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.models.momentum_breakout_scan_models import (
    DEFAULT_MAX_STOP_DISTANCE_PCT,
    DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR,
    DEFAULT_MIN_HISTORICAL_TRADES,
)
from app.services.strategy.momentum_breakout_scan_universe import (
    build_scan_universe_symbols,
)
from app.services.strategy.momentum_breakout_scanner_service import (
    MomentumBreakoutScannerService,
    _candidate_sort_key,
    is_tradable_candidate,
)
from app.storage.momentum_breakout_scan_store import (
    MomentumBreakoutScanStore,
    open_momentum_breakout_scan_store,
)
from ranking_pipeline.config import RankingPipelineConfig, default_config
from ranking_pipeline.storage.sqlite import RankingStore, open_store

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(started_at: float) -> int:
    return max(0, int(round((time.perf_counter() - started_at) * 1000)))


def _emergency_cap() -> int | None:
    raw = os.environ.get("MB_PRECOMPUTE_MAX_UNIVERSE")
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("MB_PRECOMPUTE_MAX_UNIVERSE must be positive")
    return value


def _ranking_as_of_date(
    ranking_store: RankingStore,
    ranking_run_id: str | None,
) -> str | None:
    if ranking_run_id is None:
        return None
    meta = ranking_store.get_run_meta(ranking_run_id)
    if not meta:
        return None
    return meta.get("as_of_date")


def _result_rows(candidates: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        dto = MomentumBreakoutScannerService._to_dto(candidate)  # noqa: SLF001
        payload = dto.model_dump(mode="json", by_alias=False)
        rows.append(
            {
                "rank": rank,
                "symbol": payload["symbol"],
                "entry_price": payload["entry_price"],
                "stop_price": payload["stop_price"],
                "target_price": payload["target_price"],
                "risk_reward": payload["risk_reward"],
                "historical_win_rate": payload["historical_win_rate"],
                "historical_profit_factor": payload["historical_profit_factor"],
                "historical_total_trades": payload["historical_total_trades"],
                "setup_score": payload["setup_score"],
                "stop_distance_pct": payload["stop_distance_pct"],
                "volume_ratio": payload["volume_ratio"],
                "rs_percentile": payload["rs_percentile"],
                "market_regime": payload["market_regime"],
                "risk_gate": payload["risk_gate"],
            }
        )
    return rows


def precompute_momentum_breakout_scan_snapshot(
    *,
    cfg: RankingPipelineConfig | None = None,
    ranking_store: RankingStore | None = None,
    snapshot_store: MomentumBreakoutScanStore | None = None,
    scanner: MomentumBreakoutScannerService | None = None,
    max_universe: int | None = None,
) -> dict[str, Any]:
    """
    Run the Momentum Breakout scanner off the request path and persist valid setups.

    `max_universe` is an emergency/resource-control cap. When omitted, the
    environment cap is used only if explicitly configured.
    """
    config = cfg or default_config()
    rank_store = ranking_store or open_store(config)
    scan_store = snapshot_store or open_momentum_breakout_scan_store(config)
    scan_service = scanner or MomentumBreakoutScannerService()
    run_id = str(uuid4())
    generated_at = _utc_timestamp()
    started_at = time.perf_counter()
    scan_store.start_run(run_id=run_id, generated_at=generated_at)

    try:
        cap = max_universe if max_universe is not None else _emergency_cap()
        universe = build_scan_universe_symbols(store=rank_store)
        full_symbols = universe.symbols
        symbol_list = full_symbols if cap is None else full_symbols[:cap]
        valid_candidates = scan_service._collect_candidates(symbol_list)  # noqa: SLF001
        valid_candidates.sort(key=_candidate_sort_key)

        tradable_candidates = [
            candidate
            for candidate in valid_candidates
            if is_tradable_candidate(
                candidate,
                min_historical_profit_factor=DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR,
                min_historical_trades=DEFAULT_MIN_HISTORICAL_TRADES,
                max_stop_distance_pct=DEFAULT_MAX_STOP_DISTANCE_PCT,
            )
        ]
        blocked_count = len(valid_candidates) - len(tradable_candidates)
        rows = _result_rows(valid_candidates)
        duration_ms = _duration_ms(started_at)
        as_of_date = _ranking_as_of_date(rank_store, universe.ranking_run_id)
        excluded_by_cap = max(0, len(full_symbols) - len(symbol_list))

        scan_store.complete_run(
            run_id=run_id,
            as_of_date=as_of_date,
            generated_at=generated_at,
            ranking_run_id=universe.ranking_run_id,
            ranking_snapshot_id=universe.ranking_snapshot_id,
            universe_source=universe.universe_source,
            selection_method=universe.selection_method,
            total_ranked_symbols=universe.total_ranked_symbols,
            total_eligible_symbols=len(full_symbols),
            symbols_scanned=len(symbol_list),
            excluded_by_cap=excluded_by_cap,
            valid_setups_found=len(valid_candidates),
            tradable_candidates_found=len(tradable_candidates),
            blocked_candidates_count=blocked_count,
            duration_ms=duration_ms,
            results=rows,
        )
        logger.info(
            "Momentum Breakout precompute %s completed: scanned=%d valid=%d",
            run_id,
            len(symbol_list),
            len(valid_candidates),
        )
        return {
            "run_id": run_id,
            "status": "completed",
            "as_of_date": as_of_date,
            "generated_at": generated_at,
            "ranking_run_id": universe.ranking_run_id,
            "ranking_snapshot_id": universe.ranking_snapshot_id,
            "universe_source": universe.universe_source,
            "selection_method": universe.selection_method,
            "total_ranked_symbols": universe.total_ranked_symbols,
            "total_eligible_symbols": len(full_symbols),
            "symbols_scanned": len(symbol_list),
            "excluded_by_cap": excluded_by_cap,
            "valid_setups_found": len(valid_candidates),
            "tradable_candidates_found": len(tradable_candidates),
            "blocked_candidates_count": blocked_count,
            "results_stored": len(rows),
            "duration_ms": duration_ms,
            "emergency_cap": cap,
        }
    except Exception as exc:
        duration_ms = _duration_ms(started_at)
        scan_store.fail_run(
            run_id=run_id,
            error_message=str(exc),
            duration_ms=duration_ms,
        )
        logger.exception("Momentum Breakout precompute %s failed", run_id)
        return {
            "run_id": run_id,
            "status": "failed",
            "error_message": str(exc),
            "duration_ms": duration_ms,
        }
