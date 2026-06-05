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

BLOCK_REASON_RISK_GATE = "risk_gate"
BLOCK_REASON_HISTORICAL_PROFIT_FACTOR = "historical_profit_factor"
BLOCK_REASON_HISTORICAL_TRADE_COUNT = "historical_trade_count"
BLOCK_REASON_STOP_DISTANCE = "stop_distance"
TOP_BLOCKED_SYMBOLS_LIMIT = 10


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


def _blocked_reasons(candidate: Any) -> list[str]:
    reasons: list[str] = []
    if not candidate.risk_gate.allowed:
        reasons.append(BLOCK_REASON_RISK_GATE)
    profit_factor = candidate.historical_profit_factor
    if (
        profit_factor is None
        or profit_factor < DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR
    ):
        reasons.append(BLOCK_REASON_HISTORICAL_PROFIT_FACTOR)
    total_trades = candidate.historical_total_trades
    if total_trades is None or total_trades < DEFAULT_MIN_HISTORICAL_TRADES:
        reasons.append(BLOCK_REASON_HISTORICAL_TRADE_COUNT)
    if candidate.stop_distance_pct > DEFAULT_MAX_STOP_DISTANCE_PCT:
        reasons.append(BLOCK_REASON_STOP_DISTANCE)
    return reasons


def blocked_candidate_diagnostics(candidates: list[Any]) -> dict[str, Any]:
    counts = {
        BLOCK_REASON_HISTORICAL_PROFIT_FACTOR: 0,
        BLOCK_REASON_HISTORICAL_TRADE_COUNT: 0,
        BLOCK_REASON_STOP_DISTANCE: 0,
        BLOCK_REASON_RISK_GATE: 0,
    }
    top_blocked: list[dict[str, Any]] = []

    for candidate in candidates:
        reasons = _blocked_reasons(candidate)
        if not reasons:
            continue
        for reason in reasons:
            counts[reason] += 1
        if len(top_blocked) < TOP_BLOCKED_SYMBOLS_LIMIT:
            top_blocked.append(
                {
                    "symbol": candidate.symbol,
                    "primary_block_reason": reasons[0],
                    "block_reasons": reasons,
                    "historical_profit_factor": candidate.historical_profit_factor,
                    "historical_total_trades": candidate.historical_total_trades,
                    "stop_distance_pct": candidate.stop_distance_pct,
                    "risk_gate_action": candidate.risk_gate.action,
                    "risk_gate_reasons": list(candidate.risk_gate.reasons),
                }
            )

    return {
        "blocked_by_historical_profit_factor": counts[
            BLOCK_REASON_HISTORICAL_PROFIT_FACTOR
        ],
        "blocked_by_historical_trade_count": counts[
            BLOCK_REASON_HISTORICAL_TRADE_COUNT
        ],
        "blocked_by_stop_distance": counts[BLOCK_REASON_STOP_DISTANCE],
        "blocked_by_risk_gate": counts[BLOCK_REASON_RISK_GATE],
        "top_blocked_symbols": top_blocked,
    }


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
        blocked_diagnostics = blocked_candidate_diagnostics(valid_candidates)
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
            (
                "Momentum Breakout precompute %s completed: scanned=%d valid=%d "
                "tradable=%d blocked=%d blocked_by_pf=%d blocked_by_trades=%d "
                "blocked_by_stop=%d blocked_by_risk_gate=%d"
            ),
            run_id,
            len(symbol_list),
            len(valid_candidates),
            len(tradable_candidates),
            blocked_count,
            blocked_diagnostics["blocked_by_historical_profit_factor"],
            blocked_diagnostics["blocked_by_historical_trade_count"],
            blocked_diagnostics["blocked_by_stop_distance"],
            blocked_diagnostics["blocked_by_risk_gate"],
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
            "blocked_diagnostics": blocked_diagnostics,
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
