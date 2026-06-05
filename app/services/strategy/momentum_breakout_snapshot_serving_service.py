from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from enum import Enum

from app.models.momentum_breakout_alert_models import AlertRiskGateResultDto
from app.models.momentum_breakout_scan_models import (
    DEFAULT_MAX_STOP_DISTANCE_PCT,
    DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR,
    DEFAULT_MIN_HISTORICAL_TRADES,
    MomentumBreakoutScanCandidateDto,
    MomentumBreakoutScanResponse,
)
from app.services.strategy.momentum_breakout_scanner_service import (
    MomentumBreakoutScannerService,
    _ScanCandidate,
    is_tradable_candidate,
)
from app.storage.momentum_breakout_scan_store import (
    MomentumBreakoutScanStore,
    open_momentum_breakout_scan_store,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_SNAPSHOT_AGE_HOURS = 36.0


class MomentumBreakoutServingMode(str, Enum):
    PRECOMPUTED = "precomputed"
    LIVE_EMERGENCY = "live_emergency"
    PRECOMPUTED_WITH_LIVE_FALLBACK = "precomputed_with_live_fallback"


class MomentumBreakoutSnapshotUnavailableError(RuntimeError):
    pass


def momentum_breakout_serving_mode() -> MomentumBreakoutServingMode:
    raw = os.environ.get(
        "MB_SCAN_SERVING_MODE",
        MomentumBreakoutServingMode.PRECOMPUTED.value,
    )
    try:
        return MomentumBreakoutServingMode(raw.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in MomentumBreakoutServingMode)
        raise ValueError(
            f"Invalid MB_SCAN_SERVING_MODE={raw!r}; expected one of: {allowed}"
        ) from exc


def max_snapshot_age_hours() -> float:
    raw = os.environ.get("MB_SCAN_MAX_SNAPSHOT_AGE_HOURS")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_SNAPSHOT_AGE_HOURS
    value = float(raw)
    if value <= 0:
        raise ValueError("MB_SCAN_MAX_SNAPSHOT_AGE_HOURS must be positive")
    return value


def _parse_generated_at(value: str) -> datetime:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        return instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def _candidate_from_row(row: dict) -> _ScanCandidate:
    return _ScanCandidate(
        symbol=row["symbol"],
        entry_price=float(row["entry_price"]),
        stop_price=float(row["stop_price"]),
        target_price=float(row["target_price"]),
        risk_reward=float(row["risk_reward"]),
        historical_win_rate=row["historical_win_rate"],
        historical_profit_factor=row["historical_profit_factor"],
        historical_total_trades=row["historical_total_trades"],
        setup_score=float(row["setup_score"]),
        stop_distance_pct=float(row["stop_distance_pct"]),
        volume_ratio=row["volume_ratio"],
        rs_percentile=row["rs_percentile"],
        market_regime=row["market_regime"],
        risk_gate=AlertRiskGateResultDto.model_validate(row["risk_gate"]),
    )


class MomentumBreakoutSnapshotServingService:
    def __init__(
        self,
        *,
        store: MomentumBreakoutScanStore | None = None,
        scanner: MomentumBreakoutScannerService | None = None,
    ) -> None:
        self._store = store or open_momentum_breakout_scan_store()
        self._scanner = scanner or MomentumBreakoutScannerService()

    def scan(
        self,
        *,
        limit: int = 50,
        tradable_only: bool = False,
        min_historical_profit_factor: float = DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR,
        min_historical_trades: int = DEFAULT_MIN_HISTORICAL_TRADES,
        max_stop_distance_pct: float = DEFAULT_MAX_STOP_DISTANCE_PCT,
    ) -> MomentumBreakoutScanResponse:
        mode = momentum_breakout_serving_mode()
        if mode == MomentumBreakoutServingMode.LIVE_EMERGENCY:
            logger.warning("Momentum Breakout scan using live_emergency mode")
            return self._scanner.scan(
                limit=limit,
                tradable_only=tradable_only,
                min_historical_profit_factor=min_historical_profit_factor,
                min_historical_trades=min_historical_trades,
                max_stop_distance_pct=max_stop_distance_pct,
            )

        run = self._store.latest_completed_run()
        if run is None:
            if mode == MomentumBreakoutServingMode.PRECOMPUTED_WITH_LIVE_FALLBACK:
                logger.warning(
                    "Momentum Breakout snapshot missing; falling back to live scan"
                )
                return self._scanner.scan(
                    limit=limit,
                    tradable_only=tradable_only,
                    min_historical_profit_factor=min_historical_profit_factor,
                    min_historical_trades=min_historical_trades,
                    max_stop_distance_pct=max_stop_distance_pct,
                )
            raise MomentumBreakoutSnapshotUnavailableError(
                "Momentum Breakout precomputed snapshot is not available"
            )

        age_hours = (
            datetime.now(timezone.utc) - _parse_generated_at(run.generated_at)
        ).total_seconds() / 3600
        if age_hours > max_snapshot_age_hours():
            logger.warning(
                "Momentum Breakout serving stale snapshot run_id=%s age_hours=%.2f",
                run.run_id,
                age_hours,
            )

        rows = self._store.list_results(run.run_id)
        candidates = [_candidate_from_row(row) for row in rows]
        tradable_candidates = [
            candidate
            for candidate in candidates
            if is_tradable_candidate(
                candidate,
                min_historical_profit_factor=min_historical_profit_factor,
                min_historical_trades=min_historical_trades,
                max_stop_distance_pct=max_stop_distance_pct,
            )
        ]

        pool = tradable_candidates if tradable_only else candidates
        capped = pool[: max(1, limit)]
        return MomentumBreakoutScanResponse(
            scanTime=run.generated_at,
            totalSymbolsScanned=run.symbols_scanned,
            validSetupsFound=len(candidates),
            tradableCandidatesFound=len(tradable_candidates),
            blockedCandidatesCount=len(candidates) - len(tradable_candidates),
            candidatesFound=len(capped),
            candidates=[
                MomentumBreakoutScannerService._to_dto(candidate)  # noqa: SLF001
                for candidate in capped
            ],
        )
