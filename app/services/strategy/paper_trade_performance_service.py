"""Sync paper-trading performance rows from alert lifecycle state."""

from __future__ import annotations

from datetime import datetime, timezone

from trade_planner.alerts.lifecycle_models import (
    AlertLifecycleStatus,
    MomentumBreakoutAlertRecord,
)
from trade_planner.alerts.paper_trade_models import PaperTradePerformanceRecord
from trade_planner.alerts.paper_trade_store import (
    InMemoryPaperTradePerformanceStore,
    PaperTradePerformanceStore,
)


def compute_holding_days(
    entry_triggered_at: datetime | None,
    exit_at: datetime | None,
    *,
    as_of: datetime | None = None,
) -> int | None:
    if entry_triggered_at is None:
        return None
    end = exit_at or as_of or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = entry_triggered_at
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return max(0, (end.date() - start.date()).days)


class PaperTradePerformanceService:
    def __init__(
        self,
        store: PaperTradePerformanceStore | None = None,
    ) -> None:
        self._store = store or InMemoryPaperTradePerformanceStore()

    @property
    def store(self) -> PaperTradePerformanceStore:
        return self._store

    def sync_from_alert(self, alert: MomentumBreakoutAlertRecord) -> None:
        from trade_planner.alerts.paper_trade_models import PAPER_TRADE_TRACKED_STATUSES

        if alert.status not in PAPER_TRADE_TRACKED_STATUSES:
            return
        self._upsert_from_alert(alert)

    def backfill_from_alert(self, alert: MomentumBreakoutAlertRecord) -> None:
        """Create or refresh a paper row from any alert lifecycle state."""
        self._upsert_from_alert(alert)

    def _upsert_from_alert(self, alert: MomentumBreakoutAlertRecord) -> None:
        existing = self._store.get(alert.user_id, alert.alert_id)
        created_at = existing.created_at if existing else alert.created_at
        regime = alert.market_regime
        if regime is not None and hasattr(regime, "value"):
            regime = regime.value

        entry_triggered_at = alert.triggered_at
        exit_at = alert.exit_at
        exit_price = alert.exit_price
        outcome_return_pct = alert.outcome_return_pct

        if alert.status in {
            AlertLifecycleStatus.PENDING_ENTRY,
            AlertLifecycleStatus.EXPIRED,
            AlertLifecycleStatus.CANCELLED,
        }:
            if alert.status == AlertLifecycleStatus.PENDING_ENTRY:
                entry_triggered_at = None
                exit_at = None
                exit_price = None
                outcome_return_pct = None
            elif alert.status == AlertLifecycleStatus.EXPIRED and entry_triggered_at is None:
                exit_at = exit_at or alert.expires_at
                exit_price = None
                outcome_return_pct = None
            elif alert.status == AlertLifecycleStatus.CANCELLED:
                exit_at = exit_at or alert.exit_at
                exit_price = None
                outcome_return_pct = None

        record = PaperTradePerformanceRecord(
            alert_id=alert.alert_id,
            user_id=alert.user_id,
            symbol=alert.symbol.upper(),
            setup_name=alert.setup_name,
            signal_date=alert.signal_date,
            entry_triggered_at=entry_triggered_at,
            entry_price=alert.entry_price,
            stop_price=alert.stop_price,
            target_price=alert.target_price,
            exit_at=exit_at,
            exit_price=exit_price,
            status=alert.status.value,
            outcome_return_pct=outcome_return_pct,
            holding_days=compute_holding_days(entry_triggered_at, exit_at),
            risk_gate_action=alert.risk_gate_action,
            market_regime=str(regime) if regime else None,
            volume_ratio=alert.volume_ratio,
            rs_percentile=alert.rs_percentile,
            created_at=created_at,
        )
        self._store.save(alert.user_id, record)

    def list_records(
        self, user_id: str, *, limit: int = 500
    ) -> tuple[PaperTradePerformanceRecord, ...]:
        return self._store.list_for_user(user_id, limit=limit)
