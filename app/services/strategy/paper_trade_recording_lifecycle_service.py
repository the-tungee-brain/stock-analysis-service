"""Lifecycle service that persists paper-trading performance on status changes."""

from __future__ import annotations

from datetime import datetime

from trade_planner.alerts.lifecycle_models import (
    AlertLifecycleStatus,
    MomentumBreakoutAlertRecord,
)
from app.services.strategy.notifying_alert_lifecycle_service import (
    NotifyingAlertLifecycleService,
)
from app.services.strategy.paper_trade_performance_service import (
    PaperTradePerformanceService,
)


class PaperTradeRecordingLifecycleService(NotifyingAlertLifecycleService):
    def __init__(
        self,
        store,
        emitter,
        paper_trade_service: PaperTradePerformanceService,
    ) -> None:
        super().__init__(store, emitter)
        self._paper_trade = paper_trade_service

    def update_with_latest_price(
        self,
        user_id: str,
        alert_id: str,
        *,
        symbol: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        updated = super().update_with_latest_price(
            user_id,
            alert_id,
            symbol=symbol,
            price=price,
            timestamp=timestamp,
        )
        self._maybe_sync(updated)
        return updated

    def mark_entry_triggered(
        self,
        user_id: str,
        alert_id: str,
        *,
        price: float | None = None,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        updated = super().mark_entry_triggered(
            user_id,
            alert_id,
            price=price,
            recorded_at=recorded_at,
        )
        self._maybe_sync(updated)
        return updated

    def mark_target_hit(
        self,
        user_id: str,
        alert_id: str,
        *,
        exit_price: float,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        updated = super().mark_target_hit(
            user_id,
            alert_id,
            exit_price=exit_price,
            recorded_at=recorded_at,
        )
        self._maybe_sync(updated)
        return updated

    def mark_stop_hit(
        self,
        user_id: str,
        alert_id: str,
        *,
        exit_price: float,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        updated = super().mark_stop_hit(
            user_id,
            alert_id,
            exit_price=exit_price,
            recorded_at=recorded_at,
        )
        self._maybe_sync(updated)
        return updated

    def mark_expired(
        self,
        user_id: str,
        alert_id: str,
        *,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        updated = super().mark_expired(
            user_id,
            alert_id,
            recorded_at=recorded_at,
        )
        self._maybe_sync(updated)
        return updated

    def cancel_alert(
        self,
        user_id: str,
        alert_id: str,
        *,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        updated = super().cancel_alert(
            user_id,
            alert_id,
            recorded_at=recorded_at,
        )
        self._paper_trade.sync_from_alert(updated)
        return updated

    def _maybe_sync(self, record: MomentumBreakoutAlertRecord) -> None:
        if record.status in {
            AlertLifecycleStatus.ENTRY_TRIGGERED,
            AlertLifecycleStatus.OPEN,
            AlertLifecycleStatus.TARGET_HIT,
            AlertLifecycleStatus.STOP_HIT,
            AlertLifecycleStatus.EXPIRED,
            AlertLifecycleStatus.CANCELLED,
        }:
            self._paper_trade.sync_from_alert(record)
