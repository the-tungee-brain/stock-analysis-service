"""Alert lifecycle service that emits user notifications on state changes."""

from __future__ import annotations

from datetime import datetime

from trade_planner.alerts.lifecycle_models import (
    AlertLifecycleStatus,
    MomentumBreakoutAlertRecord,
)
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import MomentumBreakoutAlertStore

from app.services.strategy.momentum_breakout_notification_emitter import (
    MomentumBreakoutNotificationEmitter,
)


class NotifyingAlertLifecycleService(AlertLifecycleService):
    def __init__(
        self,
        store: MomentumBreakoutAlertStore,
        emitter: MomentumBreakoutNotificationEmitter,
    ) -> None:
        super().__init__(store)
        self._emitter = emitter

    def create_alert(
        self, record: MomentumBreakoutAlertRecord
    ) -> MomentumBreakoutAlertRecord:
        created = super().create_alert(record)
        self._emitter.on_alert_created(created)
        return created

    def update_with_latest_price(
        self,
        user_id: str,
        alert_id: str,
        *,
        symbol: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        prior = self.get_alert(user_id, alert_id)
        prior_status = prior.status if prior else None
        updated = super().update_with_latest_price(
            user_id,
            alert_id,
            symbol=symbol,
            price=price,
            timestamp=timestamp,
        )
        if prior_status is not None and prior_status != updated.status:
            self._emitter.on_lifecycle_transition(prior_status, updated)
        return updated
