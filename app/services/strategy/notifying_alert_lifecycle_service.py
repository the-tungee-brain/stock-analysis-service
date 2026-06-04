"""Alert lifecycle service that emits user notifications on state changes."""

from __future__ import annotations

from datetime import datetime

from trade_planner.alerts.lifecycle_models import (
    AlertLifecycleStatus,
    MomentumBreakoutAlertRecord,
)
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import MomentumBreakoutAlertStore

from app.core.momentum_breakout_monitoring import log_mb_event
from app.services.strategy.momentum_breakout_notification_emitter import (
    MomentumBreakoutNotificationEmitter,
)
from app.services.strategy.momentum_breakout_ops_metrics import (
    get_momentum_breakout_ops_metrics,
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
        log_mb_event(
            "alert_created",
            alert_id=created.alert_id,
            user_id=created.user_id,
            symbol=created.symbol,
            setup_name=created.setup_name,
        )
        get_momentum_breakout_ops_metrics().record_alert_created()
        self._emitter.on_alert_created(created)
        return created

    def cancel_alert(
        self,
        user_id: str,
        alert_id: str,
        *,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        prior = self.get_alert(user_id, alert_id)
        prior_status = prior.status if prior else None
        updated = super().cancel_alert(
            user_id,
            alert_id,
            recorded_at=recorded_at,
        )
        if prior_status is not None and prior_status != updated.status:
            log_mb_event(
                "alert_cancelled",
                alert_id=updated.alert_id,
                user_id=updated.user_id,
                symbol=updated.symbol,
                setup_name=updated.setup_name,
            )
        return updated

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
