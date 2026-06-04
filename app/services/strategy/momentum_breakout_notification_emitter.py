"""Emit API notification records for Momentum Breakout lifecycle and risk events."""

from __future__ import annotations

from datetime import datetime, timezone

from trade_planner.alerts.lifecycle_models import (
    AlertLifecycleStatus,
    MomentumBreakoutAlertRecord,
)
from trade_planner.alerts.risk_models import AlertGateAction

from app.notifications.composite_service import CompositeNotificationService
from app.notifications.momentum_breakout_models import (
    MomentumBreakoutNotificationEventType,
    MomentumBreakoutUserNotification,
)
from app.services.strategy.momentum_breakout_alert_dto import (
    alert_dto_to_storage_dict,
    record_to_alert_dto,
)
from app.services.strategy.momentum_breakout_notification_payload import (
    payload_alert_created,
    payload_blocked_by_risk_gate,
    payload_entry_triggered,
    payload_expired,
    payload_stop_hit,
    payload_target_hit,
    payload_warning_by_risk_gate,
)


class MomentumBreakoutNotificationEmitter:
    def __init__(
        self,
        notification_service: CompositeNotificationService,
    ) -> None:
        self._notifications = notification_service

    def on_alert_created(self, record: MomentumBreakoutAlertRecord) -> None:
        payload = payload_alert_created(
            record.symbol,
            setup_name=record.setup_name,
            entry_price=record.entry_price,
        )
        self._publish(
            record,
            event_type=MomentumBreakoutNotificationEventType.ALERT_CREATED,
            payload=payload,
        )

    def on_lifecycle_transition(
        self,
        prior_status: AlertLifecycleStatus,
        record: MomentumBreakoutAlertRecord,
    ) -> None:
        new_status = record.status
        if new_status == prior_status:
            return

        if new_status == AlertLifecycleStatus.OPEN and prior_status in {
            AlertLifecycleStatus.PENDING_ENTRY,
            AlertLifecycleStatus.ENTRY_TRIGGERED,
        }:
            payload = payload_entry_triggered(
                record.symbol,
                setup_name=record.setup_name,
                entry_price=record.entry_price,
                stop_price=record.stop_price,
                target_price=record.target_price,
            )
            self._publish(
                record,
                event_type=MomentumBreakoutNotificationEventType.ENTRY_TRIGGERED,
                payload=payload,
            )
            return

        if new_status == AlertLifecycleStatus.TARGET_HIT:
            payload = payload_target_hit(record.symbol, setup_name=record.setup_name)
            self._publish(
                record,
                event_type=MomentumBreakoutNotificationEventType.TARGET_HIT,
                payload=payload,
            )
            return

        if new_status == AlertLifecycleStatus.STOP_HIT:
            payload = payload_stop_hit(record.symbol, setup_name=record.setup_name)
            self._publish(
                record,
                event_type=MomentumBreakoutNotificationEventType.STOP_HIT,
                payload=payload,
            )
            return

        if new_status == AlertLifecycleStatus.EXPIRED:
            payload = payload_expired(record.symbol, setup_name=record.setup_name)
            self._publish(
                record,
                event_type=MomentumBreakoutNotificationEventType.EXPIRED,
                payload=payload,
            )

    def on_risk_gate_blocked(
        self,
        user_id: str,
        *,
        symbol: str,
        setup_name: str,
        reasons: tuple[str, ...],
        entry_price: float,
        stop_price: float,
        target_price: float,
        risk_gate_action: str = AlertGateAction.BLOCK.value,
    ) -> None:
        payload = payload_blocked_by_risk_gate(
            symbol, reasons, setup_name=setup_name
        )
        record = MomentumBreakoutAlertRecord(
            alert_id="",
            user_id=user_id,
            symbol=symbol.upper(),
            setup_name=setup_name,
            created_at=datetime.now(timezone.utc),
            signal_date=datetime.now(timezone.utc).date(),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            entry_is_stop=True,
            status=AlertLifecycleStatus.CANCELLED,
            expires_at=datetime.now(timezone.utc),
            risk_gate_action=risk_gate_action,
            risk_gate_reasons=reasons,
        )
        self._publish(
            record,
            event_type=MomentumBreakoutNotificationEventType.BLOCKED_BY_RISK_GATE,
            payload=payload,
            alert_id=None,
        )

    def on_risk_gate_warning(
        self,
        user_id: str,
        *,
        symbol: str,
        setup_name: str,
        reasons: tuple[str, ...],
        entry_price: float,
        stop_price: float,
        target_price: float,
        risk_gate_action: str,
        alert_id: str | None = None,
    ) -> None:
        payload = payload_warning_by_risk_gate(
            symbol, reasons, setup_name=setup_name
        )
        record = MomentumBreakoutAlertRecord(
            alert_id=alert_id or "",
            user_id=user_id,
            symbol=symbol.upper(),
            setup_name=setup_name,
            created_at=datetime.now(timezone.utc),
            signal_date=datetime.now(timezone.utc).date(),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            entry_is_stop=True,
            status=AlertLifecycleStatus.PENDING_ENTRY,
            expires_at=datetime.now(timezone.utc),
            risk_gate_action=risk_gate_action,
            risk_gate_reasons=reasons,
        )
        self._publish(
            record,
            event_type=MomentumBreakoutNotificationEventType.WARNING_BY_RISK_GATE,
            payload=payload,
            alert_id=alert_id,
        )

    def _publish(
        self,
        record: MomentumBreakoutAlertRecord,
        *,
        event_type: MomentumBreakoutNotificationEventType,
        payload,
        alert_id: str | None = None,
    ) -> None:
        alert_dto = record_to_alert_dto(record)
        notification = MomentumBreakoutUserNotification(
            notification_id=MomentumBreakoutUserNotification.new_id(),
            user_id=record.user_id,
            event_type=event_type,
            title=payload.title,
            body=payload.body,
            symbol=record.symbol,
            alert_id=alert_id if alert_id is not None else (record.alert_id or None),
            read=False,
            created_at=datetime.now(timezone.utc),
            severity=payload.severity,
            next_action_message=alert_dto.next_action_message,
            alert_snapshot_json=alert_dto_to_storage_dict(alert_dto),
        )
        self._notifications.publish(notification)
