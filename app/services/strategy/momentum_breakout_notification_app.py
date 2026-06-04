"""Map stored notifications to API response DTOs."""

from __future__ import annotations

from app.models.momentum_breakout_alert_models import MomentumBreakoutAlertDto
from app.models.momentum_breakout_notification_models import (
    MomentumBreakoutNotificationDto,
)
from app.notifications.momentum_breakout_models import MomentumBreakoutUserNotification

NOTIFICATION_DISCLAIMER = (
    "Educational trade plan notifications only. Not investment advice. "
    "No orders are placed."
)


def notification_to_dto(
    row: MomentumBreakoutUserNotification,
) -> MomentumBreakoutNotificationDto:
    alert = MomentumBreakoutAlertDto.model_validate(row.alert_snapshot_json)
    return MomentumBreakoutNotificationDto(
        notificationId=row.notification_id,
        eventType=row.event_type.value,
        title=row.title,
        body=row.body,
        severity=row.severity,  # type: ignore[arg-type]
        nextActionMessage=row.next_action_message,
        symbol=row.symbol,
        alertId=row.alert_id,
        read=row.read,
        createdAt=row.created_at,
        alert=alert,
    )
