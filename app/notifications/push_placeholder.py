"""Placeholder push notification channel (no external provider)."""

from __future__ import annotations

import logging

from app.notifications.momentum_breakout_models import MomentumBreakoutUserNotification

logger = logging.getLogger(__name__)


class PushNotificationPlaceholder:
    def deliver(self, notification: MomentumBreakoutUserNotification) -> None:
        logger.debug(
            "Push placeholder: user=%s event=%s title=%s",
            notification.user_id,
            notification.event_type.value,
            notification.title,
        )
