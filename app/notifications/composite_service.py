"""Composite notification service: in-app store plus channel placeholders."""

from __future__ import annotations

from app.notifications.email_placeholder import EmailNotificationPlaceholder
from app.notifications.in_app_store import InAppNotificationStore
from app.notifications.momentum_breakout_models import MomentumBreakoutUserNotification
from app.notifications.push_placeholder import PushNotificationPlaceholder


class CompositeNotificationService:
    def __init__(
        self,
        in_app_store: InAppNotificationStore | None = None,
        push_channel: PushNotificationPlaceholder | None = None,
        email_channel: EmailNotificationPlaceholder | None = None,
    ) -> None:
        self._in_app = in_app_store or InAppNotificationStore()
        self._push = push_channel or PushNotificationPlaceholder()
        self._email = email_channel or EmailNotificationPlaceholder()

    @property
    def in_app_store(self) -> InAppNotificationStore:
        return self._in_app

    def publish(
        self, notification: MomentumBreakoutUserNotification
    ) -> MomentumBreakoutUserNotification:
        self._in_app.save(notification)
        self._push.deliver(notification)
        self._email.deliver(notification)
        return notification

    def list_notifications(
        self,
        user_id: str,
        *,
        unread_only: bool = False,
        limit: int = 100,
    ) -> tuple[MomentumBreakoutUserNotification, ...]:
        return self._in_app.list_for_user(
            user_id, unread_only=unread_only, limit=limit
        )

    def mark_read(
        self, user_id: str, notification_id: str
    ) -> MomentumBreakoutUserNotification:
        return self._in_app.mark_read(user_id, notification_id)
