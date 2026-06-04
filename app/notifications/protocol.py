"""Notification delivery abstractions."""

from __future__ import annotations

from typing import Protocol

from app.notifications.momentum_breakout_models import MomentumBreakoutUserNotification


class NotificationChannel(Protocol):
    """Delivers a notification to a single channel (in-app, push, email)."""

    def deliver(self, notification: MomentumBreakoutUserNotification) -> None: ...


class NotificationService(Protocol):
    """Publishes notifications and manages in-app read state."""

    def publish(
        self, notification: MomentumBreakoutUserNotification
    ) -> MomentumBreakoutUserNotification: ...

    def list_notifications(
        self,
        user_id: str,
        *,
        unread_only: bool = False,
        limit: int = 100,
    ) -> tuple[MomentumBreakoutUserNotification, ...]: ...

    def mark_read(
        self, user_id: str, notification_id: str
    ) -> MomentumBreakoutUserNotification: ...
