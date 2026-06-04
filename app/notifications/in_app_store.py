"""In-memory in-app notification store."""

from __future__ import annotations

from dataclasses import replace

from app.notifications.momentum_breakout_models import MomentumBreakoutUserNotification


class InAppNotificationStore:
    def __init__(self) -> None:
        self._by_user: dict[str, list[MomentumBreakoutUserNotification]] = {}

    def save(self, notification: MomentumBreakoutUserNotification) -> None:
        rows = self._by_user.setdefault(notification.user_id, [])
        rows.insert(0, notification)

    def list_for_user(
        self,
        user_id: str,
        *,
        unread_only: bool = False,
        limit: int = 100,
    ) -> tuple[MomentumBreakoutUserNotification, ...]:
        rows = self._by_user.get(user_id, [])
        if unread_only:
            rows = [row for row in rows if not row.read]
        return tuple(rows[:limit])

    def mark_read(
        self, user_id: str, notification_id: str
    ) -> MomentumBreakoutUserNotification:
        rows = self._by_user.get(user_id, [])
        for index, row in enumerate(rows):
            if row.notification_id == notification_id:
                updated = replace(row, read=True)
                rows[index] = updated
                return updated
        raise KeyError(f"Notification {notification_id} not found.")
