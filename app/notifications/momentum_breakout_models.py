"""Models for Momentum Breakout user notifications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import uuid4


class MomentumBreakoutNotificationEventType(str, Enum):
    ALERT_CREATED = "AlertCreated"
    ENTRY_TRIGGERED = "EntryTriggered"
    TARGET_HIT = "TargetHit"
    STOP_HIT = "StopHit"
    EXPIRED = "Expired"
    BLOCKED_BY_RISK_GATE = "BlockedByRiskGate"
    WARNING_BY_RISK_GATE = "WarningByRiskGate"


@dataclass(frozen=True, slots=True)
class MomentumBreakoutUserNotification:
    notification_id: str
    user_id: str
    event_type: MomentumBreakoutNotificationEventType
    title: str
    body: str
    symbol: str
    alert_id: str | None
    read: bool
    created_at: datetime
    severity: str
    next_action_message: str
    alert_snapshot_json: dict[str, object]

    @staticmethod
    def new_id() -> str:
        return str(uuid4())
