from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.momentum_breakout_alert_models import MomentumBreakoutAlertDto
from app.models.strategy_models import _STRATEGY_MODEL_CONFIG

NotificationSeverity = Literal["info", "watch", "warning", "critical"]


class MomentumBreakoutNotificationDto(BaseModel):
    """API contract for an in-app notification feed item."""

    model_config = _STRATEGY_MODEL_CONFIG

    notification_id: str = Field(alias="notificationId")
    event_type: str = Field(alias="eventType")
    title: str
    body: str
    severity: NotificationSeverity
    next_action_message: str = Field(alias="nextActionMessage")
    symbol: str
    alert_id: str | None = Field(default=None, alias="alertId")
    read: bool
    created_at: datetime = Field(alias="createdAt")
    alert: MomentumBreakoutAlertDto


class MomentumBreakoutNotificationListResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    disclaimer: str
    notifications: list[MomentumBreakoutNotificationDto]


class MarkNotificationReadResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    disclaimer: str
    notification: MomentumBreakoutNotificationDto
