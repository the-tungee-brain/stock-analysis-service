from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.strategy_models import _STRATEGY_MODEL_CONFIG


class MomentumBreakoutFeatureFlagsDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    alerts_enabled: bool = Field(alias="alertsEnabled")
    alert_creation_enabled: bool = Field(alias="alertCreationEnabled")
    alert_notifications_enabled: bool = Field(alias="alertNotificationsEnabled")
    paper_analytics_enabled: bool = Field(alias="paperAnalyticsEnabled")


class MomentumBreakoutFeatureStatusResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    disclaimer: str = (
        "Feature availability for educational trade-plan monitoring only. "
        "Not brokerage execution or auto-trading."
    )
    flags: MomentumBreakoutFeatureFlagsDto


class MomentumBreakoutStatusCountsDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    pending_entry: int = Field(alias="pendingEntry")
    entry_triggered: int = Field(alias="entryTriggered")
    open: int
    target_hit: int = Field(alias="targetHit")
    stop_hit: int = Field(alias="stopHit")
    expired: int
    cancelled: int
    completed: int


class MomentumBreakoutAdminMetricsResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    disclaimer: str = "Operational metrics only — not performance marketing."
    alerts_created_today: int = Field(alias="alertsCreatedToday")
    active_alerts_count: int = Field(alias="activeAlertsCount")
    status_counts: MomentumBreakoutStatusCountsDto = Field(alias="statusCounts")
    notifications_emitted_today: int = Field(alias="notificationsEmittedToday")
    scheduler_last_run_at: datetime | None = Field(
        default=None, alias="schedulerLastRunAt"
    )
    scheduler_last_error: str | None = Field(
        default=None, alias="schedulerLastError"
    )
    paper_trade_rows_count: int = Field(alias="paperTradeRowsCount")
    readiness_ready: bool | None = Field(default=None, alias="readinessReady")
    readiness_warnings: list[str] = Field(
        default_factory=list, alias="readinessWarnings"
    )
