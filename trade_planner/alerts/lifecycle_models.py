"""Lifecycle tracking for Momentum Breakout educational alerts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from enum import Enum
from uuid import uuid4


class AlertLifecycleStatus(str, Enum):
    PENDING_ENTRY = "PENDING_ENTRY"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    OPEN = "OPEN"
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


ACTIVE_STATUSES: frozenset[AlertLifecycleStatus] = frozenset(
    {
        AlertLifecycleStatus.PENDING_ENTRY,
        AlertLifecycleStatus.ENTRY_TRIGGERED,
        AlertLifecycleStatus.OPEN,
    }
)

TERMINAL_STATUSES: frozenset[AlertLifecycleStatus] = frozenset(
    {
        AlertLifecycleStatus.TARGET_HIT,
        AlertLifecycleStatus.STOP_HIT,
        AlertLifecycleStatus.EXPIRED,
        AlertLifecycleStatus.CANCELLED,
    }
)


class AlertLifecycleEventType(str, Enum):
    CREATED = "CREATED"
    STATUS_CHANGED = "STATUS_CHANGED"
    PRICE_UPDATE = "PRICE_UPDATE"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class AlertLifecycleEvent:
    event_id: str
    alert_id: str
    event_type: AlertLifecycleEventType
    from_status: AlertLifecycleStatus | None
    to_status: AlertLifecycleStatus
    price: float | None
    recorded_at: datetime
    message: str


@dataclass(frozen=True, slots=True)
class MomentumBreakoutAlertRecord:
    alert_id: str
    user_id: str
    symbol: str
    setup_name: str
    created_at: datetime
    signal_date: date
    entry_price: float
    stop_price: float
    target_price: float
    entry_is_stop: bool
    status: AlertLifecycleStatus
    expires_at: datetime
    triggered_at: datetime | None = None
    exit_at: datetime | None = None
    exit_price: float | None = None
    outcome_return_pct: float | None = None
    risk_gate_action: str = ""
    risk_gate_reasons: tuple[str, ...] = ()
    historical_win_rate: float | None = None
    historical_profit_factor: float | None = None
    historical_total_trades: int | None = None
    market_regime: str | None = None
    volume_ratio: float | None = None
    rs_percentile: float | None = None

    def with_status(
        self,
        status: AlertLifecycleStatus,
        *,
        triggered_at: datetime | None = None,
        exit_at: datetime | None = None,
        exit_price: float | None = None,
        outcome_return_pct: float | None = None,
    ) -> MomentumBreakoutAlertRecord:
        updates: dict = {"status": status}
        if triggered_at is not None:
            updates["triggered_at"] = triggered_at
        if exit_at is not None:
            updates["exit_at"] = exit_at
        if exit_price is not None:
            updates["exit_price"] = exit_price
        if outcome_return_pct is not None:
            updates["outcome_return_pct"] = outcome_return_pct
        return replace(self, **updates)


def new_alert_id() -> str:
    return str(uuid4())


class StaleMomentumSignalError(ValueError):
    """Raised when the setup bar is too old to persist a new alert."""

    def __init__(self, message: str | None = None) -> None:
        from trade_planner.alerts.market_calendar import STALE_SIGNAL_MESSAGE

        super().__init__(message or STALE_SIGNAL_MESSAGE)


def long_return_pct(entry: float, exit_price: float) -> float:
    if entry <= 0:
        return 0.0
    return (exit_price - entry) / entry
