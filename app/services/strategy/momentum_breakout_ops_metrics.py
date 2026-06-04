"""In-process operational counters for Momentum Breakout rollout."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone

from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus
from trade_planner.alerts.lifecycle_store import MomentumBreakoutAlertStore
from trade_planner.alerts.paper_trade_store import PaperTradePerformanceStore


@dataclass(frozen=True, slots=True)
class MomentumBreakoutStatusCounts:
    pending_entry: int
    entry_triggered: int
    open: int
    target_hit: int
    stop_hit: int
    expired: int
    cancelled: int
    completed: int


@dataclass(frozen=True, slots=True)
class MomentumBreakoutAdminMetricsSnapshot:
    alerts_created_today: int
    active_alerts_count: int
    status_counts: MomentumBreakoutStatusCounts
    notifications_emitted_today: int
    scheduler_last_run_at: datetime | None
    scheduler_last_error: str | None
    paper_trade_rows_count: int
    readiness_ready: bool | None
    readiness_warnings: tuple[str, ...]


class MomentumBreakoutOpsMetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._alerts_created_date: date | None = None
        self._alerts_created_today = 0
        self._notifications_date: date | None = None
        self._notifications_emitted_today = 0
        self._scheduler_last_run_at: datetime | None = None
        self._scheduler_last_error: str | None = None
        self._readiness_ready: bool | None = None
        self._readiness_warnings: tuple[str, ...] = ()

    def record_alert_created(self) -> None:
        with self._lock:
            today = date.today()
            if self._alerts_created_date != today:
                self._alerts_created_date = today
                self._alerts_created_today = 1
            else:
                self._alerts_created_today += 1

    def record_notification_emitted(self) -> None:
        with self._lock:
            today = date.today()
            if self._notifications_date != today:
                self._notifications_date = today
                self._notifications_emitted_today = 1
            else:
                self._notifications_emitted_today += 1

    def record_scheduler_success(self) -> None:
        with self._lock:
            self._scheduler_last_run_at = datetime.now(timezone.utc)
            self._scheduler_last_error = None

    def record_scheduler_failure(self, *, error: str) -> None:
        with self._lock:
            self._scheduler_last_run_at = datetime.now(timezone.utc)
            self._scheduler_last_error = error

    def record_readiness(self, *, ready: bool, warnings: tuple[str, ...]) -> None:
        with self._lock:
            self._readiness_ready = ready
            self._readiness_warnings = warnings

    def alerts_created_today(self) -> int:
        with self._lock:
            if self._alerts_created_date != date.today():
                return 0
            return self._alerts_created_today

    def notifications_emitted_today(self) -> int:
        with self._lock:
            if self._notifications_date != date.today():
                return 0
            return self._notifications_emitted_today

    def scheduler_last_run_at(self) -> datetime | None:
        with self._lock:
            return self._scheduler_last_run_at

    def scheduler_last_error(self) -> str | None:
        with self._lock:
            return self._scheduler_last_error

    def readiness_ready(self) -> bool | None:
        with self._lock:
            return self._readiness_ready

    def readiness_warnings(self) -> tuple[str, ...]:
        with self._lock:
            return self._readiness_warnings

def count_alert_statuses(
    alert_store: MomentumBreakoutAlertStore,
) -> MomentumBreakoutStatusCounts:
    alerts = alert_store.list_all_alerts()
    counts = {status.value: 0 for status in AlertLifecycleStatus}
    for alert in alerts:
        key = alert.status.value
        counts[key] = counts.get(key, 0) + 1
    completed = (
        counts.get(AlertLifecycleStatus.TARGET_HIT.value, 0)
        + counts.get(AlertLifecycleStatus.STOP_HIT.value, 0)
        + counts.get(AlertLifecycleStatus.EXPIRED.value, 0)
    )
    return MomentumBreakoutStatusCounts(
        pending_entry=counts.get(AlertLifecycleStatus.PENDING_ENTRY.value, 0),
        entry_triggered=counts.get(AlertLifecycleStatus.ENTRY_TRIGGERED.value, 0),
        open=counts.get(AlertLifecycleStatus.OPEN.value, 0),
        target_hit=counts.get(AlertLifecycleStatus.TARGET_HIT.value, 0),
        stop_hit=counts.get(AlertLifecycleStatus.STOP_HIT.value, 0),
        expired=counts.get(AlertLifecycleStatus.EXPIRED.value, 0),
        cancelled=counts.get(AlertLifecycleStatus.CANCELLED.value, 0),
        completed=completed,
    )


def count_paper_trade_rows(paper_store: PaperTradePerformanceStore) -> int:
    if hasattr(paper_store, "count_all"):
        return int(paper_store.count_all())  # type: ignore[attr-defined]
    return len(paper_store.list_all())


def build_admin_metrics_snapshot(
    *,
    registry: MomentumBreakoutOpsMetricsRegistry,
    alert_store: MomentumBreakoutAlertStore,
    paper_store: PaperTradePerformanceStore,
) -> MomentumBreakoutAdminMetricsSnapshot:
    status_counts = count_alert_statuses(alert_store)
    active = len(alert_store.list_all_active())
    return MomentumBreakoutAdminMetricsSnapshot(
        alerts_created_today=registry.alerts_created_today(),
        active_alerts_count=active,
        status_counts=status_counts,
        notifications_emitted_today=registry.notifications_emitted_today(),
        scheduler_last_run_at=registry.scheduler_last_run_at(),
        scheduler_last_error=registry.scheduler_last_error(),
        paper_trade_rows_count=count_paper_trade_rows(paper_store),
        readiness_ready=registry.readiness_ready(),
        readiness_warnings=registry.readiness_warnings(),
    )


_default_registry = MomentumBreakoutOpsMetricsRegistry()


def get_momentum_breakout_ops_metrics() -> MomentumBreakoutOpsMetricsRegistry:
    return _default_registry
