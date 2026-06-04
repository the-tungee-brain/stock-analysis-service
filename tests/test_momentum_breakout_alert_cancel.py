"""Cancel active Momentum Breakout alerts."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user, get_current_user_id
from app.main import app
from app.dependencies.service_dependencies import get_momentum_breakout_alert_service
from app.notifications.composite_service import CompositeNotificationService
from app.services.strategy.momentum_breakout_notification_emitter import (
    MomentumBreakoutNotificationEmitter,
)
from app.services.strategy.notifying_alert_lifecycle_service import (
    NotifyingAlertLifecycleService,
)
from app.services.strategy.paper_trade_performance_service import (
    PaperTradePerformanceService,
)
from app.services.strategy.paper_trade_recording_lifecycle_service import (
    PaperTradeRecordingLifecycleService,
)
from trade_planner.alerts.lifecycle_models import (
    AlertLifecycleEventType,
    AlertLifecycleStatus,
)
from trade_planner.alerts.lifecycle_service import (
    ALERT_CANCELLED_BY_USER_MESSAGE,
    AlertLifecycleService,
)
from trade_planner.alerts.lifecycle_store import (
    AlertNotCancellableError,
    InMemoryMomentumBreakoutAlertStore,
)
from trade_planner.alerts.paper_trade_store import InMemoryPaperTradePerformanceStore

USER = "user-cancel-1"


def _build_record(**kwargs):
    created_at = datetime(2025, 1, 10, 21, 0, tzinfo=timezone.utc)
    defaults = dict(
        user_id=USER,
        symbol="NVDA",
        signal_date=date(2025, 1, 10),
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        entry_is_stop=True,
        created_at=created_at,
    )
    defaults.update(kwargs)
    return AlertLifecycleService.build_record(**defaults)


@pytest.fixture
def lifecycle_stack():
    alert_store = InMemoryMomentumBreakoutAlertStore()
    paper_store = InMemoryPaperTradePerformanceStore()
    paper_service = PaperTradePerformanceService(paper_store)
    notifications = CompositeNotificationService()
    emitter = MomentumBreakoutNotificationEmitter(notifications)
    lifecycle = PaperTradeRecordingLifecycleService(
        alert_store,
        emitter,
        paper_service,
    )
    return lifecycle, paper_service


def test_cancel_pending_alert(lifecycle_stack) -> None:
    lifecycle, _paper = lifecycle_stack
    created = lifecycle.create_alert(_build_record())
    cancelled = lifecycle.cancel_alert(USER, created.alert_id)

    assert cancelled.status == AlertLifecycleStatus.CANCELLED
    assert cancelled.exit_at is not None


def test_cancel_open_alert(lifecycle_stack) -> None:
    lifecycle, _paper = lifecycle_stack
    created = lifecycle.create_alert(_build_record())
    ts = datetime(2025, 1, 11, 15, 0, tzinfo=timezone.utc)
    lifecycle.update_with_latest_price(
        USER,
        created.alert_id,
        symbol="NVDA",
        price=101.0,
        timestamp=ts,
    )
    cancelled = lifecycle.cancel_alert(USER, created.alert_id)

    assert cancelled.status == AlertLifecycleStatus.CANCELLED


def test_cannot_cancel_completed_alert(lifecycle_stack) -> None:
    lifecycle, _paper = lifecycle_stack
    created = lifecycle.create_alert(_build_record())
    lifecycle.mark_target_hit(USER, created.alert_id, exit_price=110.0)

    with pytest.raises(AlertNotCancellableError):
        lifecycle.cancel_alert(USER, created.alert_id)


def test_cancel_records_lifecycle_event(lifecycle_stack) -> None:
    lifecycle, _paper = lifecycle_stack
    created = lifecycle.create_alert(_build_record())
    lifecycle.cancel_alert(USER, created.alert_id)

    events = lifecycle.list_lifecycle_events(USER, created.alert_id)
    cancelled_events = [
        e for e in events if e.event_type == AlertLifecycleEventType.CANCELLED
    ]
    assert len(cancelled_events) == 1
    assert cancelled_events[0].message == ALERT_CANCELLED_BY_USER_MESSAGE
    assert cancelled_events[0].from_status == AlertLifecycleStatus.PENDING_ENTRY
    assert cancelled_events[0].to_status == AlertLifecycleStatus.CANCELLED


def test_active_alert_count_decreases(lifecycle_stack) -> None:
    lifecycle, _paper = lifecycle_stack
    created = lifecycle.create_alert(_build_record())
    assert len(lifecycle.list_active_alerts(USER)) == 1

    lifecycle.cancel_alert(USER, created.alert_id)

    assert len(lifecycle.list_active_alerts(USER)) == 0
    history = lifecycle.list_alert_history(USER)
    assert len(history) == 1
    assert history[0].status == AlertLifecycleStatus.CANCELLED


def test_cancel_updates_paper_trade_record(lifecycle_stack) -> None:
    lifecycle, paper = lifecycle_stack
    created = lifecycle.create_alert(_build_record())
    ts = datetime(2025, 1, 11, 15, 0, tzinfo=timezone.utc)
    lifecycle.update_with_latest_price(
        USER,
        created.alert_id,
        symbol="NVDA",
        price=101.0,
        timestamp=ts,
    )
    lifecycle.cancel_alert(USER, created.alert_id)

    row = paper.store.get(USER, created.alert_id)
    assert row is not None
    assert row.status == AlertLifecycleStatus.CANCELLED.value
    assert row.exit_at is not None


def test_cancel_api_returns_updated_dto(lifecycle_stack, monkeypatch) -> None:
    lifecycle, _paper = lifecycle_stack
    monkeypatch.setenv("MB_ALERTS_ENABLED", "true")

    class _FakeUser:
        identity_sub = USER

    async def _user() -> _FakeUser:
        return _FakeUser()

    service = type(
        "Svc",
        (),
        {"lifecycle_service": lifecycle},
    )()

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_current_user_id] = lambda: USER
    app.dependency_overrides[get_momentum_breakout_alert_service] = lambda: service

    created = lifecycle.create_alert(_build_record())
    client = TestClient(app)
    try:
        response = client.post(
            f"/api/v1/strategy/momentum-breakout/alerts/{created.alert_id}/cancel",
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "CANCELLED"
        assert any(
            e["eventType"] == "CANCELLED"
            and e["message"] == ALERT_CANCELLED_BY_USER_MESSAGE
            for e in body["lifecycleEvents"]
        )

        blocked = client.post(
            f"/api/v1/strategy/momentum-breakout/alerts/{created.alert_id}/cancel",
        )
        assert blocked.status_code == 409
    finally:
        app.dependency_overrides.clear()
