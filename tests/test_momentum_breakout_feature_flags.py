"""Momentum Breakout feature flags and rollout guards."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user, get_current_user_id
from app.main import app
from app.notifications.composite_service import CompositeNotificationService
from app.services.strategy.momentum_breakout_notification_emitter import (
    MomentumBreakoutNotificationEmitter,
)
from app.services.strategy.momentum_breakout_ops_metrics import (
    get_momentum_breakout_ops_metrics,
)
from app.dependencies.service_dependencies import get_momentum_breakout_alert_service
from app.services.strategy.momentum_breakout_admin_metrics_service import (
    MomentumBreakoutAdminMetricsService,
)
from app.services.strategy.momentum_breakout_alert_service import (
    MomentumBreakoutAlertService,
)
from app.services.strategy.notifying_alert_lifecycle_service import (
    NotifyingAlertLifecycleService,
)
from app.services.strategy.paper_trade_performance_service import (
    PaperTradePerformanceService,
)
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import InMemoryMomentumBreakoutAlertStore
from trade_planner.alerts.paper_trade_store import InMemoryPaperTradePerformanceStore


def _build_record(**kwargs):
    defaults = dict(
        user_id="user-1",
        symbol="AAPL",
        signal_date=date(2025, 1, 10),
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        entry_is_stop=True,
    )
    defaults.update(kwargs)
    return AlertLifecycleService.build_record(**defaults)


@pytest.fixture
def alert_service():
    store = InMemoryMomentumBreakoutAlertStore()
    notifications = CompositeNotificationService()
    emitter = MomentumBreakoutNotificationEmitter(notifications)
    lifecycle = NotifyingAlertLifecycleService(store, emitter)
    return MomentumBreakoutAlertService(
        lifecycle_service=lifecycle,
        notification_emitter=emitter,
    )


@pytest.fixture(autouse=True)
def _reset_metrics():
    registry = get_momentum_breakout_ops_metrics()
    registry._alerts_created_today = 0  # noqa: SLF001
    registry._alerts_created_date = None  # noqa: SLF001
    registry._notifications_emitted_today = 0  # noqa: SLF001
    registry._notifications_date = None  # noqa: SLF001
    yield


def test_feature_disabled_blocks_alert_api(monkeypatch):
    monkeypatch.setenv("MB_ALERTS_ENABLED", "false")

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    async def _user_id() -> str:
        return "user-1"

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_current_user_id] = _user_id
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/strategy/momentum-breakout/alerts/active")
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["code"] == "MB_ALERTS_DISABLED"
    finally:
        app.dependency_overrides.clear()


def test_feature_status_always_available(monkeypatch):
    monkeypatch.setenv("MB_ALERTS_ENABLED", "false")
    monkeypatch.setenv("MB_ALERT_CREATION_ENABLED", "false")
    monkeypatch.setenv("MB_PAPER_ANALYTICS_ENABLED", "false")

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    async def _user_id() -> str:
        return "user-1"

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_current_user_id] = _user_id
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/strategy/momentum-breakout/feature-status")
        assert resp.status_code == 200
        flags = resp.json()["flags"]
        assert flags["alertsEnabled"] is False
        assert flags["alertCreationEnabled"] is False
        assert flags["paperAnalyticsEnabled"] is False
    finally:
        app.dependency_overrides.clear()


def test_creation_disabled_blocks_persist_but_allows_read(
    monkeypatch, alert_service
):
    monkeypatch.setenv("MB_ALERTS_ENABLED", "true")
    monkeypatch.setenv("MB_ALERT_CREATION_ENABLED", "false")

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    async def _user_id() -> str:
        return "user-1"

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_current_user_id] = _user_id
    app.dependency_overrides[get_momentum_breakout_alert_service] = lambda: alert_service
    try:
        client = TestClient(app)
        active = client.get("/api/v1/strategy/momentum-breakout/alerts/active")
        assert active.status_code == 200

        create = client.post(
            "/api/v1/strategy/momentum-breakout/trade-plan-alert",
            json={"symbol": "AAPL", "persistAlert": True},
        )
        assert create.status_code == 503
        assert create.json()["detail"]["code"] == "MB_ALERT_CREATION_DISABLED"
    finally:
        app.dependency_overrides.clear()


def test_notifications_disabled_suppresses_emission(monkeypatch):
    monkeypatch.setenv("MB_ALERT_NOTIFICATIONS_ENABLED", "false")
    notifications = CompositeNotificationService()
    emitter = MomentumBreakoutNotificationEmitter(notifications)
    record = _build_record()
    emitter.on_alert_created(record)
    assert not notifications.list_notifications(record.user_id)


def test_paper_analytics_disabled_hides_endpoints(monkeypatch):
    monkeypatch.setenv("MB_ALERTS_ENABLED", "true")
    monkeypatch.setenv("MB_PAPER_ANALYTICS_ENABLED", "false")

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    async def _user_id() -> str:
        return "user-1"

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_current_user_id] = _user_id
    try:
        client = TestClient(app)
        resp = client.get(
            "/api/v1/strategy/momentum-breakout/performance/summary"
        )
        assert resp.status_code == 503
        assert resp.json()["detail"]["code"] == "MB_PAPER_ANALYTICS_DISABLED"
    finally:
        app.dependency_overrides.clear()


def test_admin_metrics_returns_expected_counters():
    alert_store = InMemoryMomentumBreakoutAlertStore()
    paper_store = InMemoryPaperTradePerformanceStore()
    paper_service = PaperTradePerformanceService(paper_store)
    record = _build_record()
    alert_store.save("user-1", record)
    paper_service.backfill_from_alert(record)
    get_momentum_breakout_ops_metrics().record_alert_created()

    service = MomentumBreakoutAdminMetricsService(
        alert_store=alert_store,
        paper_store=paper_store,
    )
    snapshot = service.snapshot
    assert snapshot.active_alerts_count == 1
    assert snapshot.paper_trade_rows_count == 1
    assert snapshot.alerts_created_today == 1
    assert snapshot.status_counts.pending_entry == 1
