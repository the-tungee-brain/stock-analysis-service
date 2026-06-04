"""Paper-trading performance sync and analytics."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user, get_current_user_id
from app.dependencies.service_dependencies import get_paper_trade_analytics_service
from app.main import app
from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import InMemoryMomentumBreakoutAlertStore
from trade_planner.alerts.paper_trade_store import InMemoryPaperTradePerformanceStore
from app.services.strategy.paper_trade_analytics_service import PaperTradeAnalyticsService
from app.services.strategy.paper_trade_performance_service import (
    PaperTradePerformanceService,
)
from app.services.strategy.paper_trade_recording_lifecycle_service import (
    PaperTradeRecordingLifecycleService,
)
from app.notifications.composite_service import CompositeNotificationService
from app.services.strategy.momentum_breakout_notification_emitter import (
    MomentumBreakoutNotificationEmitter,
)


def _build_record(**kwargs):
    defaults = dict(
        user_id="user-1",
        symbol="AAPL",
        signal_date=date(2025, 1, 10),
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        entry_is_stop=True,
        market_regime="RISK_ON",
        volume_ratio=1.8,
        rs_percentile=85.0,
    )
    defaults.update(kwargs)
    return AlertLifecycleService.build_record(**defaults)


@pytest.fixture
def lifecycle_stack():
    alert_store = InMemoryMomentumBreakoutAlertStore()
    paper_store = InMemoryPaperTradePerformanceStore()
    paper_service = PaperTradePerformanceService(paper_store)
    emitter = MomentumBreakoutNotificationEmitter(CompositeNotificationService())
    lifecycle = PaperTradeRecordingLifecycleService(
        alert_store,
        emitter,
        paper_service,
    )
    analytics = PaperTradeAnalyticsService(performance_service=paper_service)
    return lifecycle, analytics, paper_service


def test_sync_on_entry_and_target(lifecycle_stack):
    lifecycle, analytics, paper = lifecycle_stack
    record = lifecycle.create_alert(_build_record())
    ts = datetime(2025, 1, 11, 15, 0, tzinfo=timezone.utc)
    lifecycle.mark_entry_triggered(record.user_id, record.alert_id, recorded_at=ts)
    lifecycle.update_with_latest_price(
        record.user_id,
        record.alert_id,
        symbol=record.symbol,
        price=105.0,
        timestamp=ts + timedelta(days=1),
    )
    lifecycle.update_with_latest_price(
        record.user_id,
        record.alert_id,
        symbol=record.symbol,
        price=110.0,
        timestamp=ts + timedelta(days=2),
    )
    rows = paper.list_records(record.user_id)
    assert len(rows) == 1
    assert rows[0].status == AlertLifecycleStatus.TARGET_HIT.value
    assert rows[0].outcome_return_pct == pytest.approx(0.1, rel=1e-4)

    summary = analytics.summary(record.user_id)
    assert summary.total_alerts == 1
    assert summary.triggered_alerts == 1
    assert summary.win_rate == pytest.approx(1.0)
    assert summary.expectancy == pytest.approx(0.1, rel=1e-3)


def test_expired_without_entry(lifecycle_stack):
    lifecycle, analytics, _paper = lifecycle_stack
    record = lifecycle.create_alert(_build_record())
    ts = record.expires_at + timedelta(seconds=1)
    lifecycle.mark_expired(record.user_id, record.alert_id, recorded_at=ts)
    summary = analytics.summary(record.user_id)
    assert summary.expired_alerts == 1
    assert summary.triggered_alerts == 0


def test_performance_api_routes(lifecycle_stack):
    lifecycle, analytics, _ = lifecycle_stack
    record = lifecycle.create_alert(_build_record())
    ts = datetime(2025, 1, 11, 15, 0, tzinfo=timezone.utc)
    lifecycle.mark_entry_triggered(record.user_id, record.alert_id, recorded_at=ts)
    lifecycle.mark_stop_hit(
        record.user_id,
        record.alert_id,
        exit_price=95.0,
        recorded_at=ts + timedelta(days=1),
    )

    class _FakeUser:
        identity_sub = record.user_id

    async def _user() -> _FakeUser:
        return _FakeUser()

    async def _user_id() -> str:
        return record.user_id

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_current_user_id] = _user_id
    app.dependency_overrides[get_paper_trade_analytics_service] = lambda: analytics
    try:
        client = TestClient(app)
        summary = client.get(
            "/api/v1/strategy/momentum-breakout/performance/summary"
        )
        assert summary.status_code == 200
        body = summary.json()
        assert body["meta"]["label"] == "Live paper-trading performance"
        assert body["summary"]["totalAlerts"] == 1
        assert body["summary"]["winRate"] == 0.0

        trades = client.get(
            "/api/v1/strategy/momentum-breakout/performance/trades"
        )
        assert trades.status_code == 200
        assert len(trades.json()["trades"]) == 1

        by_symbol = client.get(
            "/api/v1/strategy/momentum-breakout/performance/by-symbol"
        )
        assert by_symbol.status_code == 200
        assert by_symbol.json()["buckets"][0]["key"] == "AAPL"
    finally:
        app.dependency_overrides.clear()
