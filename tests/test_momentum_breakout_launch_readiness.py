"""Launch readiness and paper-trade backfill."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user, get_current_user_id
from app.dependencies.service_dependencies import (
    get_momentum_breakout_launch_readiness_service,
)
from app.main import app
from app.services.strategy.momentum_breakout_launch_readiness_service import (
    MomentumBreakoutLaunchReadinessService,
)
from app.services.strategy.momentum_breakout_paper_trade_backfill_service import (
    MomentumBreakoutPaperTradeBackfillService,
)
from app.services.strategy.paper_trade_performance_service import (
    PaperTradePerformanceService,
)
from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import InMemoryMomentumBreakoutAlertStore
from trade_planner.alerts.paper_trade_store import InMemoryPaperTradePerformanceStore


def _build_record(**kwargs):
    created_at = datetime(2025, 1, 10, 21, 0, tzinfo=timezone.utc)
    defaults = dict(
        user_id="user-1",
        symbol="AAPL",
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
def stores():
    alert_store = InMemoryMomentumBreakoutAlertStore()
    paper_store = InMemoryPaperTradePerformanceStore()
    paper_service = PaperTradePerformanceService(paper_store)
    return alert_store, paper_store, paper_service


@pytest.fixture
def readiness_service(stores):
    alert_store, paper_store, _ = stores
    price = MagicMock()
    price.get_latest_price.return_value = 500.0
    return MomentumBreakoutLaunchReadinessService(
        alert_store=alert_store,
        paper_store=paper_store,
        price_provider=price,
    )


def test_backfill_creates_missing_paper_records(stores):
    alert_store, paper_store, paper_service = stores
    record = _build_record()
    triggered = record.with_status(
        AlertLifecycleStatus.OPEN,
        triggered_at=datetime(2025, 1, 11, tzinfo=timezone.utc),
    )
    alert_store.save("user-1", triggered)

    backfill = MomentumBreakoutPaperTradeBackfillService(
        alert_store=alert_store,
        paper_trade_service=paper_service,
    )
    result = backfill.run()
    assert result.alerts_scanned == 1
    assert result.rows_created == 1
    assert result.rows_skipped == 0
    assert paper_store.get("user-1", triggered.alert_id) is not None


def test_backfill_dry_run_does_not_write(stores):
    alert_store, paper_store, paper_service = stores
    record = _build_record()
    alert_store.save("user-1", record)

    backfill = MomentumBreakoutPaperTradeBackfillService(
        alert_store=alert_store,
        paper_trade_service=paper_service,
    )
    result = backfill.run(dry_run=True)
    assert result.rows_created == 1
    assert paper_store.get("user-1", record.alert_id) is None


def test_backfill_does_not_duplicate(stores):
    alert_store, paper_store, paper_service = stores
    record = _build_record()
    alert_store.save("user-1", record)
    paper_service.backfill_from_alert(record)

    backfill = MomentumBreakoutPaperTradeBackfillService(
        alert_store=alert_store,
        paper_trade_service=paper_service,
    )
    result = backfill.run()
    assert result.rows_created == 0
    assert result.rows_skipped == 1


def test_readiness_warns_on_memory_store_in_production(stores, monkeypatch):
    alert_store, paper_store, _ = stores
    monkeypatch.setenv("MB_ALERT_STORE", "memory")
    monkeypatch.setenv("MB_PAPER_TRADE_STORE", "memory")
    monkeypatch.setenv("MB_PRODUCTION", "true")
    monkeypatch.setenv("MB_ALERT_SCHEDULER_ENABLED", "true")

    price = MagicMock()
    price.get_latest_price.return_value = 100.0
    service = MomentumBreakoutLaunchReadinessService(
        alert_store=alert_store,
        paper_store=paper_store,
        price_provider=price,
    )
    report = service.evaluate()
    assert any("in-memory" in warning for warning in report.warnings)
    assert report.ready is False


def test_readiness_warns_when_scheduler_disabled(stores, monkeypatch):
    alert_store, paper_store, _ = stores
    monkeypatch.setenv("MB_ALERT_SCHEDULER_ENABLED", "false")
    monkeypatch.delenv("MB_PRODUCTION", raising=False)

    price = MagicMock()
    price.get_latest_price.return_value = 100.0
    service = MomentumBreakoutLaunchReadinessService(
        alert_store=alert_store,
        paper_store=paper_store,
        price_provider=price,
    )
    report = service.evaluate()
    assert any("scheduler is disabled" in w for w in report.warnings)
    assert report.ready is False


def test_readiness_passes_on_valid_memory_config(stores, monkeypatch):
    alert_store, paper_store, _ = stores
    monkeypatch.setenv("MB_ALERT_STORE", "memory")
    monkeypatch.setenv("MB_PAPER_TRADE_STORE", "memory")
    monkeypatch.setenv("MB_ALERT_SCHEDULER_ENABLED", "true")
    monkeypatch.delenv("MB_PRODUCTION", raising=False)

    price = MagicMock()
    price.get_latest_price.return_value = 100.0
    service = MomentumBreakoutLaunchReadinessService(
        alert_store=alert_store,
        paper_store=paper_store,
        price_provider=price,
    )
    report = service.evaluate()
    assert report.ready is True
    assert report.alert_store_type == "memory"
    assert report.paper_trade_store_type == "memory"


def test_launch_readiness_api(readiness_service, monkeypatch):
    monkeypatch.setenv("MB_LAUNCH_READINESS_PUBLIC", "true")
    monkeypatch.setenv("MB_ALERT_STORE", "memory")
    monkeypatch.setenv("MB_PAPER_TRADE_STORE", "memory")

    class _FakeUser:
        identity_sub = "ops-user"

    async def _user() -> _FakeUser:
        return _FakeUser()

    async def _user_id() -> str:
        return "ops-user"

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_current_user_id] = _user_id
    app.dependency_overrides[get_momentum_breakout_launch_readiness_service] = (
        lambda: readiness_service
    )
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/strategy/momentum-breakout/launch-readiness")
        assert resp.status_code == 200
        body = resp.json()
        assert "ready" in body
        assert body["alertStoreType"] in {"memory", "sqlite", "oracle"}
        assert "healthChecks" in body
    finally:
        app.dependency_overrides.clear()
