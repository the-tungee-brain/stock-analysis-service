"""Tests for Momentum Breakout alert lifecycle tracking."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import (
    DuplicateActiveMomentumAlertError,
    InMemoryMomentumBreakoutAlertStore,
)
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup

SETUP = MomentumBreakoutSetup.name
USER = "user-test-1"


@pytest.fixture
def lifecycle() -> AlertLifecycleService:
    return AlertLifecycleService(InMemoryMomentumBreakoutAlertStore())


def _record(
    lifecycle: AlertLifecycleService,
    *,
    symbol: str = "NVDA",
    entry: float = 100.0,
    stop: float = 95.0,
    target: float = 110.0,
    signal_date: date | None = None,
) -> object:
    sig = signal_date or date(2024, 6, 1)
    return AlertLifecycleService.build_record(
        user_id=USER,
        symbol=symbol,
        signal_date=sig,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        entry_is_stop=True,
        created_at=datetime(2024, 6, 1, 15, 0, tzinfo=timezone.utc),
    )


class TestPendingEntryTriggers:
    def test_entry_then_open(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        ts = datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc)
        updated = lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=100.5,
            timestamp=ts,
        )
        assert updated.status == AlertLifecycleStatus.OPEN
        assert updated.triggered_at is not None
        events = lifecycle.list_lifecycle_events(USER, created.alert_id)
        event_types = [e.event_type.value for e in events]
        assert "CREATED" in event_types
        assert "ENTRY_TRIGGERED" in event_types


class TestOpenTargetHit:
    def test_target_hit(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        ts1 = datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc)
        lifecycle.update_with_latest_price(
            USER, created.alert_id, symbol="NVDA", price=101.0, timestamp=ts1
        )
        ts2 = datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc)
        updated = lifecycle.update_with_latest_price(
            USER, created.alert_id, symbol="NVDA", price=110.0, timestamp=ts2
        )
        assert updated.status == AlertLifecycleStatus.TARGET_HIT
        assert updated.outcome_return_pct == pytest.approx(0.10)


class TestOpenStopHit:
    def test_stop_hit(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        ts1 = datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc)
        lifecycle.update_with_latest_price(
            USER, created.alert_id, symbol="NVDA", price=101.0, timestamp=ts1
        )
        ts2 = datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc)
        updated = lifecycle.update_with_latest_price(
            USER, created.alert_id, symbol="NVDA", price=95.0, timestamp=ts2
        )
        assert updated.status == AlertLifecycleStatus.STOP_HIT
        assert updated.outcome_return_pct == pytest.approx(-0.05)


class TestPendingExpires:
    def test_expired_after_window(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(
            _record(lifecycle, signal_date=date(2024, 6, 1))
        )
        after_expiry = datetime(2024, 6, 3, 0, 0, tzinfo=timezone.utc)
        updated = lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=99.0,
            timestamp=after_expiry,
        )
        assert updated.status == AlertLifecycleStatus.EXPIRED
        events = lifecycle.list_lifecycle_events(USER, created.alert_id)
        assert any(e.event_type.value == "EXPIRED" for e in events)


class TestDuplicateActiveBlocked:
    def test_duplicate_symbol_rejected(self, lifecycle: AlertLifecycleService) -> None:
        lifecycle.create_alert(_record(lifecycle, symbol="NVDA"))
        with pytest.raises(DuplicateActiveMomentumAlertError):
            lifecycle.create_alert(_record(lifecycle, symbol="NVDA"))

    def test_allows_after_terminal(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle, symbol="AAPL"))
        lifecycle.mark_cancelled(USER, created.alert_id)
        second = lifecycle.create_alert(_record(lifecycle, symbol="AAPL"))
        assert second.alert_id != created.alert_id


class TestLifecycleEventHistory:
    def test_events_recorded_through_lifecycle(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=100.0,
            timestamp=datetime(2024, 6, 2, 12, 0, tzinfo=timezone.utc),
        )
        events = lifecycle.list_lifecycle_events(USER, created.alert_id)
        assert len(events) >= 2
        assert events[0].recorded_at <= events[-1].recorded_at

    def test_list_active_and_history(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle, symbol="MSFT"))
        active = lifecycle.list_active_alerts(USER)
        assert len(active) == 1
        lifecycle.mark_expired(USER, created.alert_id)
        assert len(lifecycle.list_active_alerts(USER)) == 0
        history = lifecycle.list_alert_history(USER)
        assert len(history) == 1
        assert history[0].status == AlertLifecycleStatus.EXPIRED
