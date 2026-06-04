"""Tests for Momentum Breakout API DTOs and notifications."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient

from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import InMemoryMomentumBreakoutAlertStore
from trade_planner.alerts.risk_models import AlertGateAction
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup

from app.auth.dependencies import get_current_user, get_current_user_id
from app.dependencies.service_dependencies import (
    get_momentum_breakout_notification_service,
)
from app.main import app
from app.notifications.composite_service import CompositeNotificationService
from app.notifications.momentum_breakout_models import (
    MomentumBreakoutNotificationEventType,
)
from app.services.strategy.momentum_breakout_alert_dto import record_to_alert_dto
from app.services.strategy.momentum_breakout_notification_emitter import (
    MomentumBreakoutNotificationEmitter,
)
from app.services.strategy.notifying_alert_lifecycle_service import (
    NotifyingAlertLifecycleService,
)

USER = "user-notify-1"
SETUP = MomentumBreakoutSetup.name


@pytest.fixture
def notification_service() -> CompositeNotificationService:
    return CompositeNotificationService()


@pytest.fixture
def emitter(
    notification_service: CompositeNotificationService,
) -> MomentumBreakoutNotificationEmitter:
    return MomentumBreakoutNotificationEmitter(notification_service)


@pytest.fixture
def lifecycle(
    emitter: MomentumBreakoutNotificationEmitter,
) -> NotifyingAlertLifecycleService:
    return NotifyingAlertLifecycleService(
        InMemoryMomentumBreakoutAlertStore(),
        emitter,
    )


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


class TestAlertDtoMapping:
    def test_pending_entry_dto_fields(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        dto = record_to_alert_dto(created)
        assert dto.symbol == "NVDA"
        assert dto.status == "PENDING_ENTRY"
        assert dto.setup_name == SETUP
        assert dto.direction == "LONG"
        assert dto.risk_reward == 2.0
        assert dto.next_action_message
        assert "entry" in dto.next_action_message.lower()

    def test_open_dto_after_entry(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        updated = lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=101.0,
            timestamp=datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc),
        )
        dto = record_to_alert_dto(updated)
        assert dto.status == "OPEN"
        assert "stop" in dto.next_action_message.lower()


class TestLifecycleNotifications:
    def _events(
        self, notification_service: CompositeNotificationService
    ) -> list[MomentumBreakoutNotificationEventType]:
        rows = notification_service.list_notifications(USER)
        return [row.event_type for row in rows]

    def test_entry_triggered_notification(
        self,
        lifecycle: NotifyingAlertLifecycleService,
        notification_service: CompositeNotificationService,
    ) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=101.0,
            timestamp=datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc),
        )
        events = self._events(notification_service)
        assert MomentumBreakoutNotificationEventType.ALERT_CREATED in events
        assert MomentumBreakoutNotificationEventType.ENTRY_TRIGGERED in events
        entry_rows = [
            row
            for row in notification_service.list_notifications(USER)
            if row.event_type == MomentumBreakoutNotificationEventType.ENTRY_TRIGGERED
        ]
        assert len(entry_rows) == 1
        assert entry_rows[0].severity == "watch"
        assert "entry triggered" in entry_rows[0].body.lower()
        assert "$100.00" in entry_rows[0].body
        assert entry_rows[0].alert_snapshot_json["status"] == "OPEN"

    def test_target_hit_notification(
        self,
        lifecycle: NotifyingAlertLifecycleService,
        notification_service: CompositeNotificationService,
    ) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=101.0,
            timestamp=datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc),
        )
        lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=110.0,
            timestamp=datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
        )
        events = self._events(notification_service)
        assert MomentumBreakoutNotificationEventType.TARGET_HIT in events
        row = next(
            row
            for row in notification_service.list_notifications(USER)
            if row.event_type == MomentumBreakoutNotificationEventType.TARGET_HIT
        )
        assert row.severity == "info"
        assert "target reached" in row.body.lower()

    def test_stop_hit_notification(
        self,
        lifecycle: NotifyingAlertLifecycleService,
        notification_service: CompositeNotificationService,
    ) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=101.0,
            timestamp=datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc),
        )
        lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=94.0,
            timestamp=datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc),
        )
        events = self._events(notification_service)
        assert MomentumBreakoutNotificationEventType.STOP_HIT in events
        row = next(
            row
            for row in notification_service.list_notifications(USER)
            if row.event_type == MomentumBreakoutNotificationEventType.STOP_HIT
        )
        assert row.severity == "warning"

    def test_expired_notification(
        self,
        lifecycle: NotifyingAlertLifecycleService,
        notification_service: CompositeNotificationService,
    ) -> None:
        record = AlertLifecycleService.build_record(
            user_id=USER,
            symbol="NVDA",
            signal_date=date(2020, 1, 1),
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            entry_is_stop=True,
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        created = lifecycle.create_alert(record)
        lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=99.0,
            timestamp=datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc),
        )
        events = self._events(notification_service)
        assert MomentumBreakoutNotificationEventType.EXPIRED in events

    def test_risk_gate_block_notification(
        self,
        emitter: MomentumBreakoutNotificationEmitter,
        notification_service: CompositeNotificationService,
    ) -> None:
        emitter.on_risk_gate_blocked(
            USER,
            symbol="NVDA",
            setup_name=SETUP,
            reasons=(
                "Max open positions reached (5/5 active momentum_breakout trades).",
                "Educational only — not investment advice.",
            ),
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            risk_gate_action=AlertGateAction.BLOCK.value,
        )
        row = notification_service.list_notifications(USER)[0]
        assert row.event_type == MomentumBreakoutNotificationEventType.BLOCKED_BY_RISK_GATE
        assert row.severity == "critical"
        assert "blocked by risk controls" in row.body.lower()
        assert "buy now" not in row.body.lower()
        assert "guaranteed" not in row.body.lower()

    def test_risk_gate_warning_notification(
        self,
        emitter: MomentumBreakoutNotificationEmitter,
        notification_service: CompositeNotificationService,
    ) -> None:
        emitter.on_risk_gate_warning(
            USER,
            symbol="NVDA",
            setup_name=SETUP,
            reasons=("Elevated mega-cap correlation exposure.",),
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            risk_gate_action=AlertGateAction.WARN.value,
        )
        row = notification_service.list_notifications(USER)[0]
        assert row.event_type == MomentumBreakoutNotificationEventType.WARNING_BY_RISK_GATE
        assert row.severity == "warning"


class TestNotificationApi:
    def test_mark_notification_as_read(
        self,
        emitter: MomentumBreakoutNotificationEmitter,
        notification_service: CompositeNotificationService,
    ) -> None:
        emitter.on_risk_gate_blocked(
            USER,
            symbol="NVDA",
            setup_name=SETUP,
            reasons=("Max open positions reached.",),
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
        )
        notification_id = notification_service.list_notifications(USER)[0].notification_id

        class _FakeUser:
            identity_sub = USER

        async def _user() -> _FakeUser:
            return _FakeUser()

        async def _user_id() -> str:
            return USER

        app.dependency_overrides[get_current_user] = _user
        app.dependency_overrides[get_current_user_id] = _user_id
        app.dependency_overrides[get_momentum_breakout_notification_service] = (
            lambda: notification_service
        )
        try:
            client = TestClient(app)
            list_resp = client.get(
                "/api/v1/strategy/momentum-breakout/notifications"
            )
            assert list_resp.status_code == 200
            body = list_resp.json()
            assert body["notifications"][0]["read"] is False
            assert body["notifications"][0]["severity"] == "critical"
            assert "alert" in body["notifications"][0]
            assert "viewModel" not in body["notifications"][0]

            read_resp = client.post(
                f"/api/v1/strategy/momentum-breakout/notifications/"
                f"{notification_id}/read"
            )
            assert read_resp.status_code == 200
            assert read_resp.json()["notification"]["read"] is True

            list_after = client.get(
                "/api/v1/strategy/momentum-breakout/notifications",
                params={"unreadOnly": True},
            )
            assert list_after.status_code == 200
            assert list_after.json()["notifications"] == []
        finally:
            app.dependency_overrides.clear()
