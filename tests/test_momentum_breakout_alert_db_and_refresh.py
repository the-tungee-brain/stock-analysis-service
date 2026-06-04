"""DB store, scheduled refresh, and manual refresh for MB alert lifecycle."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import DuplicateActiveMomentumAlertError
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from app.adapters.strategy.sqlite_momentum_breakout_alert_store import (
    SqliteMomentumBreakoutAlertStore,
)
from app.main import app
from app.services.strategy.momentum_breakout_alert_refresh_service import (
    MomentumBreakoutAlertRefreshService,
)
from app.auth.dependencies import get_current_user, get_current_user_id
from app.dependencies.service_dependencies import (
    get_momentum_breakout_alert_refresh_service,
    get_momentum_breakout_alert_service,
)
from app.services.strategy.momentum_breakout_alert_service import (
    MomentumBreakoutAlertService,
)

USER = "user-db-1"
SETUP = MomentumBreakoutSetup.name


@pytest.fixture
def sqlite_store(tmp_path) -> SqliteMomentumBreakoutAlertStore:
    return SqliteMomentumBreakoutAlertStore(tmp_path / "mb_alerts.db")


@pytest.fixture
def lifecycle(sqlite_store) -> AlertLifecycleService:
    return AlertLifecycleService(sqlite_store)


def _record(
    lifecycle: AlertLifecycleService,
    *,
    symbol: str = "NVDA",
    signal_date: date | None = None,
) -> object:
    today = date.today()
    sig = signal_date or today
    created = datetime.now(timezone.utc)
    return AlertLifecycleService.build_record(
        user_id=USER,
        symbol=symbol,
        signal_date=sig,
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        entry_is_stop=True,
        created_at=created,
    )


class TestSqliteStoreCrud:
    def test_create_update_list(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        created = lifecycle.mark_target_hit(
            USER, created.alert_id, exit_price=110.0
        )

        history = lifecycle.list_alert_history(USER)
        assert len(history) == 1
        assert history[0].status == AlertLifecycleStatus.TARGET_HIT

        events = lifecycle.list_lifecycle_events(USER, created.alert_id)
        assert len(events) >= 2

    def test_duplicate_survives_restart(
        self, tmp_path, lifecycle: AlertLifecycleService
    ) -> None:
        db_path = tmp_path / "restart.db"
        store1 = SqliteMomentumBreakoutAlertStore(db_path)
        life1 = AlertLifecycleService(store1)
        life1.create_alert(_record(life1, symbol="AAPL"))

        store2 = SqliteMomentumBreakoutAlertStore(db_path)
        life2 = AlertLifecycleService(store2)
        with pytest.raises(DuplicateActiveMomentumAlertError):
            life2.create_alert(_record(life2, symbol="AAPL"))

    def test_list_all_active(self, lifecycle: AlertLifecycleService) -> None:
        lifecycle.create_alert(_record(lifecycle, symbol="MSFT"))
        lifecycle.create_alert(_record(lifecycle, symbol="NVDA"))
        all_active = lifecycle.store.list_all_active()
        assert len(all_active) == 2


class _FakePriceProvider:
    def __init__(self, prices: dict[str, float | None]) -> None:
        self._prices = {k.upper(): v for k, v in prices.items()}

    def get_latest_price(self, symbol: str) -> float | None:
        return self._prices.get(symbol.upper())


class TestScheduledStyleRefresh:
    def test_pending_to_open(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        refresh = MomentumBreakoutAlertRefreshService(
            lifecycle_service=lifecycle,
            price_provider=_FakePriceProvider({"NVDA": 101.0}),
        )
        result = refresh.refresh_all_active_alerts(force=True)
        updated = lifecycle.get_alert(USER, created.alert_id)
        assert updated is not None
        assert updated.status == AlertLifecycleStatus.OPEN
        assert result.updated >= 1

    def test_open_to_target_hit(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        lifecycle.update_with_latest_price(
            USER,
            created.alert_id,
            symbol="NVDA",
            price=101.0,
            timestamp=datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc),
        )
        refresh = MomentumBreakoutAlertRefreshService(
            lifecycle_service=lifecycle,
            price_provider=_FakePriceProvider({"NVDA": 110.0}),
        )
        refresh.refresh_all_active_alerts(force=True)
        updated = lifecycle.get_alert(USER, created.alert_id)
        assert updated is not None
        assert updated.status == AlertLifecycleStatus.TARGET_HIT

    def test_failed_quote_does_not_corrupt(
        self, lifecycle: AlertLifecycleService
    ) -> None:
        created = lifecycle.create_alert(_record(lifecycle))
        refresh = MomentumBreakoutAlertRefreshService(
            lifecycle_service=lifecycle,
            price_provider=_FakePriceProvider({"NVDA": None}),
        )
        result = refresh.refresh_all_active_alerts(force=True)
        updated = lifecycle.get_alert(USER, created.alert_id)
        assert updated is not None
        assert updated.status == AlertLifecycleStatus.PENDING_ENTRY
        assert len(result.warnings) == 1

    def test_pending_expires_on_refresh(
        self, lifecycle: AlertLifecycleService
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
        refresh = MomentumBreakoutAlertRefreshService(
            lifecycle_service=lifecycle,
            price_provider=_FakePriceProvider({"NVDA": 99.0}),
        )
        refresh.refresh_all_active_alerts(force=True)
        updated = lifecycle.get_alert(USER, created.alert_id)
        assert updated is not None
        assert updated.status == AlertLifecycleStatus.EXPIRED


class TestManualRefreshEndpoint:
    def test_refresh_endpoint(self, lifecycle: AlertLifecycleService) -> None:
        created = lifecycle.create_alert(_record(lifecycle, symbol="META"))
        refresh = MomentumBreakoutAlertRefreshService(
            lifecycle_service=lifecycle,
            price_provider=_FakePriceProvider({"META": 101.0}),
        )
        alert_service = MomentumBreakoutAlertService(
            lifecycle_service=lifecycle,
            risk_gate=MagicMock(),
            alert_engine=MagicMock(),
        )

        class _FakeUser:
            identity_sub = USER

        async def _user() -> _FakeUser:
            return _FakeUser()

        async def _user_id() -> str:
            return USER

        app.dependency_overrides[get_current_user] = _user
        app.dependency_overrides[get_current_user_id] = _user_id
        app.dependency_overrides[get_momentum_breakout_alert_refresh_service] = (
            lambda: refresh
        )
        app.dependency_overrides[get_momentum_breakout_alert_service] = (
            lambda: alert_service
        )
        try:
            client = TestClient(app)
            response = client.post(
                "/api/v1/strategy/momentum-breakout/alerts/refresh"
            )
            assert response.status_code == 200
            body = response.json()
            assert body["processed"] == 1
            assert body["updated"] >= 1
            assert body["alerts"][0]["status"] == "OPEN"
        finally:
            app.dependency_overrides.clear()
