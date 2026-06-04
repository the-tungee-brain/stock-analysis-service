"""Alert expiry and stale-signal rules for Momentum Breakout lifecycle."""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from trade_planner.alerts.lifecycle_models import StaleMomentumSignalError
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import InMemoryMomentumBreakoutAlertStore
from trade_planner.alerts.market_calendar import (
    STALE_SIGNAL_MESSAGE,
    end_of_next_trading_day_expiry,
    latest_completed_bar_trading_day,
    validate_signal_date_freshness,
)

_EASTERN = ZoneInfo("America/New_York")


def _et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_EASTERN)


class TestExpiryFromCreationTime:
    def test_created_after_market_close_expires_next_session(self) -> None:
        # Thursday 2026-06-04 after close; signal bar must be same day (not stale).
        created = _et(2026, 6, 4, 17, 30).astimezone(timezone.utc)
        record = AlertLifecycleService.build_record(
            user_id="u1",
            symbol="NVDA",
            signal_date=date(2026, 6, 4),
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            entry_is_stop=True,
            created_at=created,
        )
        assert record.expires_at > record.created_at
        assert record.expires_at == end_of_next_trading_day_expiry(created_at=created)
        expiry_et = record.expires_at.astimezone(_EASTERN)
        assert expiry_et.date() == date(2026, 6, 5)
        assert expiry_et.hour == 16

    def test_created_before_open_uses_previous_bar_day(self) -> None:
        # Friday 2026-06-05 pre-market: latest bar is Thursday 2026-06-04.
        created = _et(2026, 6, 5, 8, 0).astimezone(timezone.utc)
        assert latest_completed_bar_trading_day(created) == date(2026, 6, 4)
        record = AlertLifecycleService.build_record(
            user_id="u1",
            symbol="NVDA",
            signal_date=date(2026, 6, 4),
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            entry_is_stop=True,
            created_at=created,
        )
        assert record.expires_at > record.created_at
        expiry_et = record.expires_at.astimezone(_EASTERN)
        assert expiry_et.date() == date(2026, 6, 5)

    def test_expires_at_always_after_created_at(self) -> None:
        created = _et(2024, 6, 10, 12, 0).astimezone(timezone.utc)
        record = AlertLifecycleService.build_record(
            user_id="u1",
            symbol="AAPL",
            signal_date=date(2024, 6, 10),
            entry_price=50.0,
            stop_price=48.0,
            target_price=55.0,
            entry_is_stop=True,
            created_at=created,
        )
        lifecycle = AlertLifecycleService(InMemoryMomentumBreakoutAlertStore())
        saved = lifecycle.create_alert(record)
        assert saved.expires_at > saved.created_at


class TestStaleSignalBlocked:
    def test_stale_signal_rejected_on_build(self) -> None:
        # Created 2026-06-04 with signal bar 2026-06-02 (>1 trading day stale).
        created = _et(2026, 6, 4, 17, 0).astimezone(timezone.utc)
        with pytest.raises(StaleMomentumSignalError, match=STALE_SIGNAL_MESSAGE):
            AlertLifecycleService.build_record(
                user_id="u1",
                symbol="NVDA",
                signal_date=date(2026, 6, 2),
                entry_price=100.0,
                stop_price=95.0,
                target_price=110.0,
                entry_is_stop=True,
                created_at=created,
            )

    def test_one_trading_day_lag_allowed(self) -> None:
        created = _et(2026, 6, 4, 17, 0).astimezone(timezone.utc)
        record = AlertLifecycleService.build_record(
            user_id="u1",
            symbol="NVDA",
            signal_date=date(2026, 6, 3),
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            entry_is_stop=True,
            created_at=created,
        )
        assert record.signal_date == date(2026, 6, 3)

    def test_validate_signal_date_freshness_message(self) -> None:
        created = _et(2026, 6, 4, 17, 0).astimezone(timezone.utc)
        assert (
            validate_signal_date_freshness(date(2026, 6, 2), created_at=created)
            == STALE_SIGNAL_MESSAGE
        )


class TestRegressionBugScenario:
    def test_old_signal_date_no_longer_expires_before_creation(self) -> None:
        """Reproduce reported bug: created 2026-06-04 must not expire 2026-06-03."""
        created = _et(2026, 6, 4, 18, 0).astimezone(timezone.utc)
        with pytest.raises(StaleMomentumSignalError):
            AlertLifecycleService.build_record(
                user_id="u1",
                symbol="NVDA",
                signal_date=date(2026, 6, 2),
                entry_price=100.0,
                stop_price=95.0,
                target_price=110.0,
                entry_is_stop=True,
                created_at=created,
            )
