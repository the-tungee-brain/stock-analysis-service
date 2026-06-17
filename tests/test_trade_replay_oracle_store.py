from __future__ import annotations

from datetime import date, datetime, timezone

from app.adapters.trade_replay_oracle_store import (
    _event_from_row,
    _missed_move_from_row,
)


def test_oracle_event_date_datetime_is_normalized_to_date() -> None:
    event = _event_from_row(
        (
            "event-1",
            "plan-1",
            "NVDA",
            datetime(2026, 6, 16),
            "day_trade",
            "long_trigger_activated",
            datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc),
            100.0,
            100.5,
            "NVDA triggered",
            "important",
            "active",
            "historical",
            "Educational / delayed",
            "dedupe-1",
            datetime(2026, 6, 16, 14, 1, tzinfo=timezone.utc),
        )
    )

    assert event.event_date == date(2026, 6, 16)
    assert type(event.event_date) is date


def test_oracle_missed_move_event_date_datetime_is_normalized_to_date() -> None:
    missed_move = _missed_move_from_row(
        (
            "mm-1",
            "NVDA",
            "day_trade",
            datetime(2026, 6, 16),
            "Long opening range breakout",
            "long",
            datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc),
            100.0,
            "target_hit",
            0.04,
            2.0,
            100.0,
            98.0,
            104.0,
            108.0,
            99.99,
            96.0,
            98.0,
            2,
            "historical",
            "Educational / delayed",
            "event-trigger",
            "event-terminal",
            "[]",
            datetime(2026, 6, 16, 14, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 16, 14, 2, tzinfo=timezone.utc),
        )
    )

    assert missed_move.event_date == date(2026, 6, 16)
    assert type(missed_move.event_date) is date
