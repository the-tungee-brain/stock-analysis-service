"""US equity trading-day helpers for alert expiry and signal freshness."""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")
_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)

STALE_SIGNAL_MESSAGE = (
    "Signal is stale; latest setup bar is older than allowed."
)


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5


def is_trading_day(d: date) -> bool:
    """Weekday calendar days (NYSE holidays not modeled)."""
    return _is_weekday(d)


def previous_trading_day(d: date) -> date:
    cursor = d - timedelta(days=1)
    while not is_trading_day(cursor):
        cursor -= timedelta(days=1)
    return cursor


def next_trading_day(d: date) -> date:
    cursor = d + timedelta(days=1)
    while not is_trading_day(cursor):
        cursor += timedelta(days=1)
    return cursor


def add_trading_days(start: date, count: int) -> date:
    if count < 0:
        raise ValueError("count must be non-negative")
    cursor = start
    for _ in range(count):
        cursor = next_trading_day(cursor)
    return cursor


def trading_days_apart(earlier: date, later: date) -> int:
    """Number of trading sessions strictly after ``earlier`` up to ``later`` (0 = same day)."""
    if later < earlier:
        raise ValueError("later must be on or after earlier")
    if later == earlier:
        return 0
    count = 0
    cursor = earlier
    while cursor < later:
        cursor = next_trading_day(cursor)
        count += 1
    return count


def latest_completed_bar_trading_day(as_of: datetime) -> date:
    """
    Trading date of the latest completed daily bar at ``as_of`` (US regular session, ET).

    Before the open on a weekday → previous trading day.
    During/after the open through end of that weekday → that day.
    Weekends → previous Friday.
    """
    if as_of.tzinfo is None:
        instant = as_of.replace(tzinfo=timezone.utc)
    else:
        instant = as_of
    et = instant.astimezone(_EASTERN)
    day = et.date()

    if not is_trading_day(day):
        cursor = day
        while not is_trading_day(cursor):
            cursor -= timedelta(days=1)
        return cursor

    if et.time() < _REGULAR_OPEN:
        return previous_trading_day(day)
    return day


def end_of_regular_session(trading_day: date) -> datetime:
    """16:00 US/Eastern on ``trading_day``."""
    return datetime.combine(trading_day, _REGULAR_CLOSE, tzinfo=_EASTERN)


def alert_expiry_trading_day_offset() -> int:
    raw = os.getenv("MB_ALERT_EXPIRY_TRADING_DAYS_OFFSET", "1").strip()
    try:
        offset = int(raw)
    except ValueError:
        offset = 1
    return max(1, offset)


def max_signal_stale_trading_days() -> int:
    raw = os.getenv("MB_ALERT_MAX_SIGNAL_STALE_TRADING_DAYS", "1").strip()
    try:
        allowed = int(raw)
    except ValueError:
        allowed = 1
    return max(0, allowed)


def end_of_next_trading_day_expiry(
    *,
    created_at: datetime,
    trading_days_offset: int | None = None,
) -> datetime:
    """Expire at the close of the Nth trading day after the creation session day."""
    if created_at.tzinfo is None:
        created = created_at.replace(tzinfo=timezone.utc)
    else:
        created = created_at
    offset = (
        trading_days_offset
        if trading_days_offset is not None
        else alert_expiry_trading_day_offset()
    )
    session_day = latest_completed_bar_trading_day(created)
    expiry_day = add_trading_days(session_day, offset)
    expiry = end_of_regular_session(expiry_day)
    if expiry <= created.astimezone(_EASTERN):
        expiry = end_of_regular_session(add_trading_days(session_day, offset + 1))
    return expiry.astimezone(timezone.utc)


def validate_signal_date_freshness(
    signal_date: date,
    *,
    created_at: datetime,
    max_stale_trading_days: int | None = None,
) -> str | None:
    """
    Return an error message when ``signal_date`` is too old vs the latest bar at creation.
    """
    allowed = (
        max_stale_trading_days
        if max_stale_trading_days is not None
        else max_signal_stale_trading_days()
    )
    if created_at.tzinfo is None:
        created = created_at.replace(tzinfo=timezone.utc)
    else:
        created = created_at
    reference = latest_completed_bar_trading_day(created)
    if signal_date > reference:
        return STALE_SIGNAL_MESSAGE
    gap = trading_days_apart(signal_date, reference)
    if gap > allowed:
        return STALE_SIGNAL_MESSAGE
    return None
