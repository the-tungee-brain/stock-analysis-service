"""US equity regular session hours (Eastern)."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")
_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)


def is_us_regular_market_hours(now: datetime | None = None) -> bool:
    """True during NYSE regular session (Mon–Fri 09:30–16:00 ET)."""
    current = now or datetime.now(_EASTERN)
    if current.tzinfo is None:
        current = current.replace(tzinfo=_EASTERN)
    else:
        current = current.astimezone(_EASTERN)

    if current.weekday() >= 5:
        return False

    t = current.time()
    return _REGULAR_OPEN <= t <= _REGULAR_CLOSE
