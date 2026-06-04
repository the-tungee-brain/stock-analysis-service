from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.market_hours import is_us_regular_market_hours

_EASTERN = ZoneInfo("America/New_York")


def test_weekday_regular_session() -> None:
    dt = datetime(2024, 6, 3, 11, 0, tzinfo=_EASTERN)
    assert is_us_regular_market_hours(dt) is True


def test_weekend_closed() -> None:
    dt = datetime(2024, 6, 1, 11, 0, tzinfo=_EASTERN)
    assert is_us_regular_market_hours(dt) is False
