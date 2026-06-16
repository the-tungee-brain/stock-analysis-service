"""Daily scheduler for completed Day Trade Replay missed-move persistence."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, time, timezone

from app.builders.intraday_trading_bias_engine import EASTERN
from app.services.day_trade_replay_persistence_service import (
    DayTradeReplayPersistenceService,
)
from app.services.trade_replay_service import _current_trading_date

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SEC = 300
_MIN_INTERVAL_SEC = 60
_MAX_INTERVAL_SEC = 1800
_DEFAULT_RUN_START_ET = "16:15"
_DEFAULT_RUN_END_ET = "16:30"


def is_day_trade_replay_persistence_scheduler_enabled() -> bool:
    return os.getenv(
        "DAY_TRADE_REPLAY_PERSISTENCE_SCHEDULER_ENABLED",
        "false",
    ).lower() in {"1", "true", "yes", "on"}


def resolve_day_trade_replay_persistence_interval_sec() -> int:
    raw = int(
        os.getenv(
            "DAY_TRADE_REPLAY_PERSISTENCE_INTERVAL_SEC",
            str(_DEFAULT_INTERVAL_SEC),
        )
    )
    return max(_MIN_INTERVAL_SEC, min(_MAX_INTERVAL_SEC, raw))


def resolve_day_trade_replay_persistence_window_et() -> tuple[time, time]:
    start = _parse_hhmm(
        os.getenv("DAY_TRADE_REPLAY_PERSISTENCE_START_ET", _DEFAULT_RUN_START_ET)
    )
    end = _parse_hhmm(
        os.getenv("DAY_TRADE_REPLAY_PERSISTENCE_END_ET", _DEFAULT_RUN_END_ET)
    )
    if end <= start:
        raise ValueError(
            "DAY_TRADE_REPLAY_PERSISTENCE_END_ET must be after START_ET"
        )
    return start, end


async def run_day_trade_replay_persistence_scheduler(
    persistence_service: DayTradeReplayPersistenceService,
    *,
    interval_sec: int | None = None,
) -> None:
    interval = interval_sec or resolve_day_trade_replay_persistence_interval_sec()
    start_time, end_time = resolve_day_trade_replay_persistence_window_et()
    completed_dates: set[str] = set()
    logger.info(
        (
            "Day trade replay persistence scheduler started "
            "(interval=%ss window_et=%s-%s)"
        ),
        interval,
        start_time.strftime("%H:%M"),
        end_time.strftime("%H:%M"),
    )

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            now_et = now_utc.astimezone(EASTERN)
            trading_date = _current_trading_date(now_utc)
            date_key = trading_date.isoformat()
            if start_time <= now_et.time() <= end_time and date_key not in completed_dates:
                result = await asyncio.to_thread(
                    persistence_service.persist_for_trading_date,
                    trading_date,
                )
                completed_dates.add(date_key)
                logger.info(
                    "Day trade replay persistence scheduler run complete: %s",
                    result,
                )
            elif now_et.time() < start_time:
                completed_dates.discard(date_key)
        except asyncio.CancelledError:
            logger.info("Day trade replay persistence scheduler stopped")
            raise
        except Exception:
            logger.exception("Day trade replay persistence scheduler iteration failed")
        await asyncio.sleep(interval)


def _parse_hhmm(value: str) -> time:
    hour_text, minute_text = value.strip().split(":", maxsplit=1)
    return time(hour=int(hour_text), minute=int(minute_text))
