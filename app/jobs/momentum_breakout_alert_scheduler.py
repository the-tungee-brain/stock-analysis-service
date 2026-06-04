"""Background scheduler for Momentum Breakout alert lifecycle updates."""

from __future__ import annotations

import asyncio
import logging
import os

from app.core.momentum_breakout_feature_flags import mb_alerts_enabled
from app.core.momentum_breakout_monitoring import log_mb_event
from app.services.strategy.momentum_breakout_alert_refresh_service import (
    MomentumBreakoutAlertRefreshService,
)
from app.services.strategy.momentum_breakout_ops_metrics import (
    get_momentum_breakout_ops_metrics,
)

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SEC = 180
_MIN_INTERVAL_SEC = 60
_MAX_INTERVAL_SEC = 300


def resolve_refresh_interval_sec() -> int:
    raw = int(os.getenv("MB_ALERT_REFRESH_INTERVAL_SEC", str(_DEFAULT_INTERVAL_SEC)))
    return max(_MIN_INTERVAL_SEC, min(_MAX_INTERVAL_SEC, raw))


def is_scheduler_enabled() -> bool:
    return os.getenv("MB_ALERT_SCHEDULER_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def run_momentum_breakout_alert_scheduler(
    refresh_service: MomentumBreakoutAlertRefreshService,
    *,
    interval_sec: int | None = None,
) -> None:
    """Poll active alerts and apply lifecycle transitions during market hours."""
    interval = interval_sec or resolve_refresh_interval_sec()
    logger.info(
        "Momentum Breakout alert scheduler started (interval=%ss)", interval
    )
    while True:
        if not mb_alerts_enabled():
            await asyncio.sleep(interval)
            continue
        try:
            result = await asyncio.to_thread(
                refresh_service.refresh_all_active_alerts,
                force=False,
            )
            if result.skipped_market_hours:
                logger.debug("MB alert refresh skipped (outside market hours)")
            elif result.changes:
                logger.info(
                    "MB alert refresh: processed=%s updated=%s changes=%s",
                    result.processed,
                    result.updated,
                    len(result.changes),
                )
            for warning in result.warnings:
                logger.warning(warning)
        except asyncio.CancelledError:
            logger.info("Momentum Breakout alert scheduler stopped")
            raise
        except Exception as exc:
            log_mb_event("scheduler_refresh_failed", error=str(exc))
            get_momentum_breakout_ops_metrics().record_scheduler_failure(
                error=str(exc),
            )
            logger.exception("Momentum Breakout alert scheduler iteration failed")
        await asyncio.sleep(interval)
