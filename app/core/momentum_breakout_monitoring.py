"""Structured monitoring logs for Momentum Breakout alerts."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("momentum_breakout.ops")


def log_mb_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    try:
        message = json.dumps(payload, default=str)
    except TypeError:
        message = str(payload)
    logger.info(message)


def log_mb_warning(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    try:
        message = json.dumps(payload, default=str)
    except TypeError:
        message = str(payload)
    logger.warning(message)
