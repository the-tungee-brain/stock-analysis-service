"""App-layer re-exports for Momentum Breakout alert API DTO mapping."""

from __future__ import annotations

from app.services.strategy.momentum_breakout_alert_dto import (
    event_to_dto,
    record_to_alert_dto,
)

LIFECYCLE_DISCLAIMER = (
    "Educational alert tracking only. Not investment advice. No orders are placed."
)

# Backward-compatible alias used by routes.
record_to_dto = record_to_alert_dto

__all__ = [
    "LIFECYCLE_DISCLAIMER",
    "event_to_dto",
    "record_to_alert_dto",
    "record_to_dto",
]
