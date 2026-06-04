"""FastAPI dependencies for Momentum Breakout feature flags."""

from __future__ import annotations

from fastapi import HTTPException

from app.core.momentum_breakout_feature_flags import (
    get_momentum_breakout_feature_flags,
    mb_alert_creation_enabled,
    mb_alert_notifications_enabled,
    mb_alerts_enabled,
    mb_paper_analytics_enabled,
)


def _disabled_detail(feature: str, code: str) -> dict[str, str]:
    return {
        "detail": f"Momentum Breakout {feature} is temporarily unavailable.",
        "code": code,
    }


def require_mb_alerts_enabled() -> None:
    if not mb_alerts_enabled():
        raise HTTPException(
            status_code=503,
            detail=_disabled_detail("alerts", "MB_ALERTS_DISABLED"),
        )


def require_mb_alert_creation_enabled() -> None:
    require_mb_alerts_enabled()
    if not mb_alert_creation_enabled():
        raise HTTPException(
            status_code=503,
            detail=_disabled_detail("alert creation", "MB_ALERT_CREATION_DISABLED"),
        )


def require_mb_paper_analytics_enabled() -> None:
    require_mb_alerts_enabled()
    if not mb_paper_analytics_enabled():
        raise HTTPException(
            status_code=503,
            detail=_disabled_detail(
                "paper-trading analytics",
                "MB_PAPER_ANALYTICS_DISABLED",
            ),
        )


def require_mb_admin_metrics_access() -> None:
    require_mb_alerts_enabled()
