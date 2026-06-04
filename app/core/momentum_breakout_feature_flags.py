"""Feature flags for controlled Momentum Breakout alert rollout."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, *, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class MomentumBreakoutFeatureFlags:
    alerts_enabled: bool
    alert_creation_enabled: bool
    alert_notifications_enabled: bool
    paper_analytics_enabled: bool


def get_momentum_breakout_feature_flags() -> MomentumBreakoutFeatureFlags:
    alerts = _env_bool("MB_ALERTS_ENABLED", default=True)
    return MomentumBreakoutFeatureFlags(
        alerts_enabled=alerts,
        alert_creation_enabled=alerts and _env_bool("MB_ALERT_CREATION_ENABLED", default=True),
        alert_notifications_enabled=alerts
        and _env_bool("MB_ALERT_NOTIFICATIONS_ENABLED", default=True),
        paper_analytics_enabled=alerts
        and _env_bool("MB_PAPER_ANALYTICS_ENABLED", default=True),
    )


def mb_alerts_enabled() -> bool:
    return get_momentum_breakout_feature_flags().alerts_enabled


def mb_alert_creation_enabled() -> bool:
    return get_momentum_breakout_feature_flags().alert_creation_enabled


def mb_alert_notifications_enabled() -> bool:
    return get_momentum_breakout_feature_flags().alert_notifications_enabled


def mb_paper_analytics_enabled() -> bool:
    return get_momentum_breakout_feature_flags().paper_analytics_enabled
