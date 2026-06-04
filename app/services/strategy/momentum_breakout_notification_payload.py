"""API notification payload fields (title, body, severity) for client rendering."""

from __future__ import annotations

from dataclasses import dataclass

from app.notifications.momentum_breakout_models import (
    MomentumBreakoutNotificationEventType,
)

NotificationSeverity = str  # info | watch | warning | critical


@dataclass(frozen=True, slots=True)
class NotificationPayload:
    title: str
    body: str
    severity: NotificationSeverity


EVENT_SEVERITY: dict[MomentumBreakoutNotificationEventType, NotificationSeverity] = {
    MomentumBreakoutNotificationEventType.ALERT_CREATED: "info",
    MomentumBreakoutNotificationEventType.ENTRY_TRIGGERED: "watch",
    MomentumBreakoutNotificationEventType.TARGET_HIT: "info",
    MomentumBreakoutNotificationEventType.STOP_HIT: "warning",
    MomentumBreakoutNotificationEventType.EXPIRED: "info",
    MomentumBreakoutNotificationEventType.BLOCKED_BY_RISK_GATE: "critical",
    MomentumBreakoutNotificationEventType.WARNING_BY_RISK_GATE: "warning",
}


def severity_for_event(
    event_type: MomentumBreakoutNotificationEventType,
) -> NotificationSeverity:
    return EVENT_SEVERITY[event_type]


def _format_price(value: float) -> str:
    return f"${value:,.2f}"


def _summarize_reasons(reasons: tuple[str, ...]) -> str:
    filtered = [
        reason
        for reason in reasons
        if reason
        and "not investment advice" not in reason.lower()
        and "educational" not in reason.lower()
    ]
    if not filtered:
        return "risk limits applied."
    primary = filtered[0]
    if "max open positions" in primary.lower():
        return "max open positions reached."
    if len(primary) > 120:
        return primary[:117] + "..."
    return primary


def payload_alert_created(
    symbol: str,
    *,
    setup_name: str,
    entry_price: float,
) -> NotificationPayload:
    return NotificationPayload(
        title=f"{symbol} trade plan alert created",
        body=(
            f"New {setup_name} educational alert for {symbol} at entry "
            f"{_format_price(entry_price)}. Monitoring only — no orders placed."
        ),
        severity=severity_for_event(
            MomentumBreakoutNotificationEventType.ALERT_CREATED
        ),
    )


def payload_entry_triggered(
    symbol: str,
    *,
    setup_name: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> NotificationPayload:
    return NotificationPayload(
        title=f"{symbol} trade plan entry triggered",
        body=(
            f"{symbol} {setup_name} entry triggered at {_format_price(entry_price)}. "
            f"Stop: {_format_price(stop_price)}. "
            f"Target: {_format_price(target_price)}."
        ),
        severity=severity_for_event(
            MomentumBreakoutNotificationEventType.ENTRY_TRIGGERED
        ),
    )


def payload_target_hit(symbol: str, *, setup_name: str) -> NotificationPayload:
    return NotificationPayload(
        title=f"{symbol} trade plan target reached",
        body=f"{symbol} {setup_name} target reached. Trade plan completed.",
        severity=severity_for_event(MomentumBreakoutNotificationEventType.TARGET_HIT),
    )


def payload_stop_hit(symbol: str, *, setup_name: str) -> NotificationPayload:
    return NotificationPayload(
        title=f"{symbol} trade plan stop reached",
        body=f"{symbol} {setup_name} stop level reached. Trade plan exited.",
        severity=severity_for_event(MomentumBreakoutNotificationEventType.STOP_HIT),
    )


def payload_expired(symbol: str, *, setup_name: str) -> NotificationPayload:
    return NotificationPayload(
        title=f"{symbol} trade plan expired",
        body=f"{symbol} {setup_name} alert expired before entry was triggered.",
        severity=severity_for_event(MomentumBreakoutNotificationEventType.EXPIRED),
    )


def payload_blocked_by_risk_gate(
    symbol: str,
    reasons: tuple[str, ...],
    *,
    setup_name: str,
) -> NotificationPayload:
    detail = _summarize_reasons(reasons)
    return NotificationPayload(
        title=f"{symbol} alert blocked by risk controls",
        body=f"{setup_name} alert blocked by risk controls: {detail}",
        severity=severity_for_event(
            MomentumBreakoutNotificationEventType.BLOCKED_BY_RISK_GATE
        ),
    )


def payload_warning_by_risk_gate(
    symbol: str,
    reasons: tuple[str, ...],
    *,
    setup_name: str,
) -> NotificationPayload:
    detail = _summarize_reasons(reasons)
    return NotificationPayload(
        title=f"{symbol} alert risk warning",
        body=(
            f"{setup_name} alert flagged by risk controls: {detail} "
            "Review before relying on this trade plan."
        ),
        severity=severity_for_event(
            MomentumBreakoutNotificationEventType.WARNING_BY_RISK_GATE
        ),
    )
