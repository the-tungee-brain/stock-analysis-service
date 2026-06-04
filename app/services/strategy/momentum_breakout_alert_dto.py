"""Map lifecycle records to API response DTOs (no UI/layout logic)."""

from __future__ import annotations

from trade_planner.alerts.lifecycle_models import (
    AlertLifecycleEvent,
    AlertLifecycleStatus,
    MomentumBreakoutAlertRecord,
)
from trade_planner.alerts.lifecycle_service import AlertLifecycleService

from app.models.momentum_breakout_alert_models import (
    AlertLifecycleEventDto,
    MomentumBreakoutAlertDto,
)

NEXT_ACTION_BY_STATUS: dict[AlertLifecycleStatus, str] = {
    AlertLifecycleStatus.PENDING_ENTRY: (
        "Watch for a break above the entry level or wait for expiration."
    ),
    AlertLifecycleStatus.ENTRY_TRIGGERED: (
        "Confirm stop and target levels in your trade plan."
    ),
    AlertLifecycleStatus.OPEN: (
        "Track price versus stop and target; monitoring only."
    ),
    AlertLifecycleStatus.TARGET_HIT: "Review outcome in alert history.",
    AlertLifecycleStatus.STOP_HIT: "Review outcome in alert history.",
    AlertLifecycleStatus.EXPIRED: "No further action for this expired alert.",
    AlertLifecycleStatus.CANCELLED: "No further action for this cancelled alert.",
}


def long_risk_reward(entry: float, stop: float, target: float) -> float:
    risk = entry - stop
    if risk <= 0:
        return 0.0
    return round((target - entry) / risk, 2)


def resolve_priority(risk_gate_action: str) -> str:
    action = (risk_gate_action or "").upper()
    if action in {"BLOCK", ""}:
        return "LOW"
    if action in {"WARN", "SIZE_DOWN"}:
        return "MEDIUM"
    return "HIGH"


def next_action_for_status(status: AlertLifecycleStatus) -> str:
    return NEXT_ACTION_BY_STATUS[status]


def event_to_dto(event: AlertLifecycleEvent) -> AlertLifecycleEventDto:
    return AlertLifecycleEventDto(
        eventId=event.event_id,
        eventType=event.event_type.value,
        fromStatus=event.from_status.value if event.from_status else None,
        toStatus=event.to_status.value,
        price=event.price,
        recordedAt=event.recorded_at,
        message=event.message,
    )


def record_to_alert_dto(
    record: MomentumBreakoutAlertRecord,
    *,
    lifecycle: AlertLifecycleService | None = None,
    include_events: bool = False,
    direction: str = "LONG",
) -> MomentumBreakoutAlertDto:
    events: list[AlertLifecycleEventDto] = []
    if include_events and lifecycle is not None:
        raw = lifecycle.list_lifecycle_events(record.user_id, record.alert_id)
        events = [event_to_dto(event) for event in raw]

    return MomentumBreakoutAlertDto(
        alertId=record.alert_id or None,
        symbol=record.symbol,
        setupName=record.setup_name,
        direction=direction,
        status=record.status.value,
        createdAt=record.created_at,
        signalDate=record.signal_date,
        entryPrice=record.entry_price,
        stopPrice=record.stop_price,
        targetPrice=record.target_price,
        riskReward=long_risk_reward(
            record.entry_price, record.stop_price, record.target_price
        ),
        entryIsStop=record.entry_is_stop,
        expiresAt=record.expires_at,
        triggeredAt=record.triggered_at,
        exitAt=record.exit_at,
        exitPrice=record.exit_price,
        outcomeReturnPct=record.outcome_return_pct,
        riskGateAction=record.risk_gate_action,
        riskGateReasons=list(record.risk_gate_reasons),
        priority=resolve_priority(record.risk_gate_action),
        historicalWinRate=record.historical_win_rate,
        historicalProfitFactor=record.historical_profit_factor,
        historicalTotalTrades=record.historical_total_trades,
        nextActionMessage=next_action_for_status(record.status),
        lifecycleEvents=events,
    )


def alert_dto_to_storage_dict(dto: MomentumBreakoutAlertDto) -> dict[str, object]:
    return dto.model_dump(mode="json", by_alias=True)
