"""App-layer helpers for Momentum Breakout alert lifecycle DTOs."""

from __future__ import annotations

from trade_planner.alerts.lifecycle_models import (
    AlertLifecycleEvent,
    MomentumBreakoutAlertRecord,
)
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from app.models.momentum_breakout_alert_models import (
    AlertLifecycleEventDto,
    MomentumBreakoutAlertRecordDto,
)

LIFECYCLE_DISCLAIMER = (
    "Educational alert tracking only. Not investment advice. No orders are placed."
)


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


def record_to_dto(
    record: MomentumBreakoutAlertRecord,
    *,
    lifecycle: AlertLifecycleService,
    include_events: bool = False,
) -> MomentumBreakoutAlertRecordDto:
    events: list[AlertLifecycleEventDto] = []
    if include_events:
        raw = lifecycle.list_lifecycle_events(record.user_id, record.alert_id)
        events = [event_to_dto(event) for event in raw]
    return MomentumBreakoutAlertRecordDto(
        alertId=record.alert_id,
        symbol=record.symbol,
        setupName=record.setup_name,
        createdAt=record.created_at,
        signalDate=record.signal_date,
        entryPrice=record.entry_price,
        stopPrice=record.stop_price,
        targetPrice=record.target_price,
        entryIsStop=record.entry_is_stop,
        status=record.status.value,
        expiresAt=record.expires_at,
        triggeredAt=record.triggered_at,
        exitAt=record.exit_at,
        exitPrice=record.exit_price,
        outcomeReturnPct=record.outcome_return_pct,
        riskGateAction=record.risk_gate_action,
        riskGateReasons=list(record.risk_gate_reasons),
        historicalWinRate=record.historical_win_rate,
        historicalProfitFactor=record.historical_profit_factor,
        historicalTotalTrades=record.historical_total_trades,
        lifecycleEvents=events,
    )
