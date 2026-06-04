"""Map DB rows to Momentum Breakout alert lifecycle models."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

from trade_planner.alerts.lifecycle_models import (
    ACTIVE_STATUSES,
    AlertLifecycleEvent,
    AlertLifecycleEventType,
    AlertLifecycleStatus,
    MomentumBreakoutAlertRecord,
)

ACTIVE_STATUS_VALUES = frozenset(status.value for status in ACTIVE_STATUSES)


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value)[:10])


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _aware(value)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return _aware(datetime.fromisoformat(text))


def _parse_reasons(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    text = str(raw).strip()
    if not text:
        return ()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return tuple(str(item) for item in parsed)
    except json.JSONDecodeError:
        pass
    return (text,)


def record_from_row(row: dict[str, Any]) -> MomentumBreakoutAlertRecord:
    return MomentumBreakoutAlertRecord(
        alert_id=str(row["alert_id"]),
        user_id=str(row["user_id"]),
        symbol=str(row["symbol"]).upper(),
        setup_name=str(row["setup_name"]),
        created_at=_parse_datetime(row["created_at"]),
        signal_date=_parse_date(row["signal_date"]),
        entry_price=float(row["entry_price"]),
        stop_price=float(row["stop_price"]),
        target_price=float(row["target_price"]),
        entry_is_stop=bool(row["entry_is_stop"]),
        status=AlertLifecycleStatus(str(row["status"])),
        expires_at=_parse_datetime(row["expires_at"]),
        triggered_at=_parse_datetime(row["triggered_at"])
        if row.get("triggered_at")
        else None,
        exit_at=_parse_datetime(row["exit_at"]) if row.get("exit_at") else None,
        exit_price=float(row["exit_price"]) if row.get("exit_price") is not None else None,
        outcome_return_pct=float(row["outcome_return_pct"])
        if row.get("outcome_return_pct") is not None
        else None,
        risk_gate_action=str(row.get("risk_gate_action") or ""),
        risk_gate_reasons=_parse_reasons(row.get("risk_gate_reasons")),
        historical_win_rate=float(row["historical_win_rate"])
        if row.get("historical_win_rate") is not None
        else None,
        historical_profit_factor=float(row["historical_profit_factor"])
        if row.get("historical_profit_factor") is not None
        else None,
        historical_total_trades=int(row["historical_total_trades"])
        if row.get("historical_total_trades") is not None
        else None,
        market_regime=str(row["market_regime"]) if row.get("market_regime") else None,
        volume_ratio=float(row["volume_ratio"])
        if row.get("volume_ratio") is not None
        else None,
        rs_percentile=float(row["rs_percentile"])
        if row.get("rs_percentile") is not None
        else None,
    )


def event_from_row(row: dict[str, Any]) -> AlertLifecycleEvent:
    from_status = row.get("from_status")
    return AlertLifecycleEvent(
        event_id=str(row["event_id"]),
        alert_id=str(row["alert_id"]),
        event_type=AlertLifecycleEventType(str(row["event_type"])),
        from_status=AlertLifecycleStatus(from_status) if from_status else None,
        to_status=AlertLifecycleStatus(str(row["to_status"])),
        price=float(row["price"]) if row.get("price") is not None else None,
        recorded_at=_parse_datetime(row["recorded_at"]),
        message=str(row.get("message") or ""),
    )


def record_to_row(record: MomentumBreakoutAlertRecord) -> dict[str, Any]:
    return {
        "alert_id": record.alert_id,
        "user_id": record.user_id,
        "symbol": record.symbol.upper(),
        "setup_name": record.setup_name,
        "created_at": record.created_at,
        "signal_date": record.signal_date,
        "entry_price": record.entry_price,
        "stop_price": record.stop_price,
        "target_price": record.target_price,
        "entry_is_stop": 1 if record.entry_is_stop else 0,
        "status": record.status.value,
        "expires_at": record.expires_at,
        "triggered_at": record.triggered_at,
        "exit_at": record.exit_at,
        "exit_price": record.exit_price,
        "outcome_return_pct": record.outcome_return_pct,
        "risk_gate_action": record.risk_gate_action,
        "risk_gate_reasons": json.dumps(list(record.risk_gate_reasons)),
        "historical_win_rate": record.historical_win_rate,
        "historical_profit_factor": record.historical_profit_factor,
        "historical_total_trades": record.historical_total_trades,
        "market_regime": record.market_regime,
        "volume_ratio": record.volume_ratio,
        "rs_percentile": record.rs_percentile,
    }


def event_to_row(user_id: str, event: AlertLifecycleEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "alert_id": event.alert_id,
        "user_id": user_id,
        "event_type": event.event_type.value,
        "from_status": event.from_status.value if event.from_status else None,
        "to_status": event.to_status.value,
        "price": event.price,
        "recorded_at": event.recorded_at,
        "message": event.message,
    }
