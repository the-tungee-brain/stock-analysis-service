"""Map DB rows to paper-trading performance records."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from trade_planner.alerts.paper_trade_models import PaperTradePerformanceRecord


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


def record_from_row(row: dict[str, Any]) -> PaperTradePerformanceRecord:
    return PaperTradePerformanceRecord(
        alert_id=str(row["alert_id"]),
        user_id=str(row["user_id"]),
        symbol=str(row["symbol"]).upper(),
        setup_name=str(row["setup_name"]),
        signal_date=_parse_date(row["signal_date"]),
        entry_triggered_at=_parse_datetime(row["entry_triggered_at"])
        if row.get("entry_triggered_at")
        else None,
        entry_price=float(row["entry_price"]),
        stop_price=float(row["stop_price"]),
        target_price=float(row["target_price"]),
        exit_at=_parse_datetime(row["exit_at"]) if row.get("exit_at") else None,
        exit_price=float(row["exit_price"]) if row.get("exit_price") is not None else None,
        status=str(row["status"]),
        outcome_return_pct=float(row["outcome_return_pct"])
        if row.get("outcome_return_pct") is not None
        else None,
        holding_days=int(row["holding_days"])
        if row.get("holding_days") is not None
        else None,
        risk_gate_action=str(row.get("risk_gate_action") or ""),
        market_regime=str(row["market_regime"]) if row.get("market_regime") else None,
        volume_ratio=float(row["volume_ratio"])
        if row.get("volume_ratio") is not None
        else None,
        rs_percentile=float(row["rs_percentile"])
        if row.get("rs_percentile") is not None
        else None,
        created_at=_parse_datetime(row["created_at"]),
    )


def record_to_row(record: PaperTradePerformanceRecord) -> dict[str, Any]:
    return {
        "alert_id": record.alert_id,
        "user_id": record.user_id,
        "symbol": record.symbol.upper(),
        "setup_name": record.setup_name,
        "signal_date": record.signal_date,
        "entry_triggered_at": record.entry_triggered_at,
        "entry_price": record.entry_price,
        "stop_price": record.stop_price,
        "target_price": record.target_price,
        "exit_at": record.exit_at,
        "exit_price": record.exit_price,
        "status": record.status,
        "outcome_return_pct": record.outcome_return_pct,
        "holding_days": record.holding_days,
        "risk_gate_action": record.risk_gate_action,
        "market_regime": record.market_regime,
        "volume_ratio": record.volume_ratio,
        "rs_percentile": record.rs_percentile,
        "created_at": record.created_at,
    }
