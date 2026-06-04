"""Alert lifecycle state machine for Momentum Breakout."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from trade_planner.alerts.lifecycle_models import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    AlertLifecycleEvent,
    AlertLifecycleEventType,
    AlertLifecycleStatus,
    MomentumBreakoutAlertRecord,
    StaleMomentumSignalError,
    long_return_pct,
    new_alert_id,
)
from trade_planner.alerts.market_calendar import (
    end_of_next_trading_day_expiry,
    validate_signal_date_freshness,
)
from trade_planner.alerts.lifecycle_store import (
    AlertNotCancellableError,
    DuplicateActiveMomentumAlertError,
    InMemoryMomentumBreakoutAlertStore,
    MomentumBreakoutAlertStore,
)
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup


ALERT_CANCELLED_BY_USER_MESSAGE = "Alert cancelled by user."


class AlertLifecycleService:
    SETUP_NAME = MomentumBreakoutSetup.name

    def __init__(
        self,
        store: MomentumBreakoutAlertStore | None = None,
    ) -> None:
        self._store = store or InMemoryMomentumBreakoutAlertStore()

    @property
    def store(self) -> MomentumBreakoutAlertStore:
        return self._store

    def create_alert(self, record: MomentumBreakoutAlertRecord) -> MomentumBreakoutAlertRecord:
        symbol = record.symbol.upper()
        if self._store.has_active_for_symbol(
            record.user_id, symbol, record.setup_name
        ):
            raise DuplicateActiveMomentumAlertError(
                f"Active {record.setup_name} alert already exists for {symbol}."
            )

        if not record.alert_id:
            record = replace(record, alert_id=new_alert_id())
        if record.status not in ACTIVE_STATUSES:
            record = replace(record, status=AlertLifecycleStatus.PENDING_ENTRY)
        if record.expires_at.tzinfo is None:
            record = replace(
                record,
                expires_at=record.expires_at.replace(tzinfo=timezone.utc),
            )
        if record.created_at.tzinfo is None:
            record = replace(
                record,
                created_at=record.created_at.replace(tzinfo=timezone.utc),
            )
        if record.expires_at <= record.created_at:
            raise ValueError(
                "expires_at must be later than created_at for new alerts."
            )

        self._store.save(record.user_id, record)
        self._log_event(
            record.user_id,
            alert_id=record.alert_id,
            event_type=AlertLifecycleEventType.CREATED,
            from_status=None,
            to_status=record.status,
            price=None,
            recorded_at=record.created_at,
            message=f"Alert created for {symbol} {record.setup_name}.",
        )
        return record

    def update_with_latest_price(
        self,
        user_id: str,
        alert_id: str,
        *,
        symbol: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        record = self._require_record(user_id, alert_id)
        if record.symbol.upper() != symbol.upper():
            raise ValueError(f"Symbol mismatch: alert is {record.symbol}, got {symbol}")

        ts = self._aware(timestamp or datetime.now(timezone.utc))
        if record.status in TERMINAL_STATUSES:
            return record

        self._log_event(
            user_id,
            alert_id=alert_id,
            event_type=AlertLifecycleEventType.PRICE_UPDATE,
            from_status=record.status,
            to_status=record.status,
            price=price,
            recorded_at=ts,
            message=f"Price update {price:.4f} for {record.symbol}.",
        )

        if record.status == AlertLifecycleStatus.PENDING_ENTRY:
            if ts >= record.expires_at:
                return self.mark_expired(user_id, alert_id, recorded_at=ts)
            if price >= record.entry_price:
                record = self.mark_entry_triggered(
                    user_id, alert_id, price=price, recorded_at=ts
                )
                return self._set_open(user_id, record, price=price, recorded_at=ts)

        if record.status == AlertLifecycleStatus.ENTRY_TRIGGERED:
            return self._set_open(user_id, record, price=price, recorded_at=ts)

        if record.status == AlertLifecycleStatus.OPEN:
            if price <= record.stop_price:
                return self.mark_stop_hit(
                    user_id, alert_id, exit_price=price, recorded_at=ts
                )
            if price >= record.target_price:
                return self.mark_target_hit(
                    user_id, alert_id, exit_price=price, recorded_at=ts
                )

        return self._store.get(user_id, alert_id) or record

    def mark_entry_triggered(
        self,
        user_id: str,
        alert_id: str,
        *,
        price: float | None = None,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        record = self._require_record(user_id, alert_id)
        ts = self._aware(recorded_at or datetime.now(timezone.utc))
        updated = record.with_status(
            AlertLifecycleStatus.ENTRY_TRIGGERED,
            triggered_at=ts,
        )
        self._store.save(user_id, updated)
        self._log_event(
            user_id,
            alert_id=alert_id,
            event_type=AlertLifecycleEventType.ENTRY_TRIGGERED,
            from_status=record.status,
            to_status=AlertLifecycleStatus.ENTRY_TRIGGERED,
            price=price,
            recorded_at=ts,
            message=f"Entry triggered at {record.entry_price:.4f}.",
        )
        return updated

    def mark_target_hit(
        self,
        user_id: str,
        alert_id: str,
        *,
        exit_price: float,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        return self._close(
            user_id,
            alert_id,
            status=AlertLifecycleStatus.TARGET_HIT,
            event_type=AlertLifecycleEventType.TARGET_HIT,
            exit_price=exit_price,
            recorded_at=recorded_at,
            message=f"Target hit at {exit_price:.4f}.",
        )

    def mark_stop_hit(
        self,
        user_id: str,
        alert_id: str,
        *,
        exit_price: float,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        return self._close(
            user_id,
            alert_id,
            status=AlertLifecycleStatus.STOP_HIT,
            event_type=AlertLifecycleEventType.STOP_HIT,
            exit_price=exit_price,
            recorded_at=recorded_at,
            message=f"Stop hit at {exit_price:.4f}.",
        )

    def mark_expired(
        self,
        user_id: str,
        alert_id: str,
        *,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        record = self._require_record(user_id, alert_id)
        ts = self._aware(recorded_at or datetime.now(timezone.utc))
        updated = record.with_status(AlertLifecycleStatus.EXPIRED, exit_at=ts)
        self._store.save(user_id, updated)
        self._log_event(
            user_id,
            alert_id=alert_id,
            event_type=AlertLifecycleEventType.EXPIRED,
            from_status=record.status,
            to_status=AlertLifecycleStatus.EXPIRED,
            price=None,
            recorded_at=ts,
            message="Alert expired without entry fill.",
        )
        return updated

    def cancel_alert(
        self,
        user_id: str,
        alert_id: str,
        *,
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        record = self._require_record(user_id, alert_id)
        if record.status not in ACTIVE_STATUSES:
            raise AlertNotCancellableError(
                f"Alert {alert_id} cannot be cancelled from status {record.status.value}."
            )
        return self.mark_cancelled(
            user_id,
            alert_id,
            reason=ALERT_CANCELLED_BY_USER_MESSAGE,
            recorded_at=recorded_at,
        )

    def mark_cancelled(
        self,
        user_id: str,
        alert_id: str,
        *,
        reason: str = "Cancelled",
        recorded_at: datetime | None = None,
    ) -> MomentumBreakoutAlertRecord:
        record = self._require_record(user_id, alert_id)
        ts = self._aware(recorded_at or datetime.now(timezone.utc))
        updated = record.with_status(AlertLifecycleStatus.CANCELLED, exit_at=ts)
        self._store.save(user_id, updated)
        self._log_event(
            user_id,
            alert_id=alert_id,
            event_type=AlertLifecycleEventType.CANCELLED,
            from_status=record.status,
            to_status=AlertLifecycleStatus.CANCELLED,
            price=None,
            recorded_at=ts,
            message=reason,
        )
        return updated

    def list_active_alerts(self, user_id: str) -> tuple[MomentumBreakoutAlertRecord, ...]:
        return self._store.list_active(user_id)

    def list_alert_history(
        self, user_id: str, *, limit: int = 100
    ) -> tuple[MomentumBreakoutAlertRecord, ...]:
        return self._store.list_history(user_id, limit=limit)

    def list_lifecycle_events(
        self, user_id: str, alert_id: str
    ) -> tuple[AlertLifecycleEvent, ...]:
        return self._store.list_events(user_id, alert_id)

    def get_alert(
        self, user_id: str, alert_id: str
    ) -> MomentumBreakoutAlertRecord | None:
        return self._store.get(user_id, alert_id)

    @staticmethod
    def build_record(
        *,
        user_id: str,
        symbol: str,
        signal_date,
        entry_price: float,
        stop_price: float,
        target_price: float,
        entry_is_stop: bool,
        created_at: datetime | None = None,
        risk_gate_action: str = "",
        risk_gate_reasons: tuple[str, ...] | list[str] = (),
        historical_win_rate: float | None = None,
        historical_profit_factor: float | None = None,
        historical_total_trades: int | None = None,
        setup_name: str | None = None,
        market_regime: str | None = None,
        volume_ratio: float | None = None,
        rs_percentile: float | None = None,
    ) -> MomentumBreakoutAlertRecord:
        now = created_at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        stale_reason = validate_signal_date_freshness(signal_date, created_at=now)
        if stale_reason is not None:
            raise StaleMomentumSignalError(stale_reason)
        expires_at = end_of_next_trading_day_expiry(created_at=now)
        return MomentumBreakoutAlertRecord(
            alert_id=new_alert_id(),
            user_id=user_id,
            symbol=symbol.upper(),
            setup_name=setup_name or MomentumBreakoutSetup.name,
            created_at=now,
            signal_date=signal_date,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            entry_is_stop=entry_is_stop,
            status=AlertLifecycleStatus.PENDING_ENTRY,
            expires_at=expires_at,
            risk_gate_action=risk_gate_action,
            risk_gate_reasons=tuple(risk_gate_reasons),
            historical_win_rate=historical_win_rate,
            historical_profit_factor=historical_profit_factor,
            historical_total_trades=historical_total_trades,
            market_regime=market_regime,
            volume_ratio=volume_ratio,
            rs_percentile=rs_percentile,
        )

    def _set_open(
        self,
        user_id: str,
        record: MomentumBreakoutAlertRecord,
        *,
        price: float,
        recorded_at: datetime,
    ) -> MomentumBreakoutAlertRecord:
        if record.status == AlertLifecycleStatus.OPEN:
            return record
        prior = record.status
        updated = record.with_status(
            AlertLifecycleStatus.OPEN,
            triggered_at=record.triggered_at or recorded_at,
        )
        self._store.save(user_id, updated)
        self._log_event(
            user_id,
            alert_id=record.alert_id,
            event_type=AlertLifecycleEventType.STATUS_CHANGED,
            from_status=prior,
            to_status=AlertLifecycleStatus.OPEN,
            price=price,
            recorded_at=recorded_at,
            message=f"Position open for {record.symbol}.",
        )
        return updated

    def _close(
        self,
        user_id: str,
        alert_id: str,
        *,
        status: AlertLifecycleStatus,
        event_type: AlertLifecycleEventType,
        exit_price: float,
        recorded_at: datetime | None,
        message: str,
    ) -> MomentumBreakoutAlertRecord:
        record = self._require_record(user_id, alert_id)
        ts = self._aware(recorded_at or datetime.now(timezone.utc))
        entry = record.entry_price
        ret = long_return_pct(entry, exit_price)
        updated = record.with_status(
            status,
            exit_at=ts,
            exit_price=exit_price,
            outcome_return_pct=round(ret, 6),
        )
        self._store.save(user_id, updated)
        self._log_event(
            user_id,
            alert_id=alert_id,
            event_type=event_type,
            from_status=record.status,
            to_status=status,
            price=exit_price,
            recorded_at=ts,
            message=message,
        )
        return updated

    def _require_record(
        self, user_id: str, alert_id: str
    ) -> MomentumBreakoutAlertRecord:
        record = self._store.get(user_id, alert_id)
        if record is None:
            raise ValueError(f"Alert {alert_id} not found for user.")
        return record

    def _log_event(
        self,
        user_id: str,
        *,
        alert_id: str,
        event_type: AlertLifecycleEventType,
        from_status: AlertLifecycleStatus | None,
        to_status: AlertLifecycleStatus,
        price: float | None,
        recorded_at: datetime,
        message: str,
    ) -> None:
        event = AlertLifecycleEvent(
            event_id=str(uuid4()),
            alert_id=alert_id,
            event_type=event_type,
            from_status=from_status,
            to_status=to_status,
            price=price,
            recorded_at=self._aware(recorded_at),
            message=message,
        )
        self._store.append_event(user_id, event)

    @staticmethod
    def _aware(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
