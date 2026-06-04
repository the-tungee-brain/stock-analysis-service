"""Oracle-backed Momentum Breakout alert lifecycle store."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import oracledb

from trade_planner.alerts.lifecycle_models import (
    AlertLifecycleEvent,
    MomentumBreakoutAlertRecord,
)
from app.adapters.strategy.momentum_breakout_alert_mappers import (
    ACTIVE_STATUS_VALUES,
    event_from_row,
    event_to_row,
    record_from_row,
    record_to_row,
)

_ALERT_TABLE = "MOMENTUM_BREAKOUT_ALERT"
_EVENT_TABLE = "MOMENTUM_BREAKOUT_ALERT_EVENT"


class OracleMomentumBreakoutAlertStore:
    def __init__(self, client: oracledb.ConnectionPool) -> None:
        self._client = client

    def get(self, user_id: str, alert_id: str) -> MomentumBreakoutAlertRecord | None:
        sql = f"""
            SELECT alert_id, user_id, symbol, setup_name, created_at, signal_date,
                   entry_price, stop_price, target_price, entry_is_stop, status,
                   expires_at, triggered_at, exit_at, exit_price, outcome_return_pct,
                   risk_gate_action, risk_gate_reasons, historical_win_rate,
                   historical_profit_factor, historical_total_trades,
                   market_regime, volume_ratio, rs_percentile
            FROM {_ALERT_TABLE}
            WHERE user_id = :user_id AND alert_id = :alert_id
        """
        row = self._fetchone(sql, {"user_id": user_id, "alert_id": alert_id})
        if row is None:
            return None
        return record_from_row(self._alert_row_dict(row))

    def save(self, user_id: str, record: MomentumBreakoutAlertRecord) -> None:
        payload = record_to_row(record)
        payload["user_id"] = user_id
        sql = f"""
            MERGE INTO {_ALERT_TABLE} t
            USING (
                SELECT
                    :alert_id AS alert_id,
                    :user_id AS user_id,
                    :symbol AS symbol,
                    :setup_name AS setup_name,
                    :created_at AS created_at,
                    :signal_date AS signal_date,
                    :entry_price AS entry_price,
                    :stop_price AS stop_price,
                    :target_price AS target_price,
                    :entry_is_stop AS entry_is_stop,
                    :status AS status,
                    :expires_at AS expires_at,
                    :triggered_at AS triggered_at,
                    :exit_at AS exit_at,
                    :exit_price AS exit_price,
                    :outcome_return_pct AS outcome_return_pct,
                    :risk_gate_action AS risk_gate_action,
                    :risk_gate_reasons AS risk_gate_reasons,
                    :historical_win_rate AS historical_win_rate,
                    :historical_profit_factor AS historical_profit_factor,
                    :historical_total_trades AS historical_total_trades,
                    :market_regime AS market_regime,
                    :volume_ratio AS volume_ratio,
                    :rs_percentile AS rs_percentile
                FROM dual
            ) s
            ON (t.alert_id = s.alert_id)
            WHEN MATCHED THEN UPDATE SET
                t.symbol = s.symbol,
                t.setup_name = s.setup_name,
                t.entry_price = s.entry_price,
                t.stop_price = s.stop_price,
                t.target_price = s.target_price,
                t.entry_is_stop = s.entry_is_stop,
                t.status = s.status,
                t.expires_at = s.expires_at,
                t.triggered_at = s.triggered_at,
                t.exit_at = s.exit_at,
                t.exit_price = s.exit_price,
                t.outcome_return_pct = s.outcome_return_pct,
                t.risk_gate_action = s.risk_gate_action,
                t.risk_gate_reasons = s.risk_gate_reasons,
                t.historical_win_rate = s.historical_win_rate,
                t.historical_profit_factor = s.historical_profit_factor,
                t.historical_total_trades = s.historical_total_trades,
                t.market_regime = s.market_regime,
                t.volume_ratio = s.volume_ratio,
                t.rs_percentile = s.rs_percentile,
                t.updated_at = systimestamp
            WHEN NOT MATCHED THEN INSERT (
                alert_id, user_id, symbol, setup_name, created_at, signal_date,
                entry_price, stop_price, target_price, entry_is_stop, status,
                expires_at, triggered_at, exit_at, exit_price, outcome_return_pct,
                risk_gate_action, risk_gate_reasons, historical_win_rate,
                historical_profit_factor, historical_total_trades,
                market_regime, volume_ratio, rs_percentile
            ) VALUES (
                s.alert_id, s.user_id, s.symbol, s.setup_name, s.created_at, s.signal_date,
                s.entry_price, s.stop_price, s.target_price, s.entry_is_stop, s.status,
                s.expires_at, s.triggered_at, s.exit_at, s.exit_price, s.outcome_return_pct,
                s.risk_gate_action, s.risk_gate_reasons, s.historical_win_rate,
                s.historical_profit_factor, s.historical_total_trades,
                s.market_regime, s.volume_ratio, s.rs_percentile
            )
        """
        self._execute(sql, payload)

    def has_active_for_symbol(
        self, user_id: str, symbol: str, setup_name: str
    ) -> bool:
        statuses = list(ACTIVE_STATUS_VALUES)
        placeholders = ", ".join(f":status_{idx}" for idx in range(len(statuses)))
        sql = f"""
            SELECT 1
            FROM {_ALERT_TABLE}
            WHERE user_id = :user_id
              AND symbol = :symbol
              AND setup_name = :setup_name
              AND status IN ({placeholders})
            FETCH FIRST 1 ROWS ONLY
        """
        params: dict[str, Any] = {
            "user_id": user_id,
            "symbol": symbol.upper(),
            "setup_name": setup_name,
        }
        for idx, status in enumerate(statuses):
            params[f"status_{idx}"] = status
        row = self._fetchone(sql, params)
        return row is not None

    def list_active(self, user_id: str) -> tuple[MomentumBreakoutAlertRecord, ...]:
        statuses = list(ACTIVE_STATUS_VALUES)
        placeholders = ", ".join(f":status_{idx}" for idx in range(len(statuses)))
        sql = f"""
            SELECT alert_id, user_id, symbol, setup_name, created_at, signal_date,
                   entry_price, stop_price, target_price, entry_is_stop, status,
                   expires_at, triggered_at, exit_at, exit_price, outcome_return_pct,
                   risk_gate_action, risk_gate_reasons, historical_win_rate,
                   historical_profit_factor, historical_total_trades,
                   market_regime, volume_ratio, rs_percentile
            FROM {_ALERT_TABLE}
            WHERE user_id = :user_id
              AND status IN ({placeholders})
            ORDER BY created_at DESC
        """
        params: dict[str, Any] = {"user_id": user_id}
        for idx, status in enumerate(statuses):
            params[f"status_{idx}"] = status
        rows = self._fetchall(sql, params)
        return tuple(record_from_row(self._alert_row_dict(row)) for row in rows)

    def list_all_active(self) -> tuple[MomentumBreakoutAlertRecord, ...]:
        statuses = list(ACTIVE_STATUS_VALUES)
        placeholders = ", ".join(f":status_{idx}" for idx in range(len(statuses)))
        sql = f"""
            SELECT alert_id, user_id, symbol, setup_name, created_at, signal_date,
                   entry_price, stop_price, target_price, entry_is_stop, status,
                   expires_at, triggered_at, exit_at, exit_price, outcome_return_pct,
                   risk_gate_action, risk_gate_reasons, historical_win_rate,
                   historical_profit_factor, historical_total_trades,
                   market_regime, volume_ratio, rs_percentile
            FROM {_ALERT_TABLE}
            WHERE status IN ({placeholders})
            ORDER BY created_at DESC
        """
        params = {f"status_{idx}": status for idx, status in enumerate(statuses)}
        rows = self._fetchall(sql, params)
        return tuple(record_from_row(self._alert_row_dict(row)) for row in rows)

    def list_history(
        self, user_id: str, *, limit: int = 100
    ) -> tuple[MomentumBreakoutAlertRecord, ...]:
        sql = f"""
            SELECT alert_id, user_id, symbol, setup_name, created_at, signal_date,
                   entry_price, stop_price, target_price, entry_is_stop, status,
                   expires_at, triggered_at, exit_at, exit_price, outcome_return_pct,
                   risk_gate_action, risk_gate_reasons, historical_win_rate,
                   historical_profit_factor, historical_total_trades,
                   market_regime, volume_ratio, rs_percentile
            FROM {_ALERT_TABLE}
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            FETCH FIRST :limit ROWS ONLY
        """
        rows = self._fetchall(sql, {"user_id": user_id, "limit": limit})
        return tuple(record_from_row(self._alert_row_dict(row)) for row in rows)

    def append_event(self, user_id: str, event: AlertLifecycleEvent) -> None:
        payload = event_to_row(user_id, event)
        sql = f"""
            INSERT INTO {_EVENT_TABLE} (
                event_id, alert_id, user_id, event_type, from_status, to_status,
                price, recorded_at, message
            ) VALUES (
                :event_id, :alert_id, :user_id, :event_type, :from_status, :to_status,
                :price, :recorded_at, :message
            )
        """
        self._execute(sql, payload)

    def list_events(
        self, user_id: str, alert_id: str
    ) -> tuple[AlertLifecycleEvent, ...]:
        sql = f"""
            SELECT event_id, alert_id, event_type, from_status, to_status,
                   price, recorded_at, message
            FROM {_EVENT_TABLE}
            WHERE user_id = :user_id AND alert_id = :alert_id
            ORDER BY recorded_at ASC
        """
        rows = self._fetchall(sql, {"user_id": user_id, "alert_id": alert_id})
        return tuple(event_from_row(self._event_row_dict(row)) for row in rows)

    def _execute(self, sql: str, params: dict[str, Any]) -> None:
        con = self._client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, params)
            con.commit()
        finally:
            con.close()

    def _fetchone(self, sql: str, params: dict[str, Any]) -> tuple | None:
        rows = self._fetchall(sql, params)
        return rows[0] if rows else None

    def _fetchall(self, sql: str, params: dict[str, Any]) -> list[tuple]:
        con = self._client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, params)
            return list(cur.fetchall())
        finally:
            con.close()

    @staticmethod
    def _alert_row_dict(row: tuple) -> dict[str, Any]:
        (
            alert_id,
            user_id,
            symbol,
            setup_name,
            created_at,
            signal_date,
            entry_price,
            stop_price,
            target_price,
            entry_is_stop,
            status,
            expires_at,
            triggered_at,
            exit_at,
            exit_price,
            outcome_return_pct,
            risk_gate_action,
            risk_gate_reasons,
            historical_win_rate,
            historical_profit_factor,
            historical_total_trades,
            market_regime,
            volume_ratio,
            rs_percentile,
        ) = row
        return {
            "alert_id": alert_id,
            "user_id": user_id,
            "symbol": symbol,
            "setup_name": setup_name,
            "created_at": created_at,
            "signal_date": signal_date,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "entry_is_stop": bool(entry_is_stop),
            "status": status,
            "expires_at": expires_at,
            "triggered_at": triggered_at,
            "exit_at": exit_at,
            "exit_price": exit_price,
            "outcome_return_pct": outcome_return_pct,
            "risk_gate_action": risk_gate_action,
            "risk_gate_reasons": risk_gate_reasons,
            "historical_win_rate": historical_win_rate,
            "historical_profit_factor": historical_profit_factor,
            "historical_total_trades": historical_total_trades,
            "market_regime": market_regime,
            "volume_ratio": volume_ratio,
            "rs_percentile": rs_percentile,
        }

    @staticmethod
    def _event_row_dict(row: tuple) -> dict[str, Any]:
        (
            event_id,
            alert_id,
            event_type,
            from_status,
            to_status,
            price,
            recorded_at,
            message,
        ) = row
        return {
            "event_id": event_id,
            "alert_id": alert_id,
            "event_type": event_type,
            "from_status": from_status,
            "to_status": to_status,
            "price": price,
            "recorded_at": recorded_at,
            "message": message,
        }
