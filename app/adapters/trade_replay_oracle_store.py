from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

import oracledb

from app.models.trade_replay_models import TradeReplayEvent, TradeReplayWorkflow
from app.services.trade_replay_service import PlanSaveResult, TradePlanRecord

_PLAN_TABLE = "TRADE_PLANS"
_EVENT_TABLE = "TRADE_REPLAY_EVENTS"


class OracleTradeReplayStore:
    def __init__(self, client: oracledb.ConnectionPool) -> None:
        self._client = client

    def ensure_schema(self) -> None:
        statements = [
            f"""
            CREATE TABLE {_PLAN_TABLE} (
                plan_id VARCHAR2(64) PRIMARY KEY,
                symbol VARCHAR2(16) NOT NULL,
                workflow VARCHAR2(32) NOT NULL,
                plan_date DATE NOT NULL,
                generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                source VARCHAR2(24) NOT NULL,
                source_freshness_label VARCHAR2(255),
                signature VARCHAR2(64) NOT NULL,
                levels_json CLOB NOT NULL,
                risk_json CLOB,
                plan_json CLOB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT uq_trade_plans_signature UNIQUE (symbol, plan_date, workflow, signature)
            )
            """,
            f"""
            CREATE INDEX idx_trade_plans_lookup
                ON {_PLAN_TABLE} (symbol, plan_date, workflow, generated_at)
            """,
            f"""
            CREATE TABLE {_EVENT_TABLE} (
                event_id VARCHAR2(64) PRIMARY KEY,
                plan_id VARCHAR2(64),
                symbol VARCHAR2(16) NOT NULL,
                event_date DATE NOT NULL,
                workflow VARCHAR2(32) NOT NULL,
                event_type VARCHAR2(64) NOT NULL,
                event_time TIMESTAMP WITH TIME ZONE NOT NULL,
                level_price NUMBER,
                observed_price NUMBER,
                message VARCHAR2(1000) NOT NULL,
                severity VARCHAR2(24) NOT NULL,
                actionability VARCHAR2(24) NOT NULL,
                source VARCHAR2(24) NOT NULL,
                source_freshness_label VARCHAR2(255),
                dedupe_key VARCHAR2(255) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT fk_trade_replay_plan FOREIGN KEY (plan_id)
                    REFERENCES {_PLAN_TABLE}(plan_id),
                CONSTRAINT uq_trade_replay_event UNIQUE (
                    symbol, event_date, workflow, event_type, dedupe_key
                )
            )
            """,
            f"""
            CREATE INDEX idx_trade_replay_events_lookup
                ON {_EVENT_TABLE} (symbol, event_date, workflow, event_time)
            """,
        ]
        con = self._client.acquire()
        try:
            cur = con.cursor()
            for statement in statements:
                try:
                    cur.execute(statement)
                except oracledb.DatabaseError as exc:
                    if not _is_name_already_used(exc):
                        raise
            self._ensure_column(cur, _PLAN_TABLE, "RISK_JSON", "CLOB")
            self._ensure_unique_constraint(
                cur,
                table_name=_PLAN_TABLE,
                constraint_name="UQ_TRADE_PLANS_SIGNATURE",
                columns=("SYMBOL", "PLAN_DATE", "WORKFLOW", "SIGNATURE"),
            )
            self._ensure_unique_constraint(
                cur,
                table_name=_EVENT_TABLE,
                constraint_name="UQ_TRADE_REPLAY_EVENT",
                columns=("SYMBOL", "EVENT_DATE", "WORKFLOW", "EVENT_TYPE", "DEDUPE_KEY"),
            )
            con.commit()
        finally:
            con.close()

    def _ensure_column(
        self,
        cur: oracledb.Cursor,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        cur.execute(
            """
            SELECT 1
            FROM user_tab_cols
            WHERE table_name = :table_name
              AND column_name = :column_name
            """,
            {"table_name": table_name.upper(), "column_name": column_name.upper()},
        )
        if cur.fetchone() is not None:
            return
        cur.execute(f"ALTER TABLE {table_name} ADD ({column_name} {column_type})")

    def _ensure_unique_constraint(
        self,
        cur: oracledb.Cursor,
        *,
        table_name: str,
        constraint_name: str,
        columns: tuple[str, ...],
    ) -> None:
        cur.execute(
            """
            SELECT 1
            FROM user_constraints
            WHERE table_name = :table_name
              AND constraint_name = :constraint_name
              AND constraint_type = 'U'
            """,
            {
                "table_name": table_name.upper(),
                "constraint_name": constraint_name.upper(),
            },
        )
        if cur.fetchone() is not None:
            return
        column_sql = ", ".join(columns)
        try:
            cur.execute(
                f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} UNIQUE ({column_sql})"
            )
        except oracledb.DatabaseError as exc:
            if not _is_name_already_used(exc):
                raise

    def save_plan_if_changed(self, plan: TradePlanRecord) -> PlanSaveResult:
        existing = self.latest_plan(
            symbol=plan.symbol,
            workflow=plan.workflow,
            plan_date=plan.plan_date,
        )
        if existing is not None and existing.signature == plan.signature:
            return PlanSaveResult(plan=existing, created=False)

        payload = {
            "plan_id": plan.plan_id,
            "symbol": plan.symbol,
            "workflow": plan.workflow,
            "plan_date": plan.plan_date,
            "generated_at": plan.generated_at,
            "source": plan.source,
            "source_freshness_label": plan.source_freshness_label,
            "signature": plan.signature,
            "levels_json": json.dumps(plan.levels, sort_keys=True),
            "risk_json": json.dumps(plan.payload.get("risk", {}), sort_keys=True),
            "plan_json": json.dumps(plan.payload, sort_keys=True),
        }
        sql = f"""
            INSERT INTO {_PLAN_TABLE} (
                plan_id, symbol, workflow, plan_date, generated_at, source,
                source_freshness_label, signature, levels_json, risk_json, plan_json
            ) VALUES (
                :plan_id, :symbol, :workflow, :plan_date, :generated_at, :source,
                :source_freshness_label, :signature, :levels_json, :risk_json, :plan_json
            )
        """
        con = self._client.acquire()
        try:
            cur = con.cursor()
            try:
                cur.execute(sql, payload)
                con.commit()
                return PlanSaveResult(plan=plan, created=True)
            except oracledb.IntegrityError:
                con.rollback()
                existing_same = self._find_plan_by_signature(
                    symbol=plan.symbol,
                    workflow=plan.workflow,
                    plan_date=plan.plan_date,
                    signature=plan.signature,
                )
                if existing_same is None:
                    raise
                return PlanSaveResult(plan=existing_same, created=False)
        finally:
            con.close()

    def latest_plan(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        plan_date: date,
    ) -> TradePlanRecord | None:
        sql = f"""
            SELECT plan_id, symbol, workflow, plan_date, generated_at, source,
                   source_freshness_label, signature, levels_json, plan_json, created_at
            FROM {_PLAN_TABLE}
            WHERE symbol = :symbol
              AND workflow = :workflow
              AND plan_date = :plan_date
            ORDER BY generated_at DESC, created_at DESC
            FETCH FIRST 1 ROWS ONLY
        """
        row = self._fetchone(
            sql,
            {
                "symbol": symbol.upper(),
                "workflow": workflow,
                "plan_date": plan_date,
            },
        )
        return _plan_from_row(row) if row is not None else None

    def append_events(self, events: list[TradeReplayEvent]) -> int:
        if not events:
            return 0
        sql = f"""
            INSERT INTO {_EVENT_TABLE} (
                event_id, plan_id, symbol, event_date, workflow, event_type,
                event_time, level_price, observed_price, message, severity,
                actionability, source, source_freshness_label, dedupe_key, created_at
            ) VALUES (
                :event_id, :plan_id, :symbol, :event_date, :workflow, :event_type,
                :event_time, :level_price, :observed_price, :message, :severity,
                :actionability, :source, :source_freshness_label, :dedupe_key, :created_at
            )
        """
        created = 0
        con = self._client.acquire()
        try:
            cur = con.cursor()
            for event in events:
                try:
                    cur.execute(sql, _event_to_row(event))
                    created += 1
                except oracledb.IntegrityError:
                    continue
            con.commit()
        finally:
            con.close()
        return created

    def list_events(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        event_date: date,
    ) -> list[TradeReplayEvent]:
        sql = f"""
            SELECT event_id, plan_id, symbol, event_date, workflow, event_type,
                   event_time, level_price, observed_price, message, severity,
                   actionability, source, source_freshness_label, dedupe_key, created_at
            FROM {_EVENT_TABLE}
            WHERE symbol = :symbol
              AND workflow = :workflow
              AND event_date = :event_date
            ORDER BY event_time ASC, created_at ASC
        """
        rows = self._fetchall(
            sql,
            {
                "symbol": symbol.upper(),
                "workflow": workflow,
                "event_date": event_date,
            },
        )
        return [_event_from_row(row) for row in rows]

    def _find_plan_by_signature(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        plan_date: date,
        signature: str,
    ) -> TradePlanRecord | None:
        sql = f"""
            SELECT plan_id, symbol, workflow, plan_date, generated_at, source,
                   source_freshness_label, signature, levels_json, plan_json, created_at
            FROM {_PLAN_TABLE}
            WHERE symbol = :symbol
              AND workflow = :workflow
              AND plan_date = :plan_date
              AND signature = :signature
            FETCH FIRST 1 ROWS ONLY
        """
        row = self._fetchone(
            sql,
            {
                "symbol": symbol.upper(),
                "workflow": workflow,
                "plan_date": plan_date,
                "signature": signature,
            },
        )
        return _plan_from_row(row) if row is not None else None

    def _fetchone(self, sql: str, params: dict[str, Any]) -> tuple[Any, ...] | None:
        rows = self._fetchall(sql, params)
        return rows[0] if rows else None

    def _fetchall(self, sql: str, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        con = self._client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, params)
            return list(cur.fetchall())
        finally:
            con.close()


def _plan_from_row(row: tuple[Any, ...]) -> TradePlanRecord:
    (
        plan_id,
        symbol,
        workflow,
        plan_date,
        generated_at,
        source,
        source_freshness_label,
        signature,
        levels_json,
        plan_json,
        created_at,
    ) = row
    return TradePlanRecord(
        plan_id=str(plan_id),
        symbol=str(symbol),
        workflow=str(workflow),  # type: ignore[arg-type]
        plan_date=plan_date,
        generated_at=_aware_datetime(generated_at),
        source=str(source),  # type: ignore[arg-type]
        source_freshness_label=str(source_freshness_label or ""),
        signature=str(signature),
        levels=json.loads(_lob_text(levels_json) or "{}"),
        payload=json.loads(_lob_text(plan_json) or "{}"),
        created_at=_aware_datetime(created_at),
    )


def _event_to_row(event: TradeReplayEvent) -> dict[str, Any]:
    return {
        "event_id": event.id,
        "plan_id": event.plan_id,
        "symbol": event.symbol,
        "event_date": event.event_date,
        "workflow": event.workflow,
        "event_type": event.event_type,
        "event_time": event.event_time,
        "level_price": event.level_price,
        "observed_price": event.observed_price,
        "message": event.message,
        "severity": event.severity,
        "actionability": event.actionability,
        "source": event.source,
        "source_freshness_label": event.source_freshness_label,
        "dedupe_key": event.dedupe_key,
        "created_at": event.created_at or datetime.now(timezone.utc),
    }


def _event_from_row(row: tuple[Any, ...]) -> TradeReplayEvent:
    (
        event_id,
        plan_id,
        symbol,
        event_date,
        workflow,
        event_type,
        event_time,
        level_price,
        observed_price,
        message,
        severity,
        actionability,
        source,
        source_freshness_label,
        dedupe_key,
        created_at,
    ) = row
    return TradeReplayEvent(
        id=str(event_id),
        plan_id=str(plan_id) if plan_id else None,
        symbol=str(symbol),
        event_date=event_date,
        workflow=str(workflow),  # type: ignore[arg-type]
        event_type=str(event_type),
        event_time=_aware_datetime(event_time),
        level_price=float(level_price) if level_price is not None else None,
        observed_price=float(observed_price) if observed_price is not None else None,
        message=str(message),
        severity=str(severity),  # type: ignore[arg-type]
        actionability=str(actionability),  # type: ignore[arg-type]
        source=str(source),  # type: ignore[arg-type]
        source_freshness_label=(
            str(source_freshness_label) if source_freshness_label else None
        ),
        dedupe_key=str(dedupe_key),
        created_at=_aware_datetime(created_at),
    )


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _lob_text(value: Any) -> str | None:
    if value is None:
        return None
    read = getattr(value, "read", None)
    if callable(read):
        return str(read())
    return str(value)


def _is_name_already_used(exc: oracledb.DatabaseError) -> bool:
    error = exc.args[0] if exc.args else None
    code = getattr(error, "code", None)
    message = str(exc)
    return code == 955 or "ORA-00955" in message
