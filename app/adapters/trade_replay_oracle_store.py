from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any

import oracledb

from app.models.trade_replay_models import TradeReplayEvent, TradeReplayWorkflow
from app.services.trade_replay_service import (
    MissedMoveRecord,
    PlanSaveResult,
    TradePlanRecord,
)

_PLAN_TABLE = "TRADE_PLANS"
_EVENT_TABLE = "TRADE_REPLAY_EVENTS"
_MISSED_MOVE_TABLE = "MISSED_MOVES"

logger = logging.getLogger(__name__)


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
            f"""
            CREATE TABLE {_MISSED_MOVE_TABLE} (
                missed_move_id VARCHAR2(64) PRIMARY KEY,
                symbol VARCHAR2(16) NOT NULL,
                workflow VARCHAR2(32) NOT NULL,
                event_date DATE NOT NULL,
                setup_type VARCHAR2(255) NOT NULL,
                trigger_price NUMBER,
                outcome VARCHAR2(32) NOT NULL,
                max_move_after_trigger_pct NUMBER,
                setup_quality_score NUMBER,
                source VARCHAR2(24) NOT NULL,
                source_freshness_label VARCHAR2(255),
                trigger_event_id VARCHAR2(64) NOT NULL,
                terminal_event_id VARCHAR2(64) NOT NULL,
                replay_events_json CLOB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT uq_missed_move_story UNIQUE (
                    symbol, event_date, workflow, trigger_event_id, terminal_event_id
                )
            )
            """,
            f"""
            CREATE INDEX idx_missed_moves_lookup
                ON {_MISSED_MOVE_TABLE} (symbol, event_date, workflow)
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
            self._ensure_unique_constraint(
                cur,
                table_name=_MISSED_MOVE_TABLE,
                constraint_name="UQ_MISSED_MOVE_STORY",
                columns=(
                    "SYMBOL",
                    "EVENT_DATE",
                    "WORKFLOW",
                    "TRIGGER_EVENT_ID",
                    "TERMINAL_EVENT_ID",
                ),
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

    def save_missed_moves(self, missed_moves: list[MissedMoveRecord]) -> int:
        if not missed_moves:
            logger.info("Missed moves save skipped: no completed missed moves to persist")
            return 0
        sql = f"""
            INSERT INTO {_MISSED_MOVE_TABLE} (
                missed_move_id, symbol, workflow, event_date, setup_type,
                trigger_price, outcome, max_move_after_trigger_pct,
                setup_quality_score, source, source_freshness_label,
                trigger_event_id, terminal_event_id, replay_events_json, created_at
            ) VALUES (
                :missed_move_id, :symbol, :workflow, :event_date, :setup_type,
                :trigger_price, :outcome, :max_move_after_trigger_pct,
                :setup_quality_score, :source, :source_freshness_label,
                :trigger_event_id, :terminal_event_id, :replay_events_json, :created_at
            )
        """
        created = 0
        con = self._client.acquire()
        try:
            cur = con.cursor()
            for missed_move in missed_moves:
                try:
                    cur.execute(sql, _missed_move_to_row(missed_move))
                    created += 1
                except oracledb.IntegrityError:
                    continue
            con.commit()
        finally:
            con.close()
        logger.info(
            "Missed moves save completed: attempted=%s created=%s symbols=%s dates=%s setup_types=%s",
            len(missed_moves),
            created,
            _count_by_attr(missed_moves, "symbol"),
            _count_by_event_date(missed_moves),
            _count_by_attr(missed_moves, "setup_type"),
        )
        return created

    def list_missed_moves(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        start_date: date,
        end_date: date,
    ) -> list[MissedMoveRecord]:
        sql = f"""
            SELECT missed_move_id, symbol, workflow, event_date, setup_type,
                   trigger_price, outcome, max_move_after_trigger_pct,
                   setup_quality_score, source, source_freshness_label,
                   trigger_event_id, terminal_event_id, replay_events_json, created_at
            FROM {_MISSED_MOVE_TABLE}
            WHERE symbol = :symbol
              AND workflow = :workflow
              AND event_date BETWEEN :start_date AND :end_date
            ORDER BY event_date DESC, created_at DESC
        """
        rows = self._fetchall(
            sql,
            {
                "symbol": symbol.upper(),
                "workflow": workflow,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        records = [_missed_move_from_row(row) for row in rows]
        logger.info(
            (
                "Missed moves SQL result: symbol=%s workflow=%s start_date=%s "
                "end_date=%s returned_rows=%s latest_date=%s count_by_date=%s "
                "count_by_setup_type=%s"
            ),
            symbol.upper(),
            workflow,
            start_date.isoformat(),
            end_date.isoformat(),
            len(records),
            max((record.event_date for record in records), default=None),
            _count_by_event_date(records),
            _count_by_attr(records, "setup_type"),
        )
        self._log_missed_move_table_snapshot(symbol=symbol.upper(), workflow=workflow)
        return records

    def _log_missed_move_table_snapshot(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
    ) -> None:
        total_sql = f"""
            SELECT COUNT(*), MAX(event_date)
            FROM {_MISSED_MOVE_TABLE}
            WHERE symbol = :symbol
              AND workflow = :workflow
        """
        by_symbol_sql = f"""
            SELECT symbol, COUNT(*)
            FROM {_MISSED_MOVE_TABLE}
            WHERE workflow = :workflow
            GROUP BY symbol
            ORDER BY COUNT(*) DESC
            FETCH FIRST 20 ROWS ONLY
        """
        by_date_sql = f"""
            SELECT event_date, COUNT(*)
            FROM {_MISSED_MOVE_TABLE}
            WHERE symbol = :symbol
              AND workflow = :workflow
            GROUP BY event_date
            ORDER BY event_date DESC
            FETCH FIRST 10 ROWS ONLY
        """
        by_setup_sql = f"""
            SELECT setup_type, COUNT(*)
            FROM {_MISSED_MOVE_TABLE}
            WHERE symbol = :symbol
              AND workflow = :workflow
            GROUP BY setup_type
            ORDER BY COUNT(*) DESC
        """
        try:
            total_row = self._fetchone(
                total_sql,
                {"symbol": symbol, "workflow": workflow},
            )
            by_symbol = self._fetchall(by_symbol_sql, {"workflow": workflow})
            by_date = self._fetchall(
                by_date_sql,
                {"symbol": symbol, "workflow": workflow},
            )
            by_setup = self._fetchall(
                by_setup_sql,
                {"symbol": symbol, "workflow": workflow},
            )
        except Exception:
            logger.warning(
                "Missed moves table snapshot failed for %s/%s",
                symbol,
                workflow,
                exc_info=True,
            )
            return
        logger.info(
            (
                "Missed moves table snapshot: symbol=%s workflow=%s "
                "symbol_total=%s latest_symbol_date=%s count_by_symbol=%s "
                "count_by_trading_date=%s count_by_setup_type=%s"
            ),
            symbol,
            workflow,
            total_row[0] if total_row else 0,
            total_row[1] if total_row else None,
            {str(row[0]): int(row[1]) for row in by_symbol},
            {str(row[0]): int(row[1]) for row in by_date},
            {str(row[0]): int(row[1]) for row in by_setup},
        )

    def get_missed_move(self, missed_move_id: str) -> MissedMoveRecord | None:
        sql = f"""
            SELECT missed_move_id, symbol, workflow, event_date, setup_type,
                   trigger_price, outcome, max_move_after_trigger_pct,
                   setup_quality_score, source, source_freshness_label,
                   trigger_event_id, terminal_event_id, replay_events_json, created_at
            FROM {_MISSED_MOVE_TABLE}
            WHERE missed_move_id = :missed_move_id
            FETCH FIRST 1 ROWS ONLY
        """
        row = self._fetchone(sql, {"missed_move_id": missed_move_id})
        return _missed_move_from_row(row) if row is not None else None

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


def _missed_move_to_row(missed_move: MissedMoveRecord) -> dict[str, Any]:
    return {
        "missed_move_id": missed_move.missed_move_id,
        "symbol": missed_move.symbol,
        "workflow": missed_move.workflow,
        "event_date": missed_move.event_date,
        "setup_type": missed_move.setup_type,
        "trigger_price": missed_move.trigger_price,
        "outcome": missed_move.outcome,
        "max_move_after_trigger_pct": missed_move.max_move_after_trigger_pct,
        "setup_quality_score": missed_move.setup_quality_score,
        "source": missed_move.source,
        "source_freshness_label": missed_move.source_freshness_label,
        "trigger_event_id": missed_move.trigger_event_id,
        "terminal_event_id": missed_move.terminal_event_id,
        "replay_events_json": json.dumps(
            [
                event.model_dump(mode="json", by_alias=True)
                for event in missed_move.replay_events
            ],
            sort_keys=True,
        ),
        "created_at": missed_move.created_at or datetime.now(timezone.utc),
    }


def _count_by_attr(records: list[MissedMoveRecord], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(getattr(record, attr))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_by_event_date(records: list[MissedMoveRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = record.event_date.isoformat()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _missed_move_from_row(row: tuple[Any, ...]) -> MissedMoveRecord:
    (
        missed_move_id,
        symbol,
        workflow,
        event_date,
        setup_type,
        trigger_price,
        outcome,
        max_move_after_trigger_pct,
        setup_quality_score,
        source,
        source_freshness_label,
        trigger_event_id,
        terminal_event_id,
        replay_events_json,
        created_at,
    ) = row
    events_payload = json.loads(_lob_text(replay_events_json) or "[]")
    return MissedMoveRecord(
        missed_move_id=str(missed_move_id),
        symbol=str(symbol),
        workflow=str(workflow),  # type: ignore[arg-type]
        event_date=event_date,
        setup_type=str(setup_type),
        trigger_price=float(trigger_price) if trigger_price is not None else None,
        outcome=str(outcome),  # type: ignore[arg-type]
        max_move_after_trigger_pct=(
            float(max_move_after_trigger_pct)
            if max_move_after_trigger_pct is not None
            else None
        ),
        setup_quality_score=(
            float(setup_quality_score) if setup_quality_score is not None else None
        ),
        source=str(source),  # type: ignore[arg-type]
        source_freshness_label=(
            str(source_freshness_label) if source_freshness_label else None
        ),
        trigger_event_id=str(trigger_event_id),
        terminal_event_id=str(terminal_event_id),
        replay_events=[
            TradeReplayEvent.model_validate(item) for item in events_payload
        ],
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
