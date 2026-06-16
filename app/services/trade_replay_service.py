from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Protocol

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.intraday_trading_bias_engine import EASTERN
from app.models.day_trade_backtest_models import DayTradeDirectionMode
from app.models.trade_replay_models import (
    MissedMoveOutcome,
    MissedMoveSummaryRow,
    MissedMovesRange,
    MissedMovesSort,
    MissedMovesSummaryResponse,
    TradeReplayEvent,
    TradeReplayRefreshResponse,
    TradeReplayResponse,
    TradeReplaySource,
    TradeReplayWorkflow,
)
from app.services.day_trade_direction import day_trade_direction_allowed
from app.services.intraday_trading_bias_service import build_intraday_trading_bias
from app.services.trader_playbook_service import build_trader_playbook
from models.prediction_service import LoadedModel

SOURCE_FRESHNESS_DELAYED = "Educational / delayed — not for live execution."
TARGET_2_OFFSET_MICROSECONDS = 1
DEFAULT_LIVE_DAY_TRADE_DIRECTION_MODE: DayTradeDirectionMode = "long_only"
COMPLETED_MISSED_MOVE_OUTCOMES = {
    "target_1_hit",
    "target_2_hit",
    "target_hit",
    "setup_extended",
    "setup_invalidated",
    "stop_hit",
    "rr_degraded",
    "entry_missed",
}
MARKET_OPEN = time(9, 30)
OPENING_RANGE_END = time(10, 0)
MARKET_CLOSE = time(16, 0)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradePlanRecord:
    plan_id: str
    symbol: str
    workflow: TradeReplayWorkflow
    plan_date: date
    generated_at: datetime
    source: TradeReplaySource
    source_freshness_label: str
    signature: str
    levels: dict[str, float | str | None]
    payload: dict[str, Any]
    created_at: datetime | None = None


@dataclass(frozen=True)
class ReplayBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


@dataclass(frozen=True)
class PlanSaveResult:
    plan: TradePlanRecord
    created: bool


@dataclass(frozen=True)
class MissedMoveRecord:
    missed_move_id: str
    symbol: str
    workflow: TradeReplayWorkflow
    event_date: date
    setup_type: str
    direction: str
    trigger_time: datetime | None
    trigger_price: float | None
    outcome: MissedMoveOutcome
    max_move_after_trigger_pct: float | None
    setup_quality_score: float | None
    entry: float | None
    stop: float | None
    target_1: float | None
    target_2: float | None
    open_range_high: float | None
    open_range_low: float | None
    vwap: float | None
    event_count: int
    source: TradeReplaySource
    source_freshness_label: str | None
    trigger_event_id: str
    terminal_event_id: str
    replay_events: list[TradeReplayEvent]
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TradeReplayStore(Protocol):
    def save_plan_if_changed(self, plan: TradePlanRecord) -> PlanSaveResult: ...

    def latest_plan(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        plan_date: date,
    ) -> TradePlanRecord | None: ...

    def append_events(self, events: list[TradeReplayEvent]) -> int: ...

    def list_events(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        event_date: date,
    ) -> list[TradeReplayEvent]: ...

    def save_missed_moves(self, missed_moves: list[MissedMoveRecord]) -> int: ...

    def list_missed_moves(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        start_date: date,
        end_date: date,
    ) -> list[MissedMoveRecord]: ...

    def get_missed_move(self, missed_move_id: str) -> MissedMoveRecord | None: ...


class InMemoryTradeReplayStore:
    """Small test/local store with the same idempotency semantics as Oracle."""

    def __init__(self) -> None:
        self.plans: list[TradePlanRecord] = []
        self.events: list[TradeReplayEvent] = []
        self.missed_moves: list[MissedMoveRecord] = []
        self._event_keys: set[tuple[str, date, str, str, str]] = set()
        self._missed_move_ids: set[str] = set()
        self._missed_move_keys: dict[tuple[str, date, str, str, str], str] = {}

    def save_plan_if_changed(self, plan: TradePlanRecord) -> PlanSaveResult:
        existing = self.latest_plan(
            symbol=plan.symbol,
            workflow=plan.workflow,
            plan_date=plan.plan_date,
        )
        if existing is not None and existing.signature == plan.signature:
            return PlanSaveResult(plan=existing, created=False)
        self.plans.append(plan)
        return PlanSaveResult(plan=plan, created=True)

    def latest_plan(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        plan_date: date,
    ) -> TradePlanRecord | None:
        matches = [
            plan
            for plan in self.plans
            if plan.symbol == symbol.upper()
            and plan.workflow == workflow
            and plan.plan_date == plan_date
        ]
        return max(matches, key=lambda plan: plan.generated_at) if matches else None

    def append_events(self, events: list[TradeReplayEvent]) -> int:
        created = 0
        for event in events:
            key = (
                event.symbol.upper(),
                event.event_date,
                event.workflow,
                event.event_type,
                event.dedupe_key,
            )
            if key in self._event_keys:
                continue
            self._event_keys.add(key)
            self.events.append(event)
            created += 1
        return created

    def list_events(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        event_date: date,
    ) -> list[TradeReplayEvent]:
        return sorted(
            [
                event
                for event in self.events
                if event.symbol == symbol.upper()
                and event.workflow == workflow
                and event.event_date == event_date
            ],
            key=lambda event: event.event_time,
        )

    def save_missed_moves(self, missed_moves: list[MissedMoveRecord]) -> int:
        created = 0
        for missed_move in missed_moves:
            key = _missed_move_natural_key(missed_move)
            existing_id = self._missed_move_keys.get(key)
            if existing_id is not None:
                self.missed_moves = [
                    replace(
                        missed_move,
                        missed_move_id=existing_id,
                        created_at=item.created_at or missed_move.created_at,
                    )
                    if item.missed_move_id == existing_id
                    else item
                    for item in self.missed_moves
                ]
                continue
            if missed_move.missed_move_id in self._missed_move_ids:
                continue
            self._missed_move_keys[key] = missed_move.missed_move_id
            self._missed_move_ids.add(missed_move.missed_move_id)
            self.missed_moves.append(missed_move)
            created += 1
        return created

    def list_missed_moves(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        start_date: date,
        end_date: date,
    ) -> list[MissedMoveRecord]:
        symbol_upper = symbol.upper()
        return [
            missed_move
            for missed_move in self.missed_moves
            if missed_move.symbol == symbol_upper
            and missed_move.workflow == workflow
            and start_date <= missed_move.event_date <= end_date
        ]

    def get_missed_move(self, missed_move_id: str) -> MissedMoveRecord | None:
        return next(
            (
                missed_move
                for missed_move in self.missed_moves
                if missed_move.missed_move_id == missed_move_id
            ),
            None,
        )


@dataclass
class TradeReplayService:
    store: TradeReplayStore
    yfinance_adapter: YFinanceAdapter
    loaded_model: LoadedModel | None = None
    pattern_analysis_service: Any = None
    research_events_service: Any = None
    now: datetime | None = None

    def get_replay(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        event_date: date,
        missed_move_id: str | None = None,
        direction_mode: DayTradeDirectionMode = DEFAULT_LIVE_DAY_TRADE_DIRECTION_MODE,
    ) -> TradeReplayResponse:
        if missed_move_id:
            missed_move = self.store.get_missed_move(missed_move_id)
            if missed_move is not None:
                return TradeReplayResponse(
                    symbol=missed_move.symbol,
                    date=missed_move.event_date,
                    workflow=missed_move.workflow,
                    direction_mode=direction_mode if workflow == "day_trade" else None,
                    source=missed_move.source,
                    source_freshness_label=missed_move.source_freshness_label,
                    events=(
                        _filter_events_for_direction(
                            _sort_story_events(missed_move.replay_events),
                            direction_mode,
                        )
                        if missed_move.workflow == "day_trade"
                        else _sort_story_events(missed_move.replay_events)
                    ),
                )
            return TradeReplayResponse(
                symbol=symbol.strip().upper(),
                date=event_date,
                workflow=workflow,
                direction_mode=direction_mode if workflow == "day_trade" else None,
                source="historical",
                source_freshness_label=SOURCE_FRESHNESS_DELAYED,
                events=[],
            )

        symbol_upper = symbol.strip().upper()
        if workflow == "day_trade":
            plan = self._build_day_trade_plan(
                symbol_upper,
                event_date,
                direction_mode=direction_mode,
            )
            bars = self._load_bars(
                symbol_upper,
                workflow,
                event_date,
                plan.generated_at,
            )
            events = generate_replay_events(
                plan,
                bars,
                direction_mode=direction_mode,
            )
            if not events and not bars:
                events = _filter_events_for_direction(
                    _sort_story_events(
                        self.store.list_events(
                            symbol=symbol_upper,
                            workflow=workflow,
                            event_date=event_date,
                        )
                    ),
                    direction_mode,
                )
            return TradeReplayResponse(
                symbol=symbol_upper,
                date=event_date,
                workflow=workflow,
                direction_mode=direction_mode,
                source=plan.source,
                source_freshness_label=plan.source_freshness_label,
                events=events,
            )

        plan = self.store.latest_plan(
            symbol=symbol_upper,
            workflow=workflow,
            plan_date=event_date,
        )
        events = self.store.list_events(
            symbol=symbol_upper,
            workflow=workflow,
            event_date=event_date,
        )
        events = _sort_story_events(events)
        return TradeReplayResponse(
            symbol=symbol_upper,
            date=event_date,
            workflow=workflow,
            direction_mode=None,
            source=plan.source if plan is not None else "delayed",
            source_freshness_label=(
                plan.source_freshness_label
                if plan is not None
                else SOURCE_FRESHNESS_DELAYED
            ),
            events=events,
        )

    def refresh(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        event_date: date,
        direction_mode: DayTradeDirectionMode = DEFAULT_LIVE_DAY_TRADE_DIRECTION_MODE,
    ) -> TradeReplayRefreshResponse:
        symbol_upper = symbol.strip().upper()
        plan = self._build_plan(
            symbol_upper,
            workflow,
            event_date,
            direction_mode=direction_mode,
        )
        saved = self.store.save_plan_if_changed(plan)
        bars = self._load_bars(
            symbol_upper,
            workflow,
            event_date,
            saved.plan.generated_at,
        )
        events = generate_replay_events(
            saved.plan,
            bars,
            direction_mode=direction_mode if workflow == "day_trade" else "long_and_short",
        )
        created = self.store.append_events(events)
        missed_moves = build_completed_missed_moves(saved.plan, events, bars)
        missed_moves_created = self.store.save_missed_moves(missed_moves)
        logger.info(
            (
                "Trade replay refresh trace: symbol=%s workflow=%s date=%s "
                "direction_mode=%s bars=%s events=%s events_created=%s "
                "completed_missed_moves=%s missed_moves_created=%s"
            ),
            symbol_upper,
            workflow,
            event_date.isoformat(),
            direction_mode if workflow == "day_trade" else None,
            len(bars),
            len(events),
            created,
            len(missed_moves),
            missed_moves_created,
        )
        return TradeReplayRefreshResponse(
            success=True,
            symbol=symbol_upper,
            date=event_date,
            workflow=workflow,
            direction_mode=direction_mode if workflow == "day_trade" else None,
            plan_id=saved.plan.plan_id,
            plan_created=saved.created,
            events_created=created,
            source=saved.plan.source,
            source_freshness_label=saved.plan.source_freshness_label,
        )

    def list_missed_moves(
        self,
        *,
        symbol: str,
        workflow: TradeReplayWorkflow,
        range_: MissedMovesRange,
        sort: MissedMovesSort,
    ) -> MissedMovesSummaryResponse:
        symbol_upper = symbol.strip().upper()
        end_date = _current_trading_date(self.now or datetime.now(timezone.utc))
        trading_dates = _trading_day_window(end_date, 1 if range_ == "today" else 5)
        start_date = min(trading_dates)
        logger.info(
            (
                "Missed moves query trace: symbol=%s workflow=%s range=%s sort=%s "
                "timezone=%s start_date=%s end_date=%s trading_days=%s"
            ),
            symbol_upper,
            workflow,
            range_,
            sort,
            str(EASTERN),
            start_date.isoformat(),
            end_date.isoformat(),
            ",".join(day.isoformat() for day in sorted(trading_dates)),
        )
        if workflow == "day_trade" and end_date in trading_dates:
            try:
                self.refresh(
                    symbol=symbol_upper,
                    workflow=workflow,
                    event_date=end_date,
                    direction_mode="long_and_short",
                )
            except Exception:
                logger.warning(
                    (
                        "Missed moves current-day materialization failed: "
                        "symbol=%s workflow=%s date=%s"
                    ),
                    symbol_upper,
                    workflow,
                    end_date.isoformat(),
                    exc_info=True,
                )
        rows = self.store.list_missed_moves(
            symbol=symbol_upper,
            workflow=workflow,
            start_date=start_date,
            end_date=end_date,
        )
        rows = [
            row
            for row in rows
            if row.event_date in trading_dates and _is_trading_day(row.event_date)
        ]
        rows = _sort_missed_moves(rows, sort)
        logger.info(
            (
                "Missed moves query result: symbol=%s workflow=%s range=%s "
                "returned_rows=%s count_by_date=%s count_by_setup_type=%s"
            ),
            symbol_upper,
            workflow,
            range_,
            len(rows),
            _count_by_date(rows),
            _count_by_setup_type(rows),
        )
        return MissedMovesSummaryResponse(
            range=range_,
            sort=sort,
            source="historical",
            source_freshness_label=SOURCE_FRESHNESS_DELAYED,
            rows=[_summary_row(record) for record in rows],
        )

    def _build_plan(
        self,
        symbol: str,
        workflow: TradeReplayWorkflow,
        event_date: date,
        direction_mode: DayTradeDirectionMode = DEFAULT_LIVE_DAY_TRADE_DIRECTION_MODE,
    ) -> TradePlanRecord:
        if workflow == "day_trade":
            return self._build_day_trade_plan(
                symbol,
                event_date,
                direction_mode=direction_mode,
            )
        return self._build_swing_trade_plan(symbol, event_date)

    def _build_day_trade_plan(
        self,
        symbol: str,
        event_date: date,
        *,
        direction_mode: DayTradeDirectionMode = DEFAULT_LIVE_DAY_TRADE_DIRECTION_MODE,
    ) -> TradePlanRecord:
        bias = build_intraday_trading_bias(
            symbol,
            yfinance_adapter=self.yfinance_adapter,
            loaded_model=self.loaded_model,
            pattern_analysis_service=self.pattern_analysis_service,
            research_events_service=self.research_events_service,
        )
        levels = bias.levels
        or_high = _round_price(levels.open_range_high)
        or_low = _round_price(levels.open_range_low)
        vwap = _round_price(levels.vwap)
        if or_high is None or or_low is None or vwap is None:
            historical_levels = self._historical_day_trade_levels(symbol, event_date)
            or_high = or_high if or_high is not None else historical_levels.get("or_high")
            or_low = or_low if or_low is not None else historical_levels.get("or_low")
            vwap = vwap if vwap is not None else historical_levels.get("vwap")
        width = _round_price(
            (or_high - or_low)
            if or_high is not None and or_low is not None
            else None
        )
        buffer = 0.01
        allow_long = day_trade_direction_allowed("long", direction_mode)
        allow_short = day_trade_direction_allowed("short", direction_mode)
        plan_levels: dict[str, float | str | None] = {
            "long_entry": (
                _round_price(or_high + buffer) if or_high is not None else None
            )
            if allow_long
            else None,
            "long_stop": vwap if allow_long else None,
            "long_target_1": (
                _round_price(or_high + buffer + width)
                if or_high is not None and width is not None
                else None
            )
            if allow_long
            else None,
            "long_target_2": _round_price(levels.resistance) if allow_long else None,
            "short_entry": (
                _round_price(or_low - buffer) if or_low is not None else None
            )
            if allow_short
            else None,
            "short_stop": vwap if allow_short else None,
            "short_target_1": (
                _round_price(or_low - buffer - width)
                if or_low is not None and width is not None
                else None
            )
            if allow_short
            else None,
            "short_target_2": _round_price(levels.support) if allow_short else None,
            "open_range_high": or_high,
            "open_range_low": or_low,
            "vwap": vwap,
            "setup_type": (
                bias.setup_type
                if bias.setup_type != "None" or or_high is None or or_low is None
                else "OpeningRangeBreakout"
            ),
            "direction_mode": direction_mode,
        }
        logger.info(
            (
                "Day trade plan levels trace: symbol=%s date=%s direction_mode=%s "
                "or_high=%s or_low=%s vwap=%s width=%s long_entry=%s "
                "short_entry=%s data_gaps=%s warnings=%s"
            ),
            symbol,
            event_date.isoformat(),
            direction_mode,
            or_high,
            or_low,
            vwap,
            width,
            plan_levels["long_entry"],
            plan_levels["short_entry"],
            bias.data_gaps,
            bias.warnings,
        )
        generated_at = _day_plan_start(event_date)
        payload = {
            "workflow": "day_trade",
            "bias": bias.bias,
            "confidence": bias.confidence,
            "action": bias.action,
            "levels": plan_levels,
            "direction_mode": direction_mode,
            "warnings": bias.warnings,
            "data_gaps": bias.data_gaps,
            "methodology": "Opening range breakout/breakdown plan using VWAP as intraday control.",
        }
        return _plan_record(
            symbol=symbol,
            workflow="day_trade",
            plan_date=event_date,
            generated_at=generated_at,
            levels=plan_levels,
            payload=payload,
        )

    def _historical_day_trade_levels(
        self,
        symbol: str,
        event_date: date,
    ) -> dict[str, float | None]:
        hist = self.yfinance_adapter.get_history(
            symbol,
            period="5d",
            interval="5m",
            prepost=True,
        )
        bars = [
            bar
            for bar in _normalize_history(hist)
            if _market_date(bar.timestamp) == event_date and _is_regular_session(bar)
        ]
        opening_range_bars = [
            bar for bar in bars if _local_time(bar.timestamp) < OPENING_RANGE_END
        ]
        levels = {
            "or_high": _round_price(_max_high(opening_range_bars)),
            "or_low": _round_price(_min_low(opening_range_bars)),
            "vwap": _round_price(_vwap(opening_range_bars)),
        }
        logger.info(
            (
                "Historical day trade levels trace: symbol=%s date=%s "
                "regular_bars=%s opening_range_bars=%s or_high=%s or_low=%s vwap=%s"
            ),
            symbol,
            event_date.isoformat(),
            len(bars),
            len(opening_range_bars),
            levels["or_high"],
            levels["or_low"],
            levels["vwap"],
        )
        return levels

    def _build_swing_trade_plan(self, symbol: str, event_date: date) -> TradePlanRecord:
        playbook = build_trader_playbook(
            symbol,
            loaded_model=self.loaded_model,
            pattern_analysis_service=self.pattern_analysis_service,
            research_events_service=self.research_events_service,
        )
        levels = playbook.levels
        plan_levels: dict[str, float | str | None] = {
            "side": playbook.side,
            "entry": _round_price(levels.entry),
            "stop": _round_price(levels.stop),
            "target_1": _round_price(levels.target1),
            "target_2": _round_price(levels.target2),
            "support": _round_price(levels.support),
            "resistance": _round_price(levels.resistance),
            "breakout_level": _round_price(levels.breakout_level),
        }
        payload = {
            "workflow": "swing_trade",
            "direction": playbook.direction,
            "confidence": playbook.confidence,
            "best_setup": playbook.best_setup,
            "side": playbook.side,
            "status": playbook.status,
            "levels": plan_levels,
            "risk": playbook.risk.model_dump(mode="json", by_alias=True),
            "warnings": playbook.warnings,
            "data_gaps": playbook.data_gaps,
        }
        return _plan_record(
            symbol=symbol,
            workflow="swing_trade",
            plan_date=event_date,
            generated_at=_session_start_utc(event_date),
            levels=plan_levels,
            payload=payload,
        )

    def _load_bars(
        self,
        symbol: str,
        workflow: TradeReplayWorkflow,
        event_date: date,
        generated_at: datetime,
    ) -> list[ReplayBar]:
        if workflow == "day_trade":
            hist = self.yfinance_adapter.get_history(
                symbol,
                period="5d",
                interval="5m",
                prepost=True,
            )
        else:
            hist = self.yfinance_adapter.get_history(
                symbol,
                period="3mo",
                interval="1d",
                prepost=False,
            )
        bars = _normalize_history(hist)
        return [
            bar
            for bar in bars
            if _market_date(bar.timestamp) == event_date and bar.timestamp >= generated_at
        ]


def generate_replay_events(
    plan: TradePlanRecord,
    bars: list[ReplayBar],
    direction_mode: DayTradeDirectionMode = "long_and_short",
) -> list[TradeReplayEvent]:
    if plan.workflow == "day_trade":
        return _generate_day_trade_events(plan, bars, direction_mode=direction_mode)
    return _generate_swing_trade_events(plan, bars)


def build_completed_missed_moves(
    plan: TradePlanRecord,
    events: list[TradeReplayEvent],
    bars: list[ReplayBar],
) -> list[MissedMoveRecord]:
    completed: list[MissedMoveRecord] = []
    active_trigger: TradeReplayEvent | None = None
    story_events: list[TradeReplayEvent] = []

    for event in _sort_story_events(events):
        if _is_trigger_event(event):
            active_trigger = event
            story_events = [event]
            continue

        if active_trigger is None:
            continue

        story_events.append(event)
        if event.event_type not in COMPLETED_MISSED_MOVE_OUTCOMES:
            continue

        outcome = _missed_move_outcome(event)
        if outcome is None:
            active_trigger = None
            story_events = []
            continue

        completed.append(
            _missed_move_record(
                plan=plan,
                trigger=active_trigger,
                terminal=event,
                replay_events=list(story_events),
                bars=bars,
                outcome=outcome,
            )
        )
        active_trigger = None
        story_events = []

    return completed


def plan_signature(
    levels: dict[str, Any],
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    compact = {
        "levels": _stable_json_ready(levels),
        "payload": _stable_json_ready(payload),
        "metadata": _stable_json_ready(metadata or {}),
    }
    text = json.dumps(compact, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _generate_day_trade_events(
    plan: TradePlanRecord,
    bars: list[ReplayBar],
    direction_mode: DayTradeDirectionMode = "long_and_short",
) -> list[TradeReplayEvent]:
    levels = plan.levels
    allow_long = day_trade_direction_allowed("long", direction_mode)
    allow_short = day_trade_direction_allowed("short", direction_mode)
    long_entry = _float(levels.get("long_entry")) if allow_long else None
    long_stop = _float(levels.get("long_stop")) if allow_long else None
    long_target = _float(levels.get("long_target_1")) if allow_long else None
    long_target_2 = _float(levels.get("long_target_2")) if allow_long else None
    short_entry = _float(levels.get("short_entry")) if allow_short else None
    short_stop = _float(levels.get("short_stop")) if allow_short else None
    short_target = _float(levels.get("short_target_1")) if allow_short else None
    short_target_2 = _float(levels.get("short_target_2")) if allow_short else None
    or_high = _float(levels.get("open_range_high"))
    or_low = _float(levels.get("open_range_low"))
    vwap = _float(levels.get("vwap"))

    events: list[TradeReplayEvent] = []
    long_active = False
    short_active = False
    long_target_1_hit = False
    short_target_1_hit = False
    long_done = False
    short_done = False

    for bar in sorted(bars, key=lambda item: item.timestamp):
        if (
            long_entry is not None
            and not long_done
            and not long_active
            and bar.high >= long_entry
        ):
            long_active = True
            events.append(
                _event(
                    plan,
                    event_type="long_trigger_activated",
                    bar=bar,
                    level_price=long_entry,
                    observed_price=max(bar.close, long_entry),
                    message=_with_session_context(
                        bar,
                        f"{plan.symbol} broke above the opening range trigger near ${long_entry:.2f}.",
                    ),
                    severity="important",
                    actionability="active",
                    dedupe_key=f"long-trigger:{long_entry:.2f}",
                )
            )
        if (
            short_entry is not None
            and not short_done
            and not short_active
            and bar.low <= short_entry
        ):
            short_active = True
            events.append(
                _event(
                    plan,
                    event_type="short_trigger_activated",
                    bar=bar,
                    level_price=short_entry,
                    observed_price=min(bar.close, short_entry),
                    message=_with_session_context(
                        bar,
                        f"{plan.symbol} broke below the opening range trigger near ${short_entry:.2f}.",
                    ),
                    severity="important",
                    actionability="active",
                    dedupe_key=f"short-trigger:{short_entry:.2f}",
                )
            )
        if (
            long_active
            and not long_target_1_hit
            and long_target is not None
            and bar.high >= long_target
        ):
            if long_stop is not None and bar.low <= long_stop:
                events.append(
                    _ambiguous_stop_target_event(
                        plan,
                        bar=bar,
                        side="Long",
                        stop=long_stop,
                        target=long_target,
                    )
                )
                long_active = False
                long_done = True
                continue
            long_target_1_hit = True
            events.append(
                _event(
                    plan,
                    event_type="target_1_hit",
                    bar=bar,
                    level_price=long_target,
                    observed_price=max(bar.close, long_target),
                    message=_with_session_context(
                        bar,
                        f"Long target 1 was reached near ${long_target:.2f}. Planned expansion delivered.",
                    ),
                    severity="important",
                    actionability="missed",
                    dedupe_key=f"long-target-1:{long_target:.2f}",
                )
            )
            if long_target_2 is None:
                long_active = False
                long_done = True
        if (
            short_active
            and not short_target_1_hit
            and short_target is not None
            and bar.low <= short_target
        ):
            if short_stop is not None and bar.high >= short_stop:
                events.append(
                    _ambiguous_stop_target_event(
                        plan,
                        bar=bar,
                        side="Short",
                        stop=short_stop,
                        target=short_target,
                    )
                )
                short_active = False
                short_done = True
                continue
            short_target_1_hit = True
            events.append(
                _event(
                    plan,
                    event_type="target_1_hit",
                    bar=bar,
                    level_price=short_target,
                    observed_price=min(bar.close, short_target),
                    message=_with_session_context(
                        bar,
                        f"Short target 1 was reached near ${short_target:.2f}. Planned expansion delivered.",
                    ),
                    severity="important",
                    actionability="missed",
                    dedupe_key=f"short-target-1:{short_target:.2f}",
                )
            )
            if short_target_2 is None:
                short_active = False
                short_done = True
        if (
            long_active
            and long_target_1_hit
            and long_target_2 is not None
            and bar.high >= long_target_2
        ):
            events.append(
                _event(
                    plan,
                    event_type="target_2_hit",
                    bar=bar,
                    level_price=long_target_2,
                    observed_price=max(bar.close, long_target_2),
                    message=_with_session_context(
                        bar,
                        f"Long target 2 was reached near ${long_target_2:.2f}. Extended plan objective was reached.",
                    ),
                    severity="important",
                    actionability="missed",
                    dedupe_key=f"long-target-2:{long_target_2:.2f}",
                    time_offset_microseconds=TARGET_2_OFFSET_MICROSECONDS,
                )
            )
            long_active = False
            long_done = True
        if (
            short_active
            and short_target_1_hit
            and short_target_2 is not None
            and bar.low <= short_target_2
        ):
            events.append(
                _event(
                    plan,
                    event_type="target_2_hit",
                    bar=bar,
                    level_price=short_target_2,
                    observed_price=min(bar.close, short_target_2),
                    message=_with_session_context(
                        bar,
                        f"Short target 2 was reached near ${short_target_2:.2f}. Extended plan objective was reached.",
                    ),
                    severity="important",
                    actionability="missed",
                    dedupe_key=f"short-target-2:{short_target_2:.2f}",
                    time_offset_microseconds=TARGET_2_OFFSET_MICROSECONDS,
                )
            )
            short_active = False
            short_done = True
        if long_active and long_stop is not None and bar.low <= long_stop:
            events.append(
                _event(
                    plan,
                    event_type="stop_hit",
                    bar=bar,
                    level_price=long_stop,
                    observed_price=min(bar.close, long_stop),
                    message=_with_session_context(
                        bar,
                        f"Long plan control level was lost near ${long_stop:.2f}.",
                    ),
                    severity="warning",
                    actionability="invalidated",
                    dedupe_key=f"long-stop:{long_stop:.2f}",
                )
            )
            long_active = False
            long_done = True
        if short_active and short_stop is not None and bar.high >= short_stop:
            events.append(
                _event(
                    plan,
                    event_type="stop_hit",
                    bar=bar,
                    level_price=short_stop,
                    observed_price=max(bar.close, short_stop),
                    message=_with_session_context(
                        bar,
                        f"Short plan control level was reclaimed near ${short_stop:.2f}.",
                    ),
                    severity="warning",
                    actionability="invalidated",
                    dedupe_key=f"short-stop:{short_stop:.2f}",
                )
            )
            short_active = False
            short_done = True
        long_extension_target = (
            long_target_2 if long_target_1_hit and long_target_2 is not None else long_target
        )
        if (
            long_active
            and long_extension_target is not None
            and long_stop is not None
        ):
            if (long_extension_target - bar.close) <= max(bar.close - long_stop, 0):
                events.append(
                    _event(
                        plan,
                        event_type="setup_extended",
                        bar=bar,
                        level_price=long_extension_target,
                        observed_price=bar.close,
                        message=_with_session_context(
                            bar,
                            "Long setup became extended as remaining reward compressed versus current risk.",
                        ),
                        severity="warning",
                        actionability="missed",
                        dedupe_key=(
                            f"long-extended:{long_entry:.2f}"
                            if long_entry is not None
                            else "long-extended"
                        ),
                    )
                )
                long_active = False
                long_done = True
        short_extension_target = (
            short_target_2
            if short_target_1_hit and short_target_2 is not None
            else short_target
        )
        if (
            short_active
            and short_extension_target is not None
            and short_stop is not None
        ):
            if (bar.close - short_extension_target) <= max(short_stop - bar.close, 0):
                events.append(
                    _event(
                        plan,
                        event_type="setup_extended",
                        bar=bar,
                        level_price=short_extension_target,
                        observed_price=bar.close,
                        message=_with_session_context(
                            bar,
                            "Short setup became extended as remaining reward compressed versus current risk.",
                        ),
                        severity="warning",
                        actionability="missed",
                        dedupe_key=(
                            f"short-extended:{short_entry:.2f}"
                            if short_entry is not None
                            else "short-extended"
                        ),
                    )
                )
                short_active = False
                short_done = True
        if long_active and or_high is not None and bar.close < or_high:
            events.append(
                _event(
                    plan,
                    event_type="setup_invalidated",
                    bar=bar,
                    level_price=or_high,
                    observed_price=bar.close,
                    message=_with_session_context(
                        bar,
                        "Long breakout failed by closing back inside the opening range.",
                    ),
                    severity="warning",
                    actionability="invalidated",
                    dedupe_key=f"long-invalidated:{or_high:.2f}",
                )
            )
            long_active = False
            long_done = True
        if short_active and or_low is not None and bar.close > or_low:
            events.append(
                _event(
                    plan,
                    event_type="setup_invalidated",
                    bar=bar,
                    level_price=or_low,
                    observed_price=bar.close,
                    message=_with_session_context(
                        bar,
                        "Short breakdown failed by closing back inside the opening range.",
                    ),
                    severity="warning",
                    actionability="invalidated",
                    dedupe_key=f"short-invalidated:{or_low:.2f}",
                )
            )
            short_active = False
            short_done = True

    return _story_events(events)


def _generate_swing_trade_events(
    plan: TradePlanRecord,
    bars: list[ReplayBar],
) -> list[TradeReplayEvent]:
    levels = plan.levels
    side = str(levels.get("side") or "NoTrade")
    entry = _float(levels.get("entry"))
    stop = _float(levels.get("stop"))
    target = _float(levels.get("target_1"))
    if side not in {"Long", "Short"} or entry is None:
        return []

    events: list[TradeReplayEvent] = []
    active = False
    for bar in sorted(bars, key=lambda item: item.timestamp):
        entry_hit = bar.high >= entry if side == "Long" else bar.low <= entry
        if not active and entry_hit:
            active = True
            events.append(
                _event(
                    plan,
                    event_type="entry_triggered",
                    bar=bar,
                    level_price=entry,
                    observed_price=entry,
                    message=_with_session_context(
                        bar,
                        f"{side} swing entry triggered near ${entry:.2f}.",
                    ),
                    severity="important",
                    actionability="active",
                    dedupe_key=f"entry:{side.lower()}:{entry:.2f}",
                )
            )
        if not active:
            continue
        if (
            target is not None
            and stop is not None
            and (
                (
                    side == "Long"
                    and bar.high >= target
                    and bar.low <= stop
                )
                or (
                    side == "Short"
                    and bar.low <= target
                    and bar.high >= stop
                )
            )
        ):
            events.append(
                _ambiguous_stop_target_event(
                    plan,
                    bar=bar,
                    side=side,
                    stop=stop,
                    target=target,
                )
            )
            active = False
            continue
        if target is not None and (
            (side == "Long" and bar.high >= target)
            or (side == "Short" and bar.low <= target)
        ):
            events.append(
                _event(
                    plan,
                    event_type="target_hit",
                    bar=bar,
                    level_price=target,
                    observed_price=target,
                    message=_with_session_context(
                        bar,
                        f"Swing target was reached near ${target:.2f}.",
                    ),
                    severity="important",
                    actionability="missed",
                    dedupe_key=f"target:{side.lower()}:{target:.2f}",
                )
            )
            active = False
        if active and stop is not None and (
            (side == "Long" and bar.low <= stop)
            or (side == "Short" and bar.high >= stop)
        ):
            events.append(
                _event(
                    plan,
                    event_type="stop_hit",
                    bar=bar,
                    level_price=stop,
                    observed_price=stop,
                    message=_with_session_context(
                        bar,
                        f"Swing setup invalidated at the stop level near ${stop:.2f}.",
                    ),
                    severity="warning",
                    actionability="invalidated",
                    dedupe_key=f"stop:{side.lower()}:{stop:.2f}",
                )
            )
            active = False
        if active and target is not None and stop is not None:
            remaining = target - bar.close if side == "Long" else bar.close - target
            risk = bar.close - stop if side == "Long" else stop - bar.close
            if remaining > 0 and risk > 0 and remaining / risk < 1:
                events.append(
                    _event(
                        plan,
                        event_type="rr_degraded",
                        bar=bar,
                        level_price=target,
                        observed_price=bar.close,
                        message=_with_session_context(
                            bar,
                            "Current swing R/R degraded below 1R as price moved away from the original trigger.",
                        ),
                        severity="warning",
                        actionability="missed",
                        dedupe_key=f"rr-degraded:{side.lower()}:{entry:.2f}",
                    )
                )
                active = False
    return _story_events(events)


def _plan_record(
    *,
    symbol: str,
    workflow: TradeReplayWorkflow,
    plan_date: date,
    generated_at: datetime,
    levels: dict[str, float | str | None],
    payload: dict[str, Any],
) -> TradePlanRecord:
    metadata = {
        "workflow": workflow,
        "plan_date": plan_date,
        "source": "delayed",
        "source_freshness_label": SOURCE_FRESHNESS_DELAYED,
    }
    signature = plan_signature(levels, payload, metadata)
    return TradePlanRecord(
        plan_id=f"plan-{uuid.uuid4().hex}",
        symbol=symbol,
        workflow=workflow,
        plan_date=plan_date,
        generated_at=generated_at,
        source="delayed",
        source_freshness_label=SOURCE_FRESHNESS_DELAYED,
        signature=signature,
        levels=levels,
        payload=payload,
    )


def _ambiguous_stop_target_event(
    plan: TradePlanRecord,
    *,
    bar: ReplayBar,
    side: str,
    stop: float,
    target: float,
) -> TradeReplayEvent:
    return _event(
        plan,
        event_type="setup_invalidated",
        bar=bar,
        level_price=stop,
        observed_price=bar.close,
        message=(
            f"{side} plan became ambiguous because the same bar touched both "
            f"stop ${stop:.2f} and target ${target:.2f}. Replay treats this conservatively."
        ),
        severity="warning",
        actionability="invalidated",
        dedupe_key=f"{side.lower()}-ambiguous-stop-target:{stop:.2f}:{target:.2f}",
    )


def _event(
    plan: TradePlanRecord,
    *,
    event_type: str,
    bar: ReplayBar,
    level_price: float | None,
    observed_price: float | None,
    message: str,
    severity: str,
    actionability: str,
    dedupe_key: str,
    time_offset_microseconds: int = 0,
) -> TradeReplayEvent:
    event_time = bar.timestamp
    if time_offset_microseconds:
        event_time = event_time + pd.Timedelta(
            microseconds=time_offset_microseconds
        ).to_pytimedelta()
    return TradeReplayEvent(
        id=f"evt-{uuid.uuid4().hex}",
        plan_id=plan.plan_id,
        symbol=plan.symbol,
        event_date=plan.plan_date,
        workflow=plan.workflow,
        event_type=event_type,
        event_time=event_time,
        level_price=_round_price(level_price),
        observed_price=_round_price(observed_price),
        message=message,
        severity=severity,  # type: ignore[arg-type]
        actionability=actionability,  # type: ignore[arg-type]
        source=plan.source,
        source_freshness_label=plan.source_freshness_label,
        dedupe_key=dedupe_key,
        created_at=datetime.now(timezone.utc),
    )


def _normalize_history(hist: pd.DataFrame | None) -> list[ReplayBar]:
    if hist is None or hist.empty or not isinstance(hist.index, pd.DatetimeIndex):
        return []
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(hist.columns):
        return []
    bars: list[ReplayBar] = []
    for timestamp, row in hist.iterrows():
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        open_price = _float(row.get("Open"))
        high = _float(row.get("High"))
        low = _float(row.get("Low"))
        close = _float(row.get("Close"))
        if open_price is None or high is None or low is None or close is None:
            continue
        bars.append(
            ReplayBar(
                timestamp=ts.to_pydatetime(),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=max(0, int(_float(row.get("Volume")) or 0)),
            )
        )
    return sorted(bars, key=lambda bar: bar.timestamp)


def _day_plan_start(value: date) -> datetime:
    return datetime.combine(value, time(10, 0), tzinfo=EASTERN).astimezone(timezone.utc)


def _session_start_utc(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _market_date(value: datetime) -> date:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(EASTERN).date()


def _local_time(value: datetime) -> time:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(EASTERN).time()


def _is_regular_session(bar: ReplayBar) -> bool:
    local_time = _local_time(bar.timestamp)
    return MARKET_OPEN <= local_time < MARKET_CLOSE


def _max_high(bars: list[ReplayBar]) -> float | None:
    if not bars:
        return None
    return max(bar.high for bar in bars)


def _min_low(bars: list[ReplayBar]) -> float | None:
    if not bars:
        return None
    return min(bar.low for bar in bars)


def _vwap(bars: list[ReplayBar]) -> float | None:
    volume_total = sum(max(bar.volume, 0) for bar in bars)
    if volume_total <= 0:
        return None
    dollar_volume = sum(
        ((bar.high + bar.low + bar.close) / 3) * max(bar.volume, 0)
        for bar in bars
    )
    return dollar_volume / volume_total


def _stable_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable_json_ready(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _dedupe_events(events: list[TradeReplayEvent]) -> list[TradeReplayEvent]:
    seen: set[tuple[str, str]] = set()
    result: list[TradeReplayEvent] = []
    for event in events:
        key = (event.event_type, event.dedupe_key)
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return sorted(result, key=lambda event: event.event_time)


def _event_direction(event: TradeReplayEvent) -> str | None:
    text = f"{event.event_type} {event.dedupe_key} {event.message}".lower()
    if "short" in text:
        return "short"
    if "long" in text:
        return "long"
    return None


def _filter_events_for_direction(
    events: list[TradeReplayEvent],
    direction_mode: DayTradeDirectionMode,
) -> list[TradeReplayEvent]:
    if direction_mode == "long_and_short":
        return events
    target_direction = "long" if direction_mode == "long_only" else "short"
    return [
        event
        for event in events
        if _event_direction(event) in {None, target_direction}
    ]


_STORY_EVENT_TYPES = {
    "entry_triggered",
    "long_trigger_activated",
    "short_trigger_activated",
    "target_1_hit",
    "target_2_hit",
    "target_hit",
    "setup_extended",
    "setup_invalidated",
    "stop_hit",
    "rr_degraded",
    "pullback_opportunity",
    "entry_missed",
}

_EVENT_PRIORITY = {
    "entry_triggered": 10,
    "long_trigger_activated": 10,
    "short_trigger_activated": 10,
    "target_1_hit": 20,
    "target_hit": 20,
    "target_2_hit": 21,
    "setup_extended": 30,
    "rr_degraded": 30,
    "entry_missed": 30,
    "pullback_opportunity": 30,
    "setup_invalidated": 40,
    "stop_hit": 40,
}


def _story_events(events: list[TradeReplayEvent]) -> list[TradeReplayEvent]:
    deduped = _dedupe_events(
        [event for event in events if event.event_type in _STORY_EVENT_TYPES]
    )
    return _sort_story_events(deduped)


def _sort_story_events(events: list[TradeReplayEvent]) -> list[TradeReplayEvent]:
    return sorted(
        events,
        key=lambda event: (
            event.event_time,
            _EVENT_PRIORITY.get(event.event_type, 100),
        ),
    )


def _is_trigger_event(event: TradeReplayEvent) -> bool:
    return event.event_type in {
        "entry_triggered",
        "long_trigger_activated",
        "short_trigger_activated",
    }


def _missed_move_outcome(event: TradeReplayEvent) -> MissedMoveOutcome | None:
    if event.event_type in {"target_1_hit", "target_2_hit", "target_hit"}:
        return "target_hit"
    if event.event_type in {"setup_extended", "rr_degraded", "entry_missed"}:
        return "extended"
    if event.event_type == "stop_hit":
        return "stopped"
    if event.event_type == "setup_invalidated":
        return "invalidated"
    return None


def _missed_move_record(
    *,
    plan: TradePlanRecord,
    trigger: TradeReplayEvent,
    terminal: TradeReplayEvent,
    replay_events: list[TradeReplayEvent],
    bars: list[ReplayBar],
    outcome: MissedMoveOutcome,
) -> MissedMoveRecord:
    direction = _trigger_side(trigger) or "unknown"
    now = datetime.now(timezone.utc)
    return MissedMoveRecord(
        missed_move_id=_missed_move_id(plan, trigger, terminal),
        symbol=plan.symbol,
        workflow=plan.workflow,
        event_date=trigger.event_date,
        setup_type=_setup_type(plan, trigger),
        direction=direction,
        trigger_time=trigger.event_time,
        trigger_price=trigger.level_price,
        outcome=outcome,
        max_move_after_trigger_pct=_max_move_after_trigger_pct(trigger, bars),
        setup_quality_score=_setup_quality_score(plan),
        entry=_level_for_direction(plan, direction, "entry"),
        stop=_level_for_direction(plan, direction, "stop"),
        target_1=_level_for_direction(plan, direction, "target_1"),
        target_2=_level_for_direction(plan, direction, "target_2"),
        open_range_high=_float(plan.levels.get("open_range_high")),
        open_range_low=_float(plan.levels.get("open_range_low")),
        vwap=_float(plan.levels.get("vwap")),
        event_count=len(replay_events),
        source=plan.source,
        source_freshness_label=plan.source_freshness_label,
        trigger_event_id=trigger.id,
        terminal_event_id=terminal.id,
        replay_events=_sort_story_events(replay_events),
        created_at=now,
        updated_at=now,
    )


def _missed_move_id(
    plan: TradePlanRecord,
    trigger: TradeReplayEvent,
    terminal: TradeReplayEvent,
) -> str:
    seed = "|".join(
        [
            plan.symbol,
            plan.workflow,
            plan.plan_date.isoformat(),
            plan.signature,
            trigger.event_type,
            trigger.dedupe_key,
            terminal.event_type,
            terminal.dedupe_key,
        ]
    )
    return f"mm-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:32]}"


def _setup_type(plan: TradePlanRecord, trigger: TradeReplayEvent) -> str:
    setup = plan.levels.get("setup_type") or plan.payload.get("best_setup")
    if isinstance(setup, str) and setup.strip():
        return setup.strip()
    if trigger.event_type == "long_trigger_activated":
        return "Long opening range breakout"
    if trigger.event_type == "short_trigger_activated":
        return "Short opening range breakdown"
    return "Swing entry"


def _setup_quality_score(plan: TradePlanRecord) -> float | None:
    for key in ("confidence", "setup_quality_score", "quality_score"):
        score = _score_value(plan.payload.get(key))
        if score is not None:
            return score
    return None


def _level_for_direction(
    plan: TradePlanRecord,
    direction: str,
    level: str,
) -> float | None:
    if plan.workflow == "day_trade" and direction in {"long", "short"}:
        return _float(plan.levels.get(f"{direction}_{level}"))
    return _float(plan.levels.get(level))


def _missed_move_natural_key(
    missed_move: MissedMoveRecord,
) -> tuple[str, date, str, str, str]:
    return (
        missed_move.symbol.upper(),
        missed_move.event_date,
        missed_move.workflow,
        missed_move.setup_type,
        missed_move.direction,
    )


def _score_value(value: Any) -> float | None:
    if isinstance(value, str):
        labels = {"low": 1.0, "medium": 2.0, "high": 3.0}
        return labels.get(value.strip().lower())
    number = _float(value)
    if number is None:
        return None
    return number


def _max_move_after_trigger_pct(
    trigger: TradeReplayEvent,
    bars: list[ReplayBar],
) -> float | None:
    if trigger.level_price is None or trigger.level_price <= 0:
        return None
    side = _trigger_side(trigger)
    if side is None:
        return None

    after_trigger = [
        bar
        for bar in bars
        if bar.timestamp >= trigger.event_time
        and _market_date(bar.timestamp) == trigger.event_date
    ]
    if not after_trigger:
        return None

    if side == "long":
        best_price = max(bar.high for bar in after_trigger)
        return round((best_price - trigger.level_price) / trigger.level_price, 4)

    best_price = min(bar.low for bar in after_trigger)
    return round((trigger.level_price - best_price) / trigger.level_price, 4)


def _trigger_side(event: TradeReplayEvent) -> str | None:
    text = f"{event.event_type} {event.dedupe_key} {event.message}".lower()
    if "short" in text:
        return "short"
    if "long" in text:
        return "long"
    return None


def _sort_missed_moves(
    records: list[MissedMoveRecord],
    sort: MissedMovesSort,
) -> list[MissedMoveRecord]:
    if sort == "biggest_move":
        return sorted(
            records,
            key=lambda item: (
                -(item.max_move_after_trigger_pct or 0),
                -_date_ordinal(item.event_date),
                item.symbol,
            ),
        )
    if sort == "highest_setup_quality":
        return sorted(
            records,
            key=lambda item: (
                -(item.setup_quality_score or 0),
                -_date_ordinal(item.event_date),
                item.symbol,
            ),
        )
    return sorted(
        records,
        key=lambda item: (
            -_date_ordinal(item.event_date),
            -max((event.event_time for event in item.replay_events), default=datetime.min.replace(tzinfo=timezone.utc)).timestamp(),
            item.symbol,
        ),
    )


def _count_by_date(records: list[MissedMoveRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = record.event_date.isoformat()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_by_setup_type(records: list[MissedMoveRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.setup_type] = counts.get(record.setup_type, 0) + 1
    return counts


def _date_ordinal(value: date) -> int:
    return value.toordinal()


def _summary_row(record: MissedMoveRecord) -> MissedMoveSummaryRow:
    return MissedMoveSummaryRow(
        id=record.missed_move_id,
        date=record.event_date,
        symbol=record.symbol,
        workflow=record.workflow,
        setup_type=record.setup_type,
        direction=record.direction,
        trigger_time=record.trigger_time,
        trigger_price=record.trigger_price,
        outcome=record.outcome,
        max_move_after_trigger_pct=record.max_move_after_trigger_pct,
        setup_quality_score=record.setup_quality_score,
        entry=record.entry,
        stop=record.stop,
        target_1=record.target_1,
        target_2=record.target_2,
        open_range_high=record.open_range_high,
        open_range_low=record.open_range_low,
        vwap=record.vwap,
        event_count=record.event_count,
        source=record.source,
        source_freshness_label=record.source_freshness_label,
    )


def _current_trading_date(value: datetime) -> date:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    local_value = value.astimezone(EASTERN)
    local_day = local_value.date()
    if local_value.time() < MARKET_OPEN:
        local_day -= timedelta(days=1)
    while not _is_trading_day(local_day):
        local_day -= timedelta(days=1)
    return local_day


def _subtract_trading_days(value: date, count: int) -> date:
    cursor = value
    remaining = count
    while remaining > 0:
        cursor -= timedelta(days=1)
        if _is_trading_day(cursor):
            remaining -= 1
    return cursor


def _trading_day_window(end_date: date, count: int) -> set[date]:
    if count <= 0:
        return set()
    dates = {end_date}
    cursor = end_date
    while len(dates) < count:
        cursor = _subtract_trading_days(cursor, 1)
        dates.add(cursor)
    return dates


def _is_trading_day(value: date) -> bool:
    return value.weekday() < 5 and value not in _market_holidays_for_date(value)


def _market_holidays_for_date(value: date) -> set[date]:
    return (
        _market_holidays(value.year - 1)
        | _market_holidays(value.year)
        | _market_holidays(value.year + 1)
    )


def _market_holidays(year: int) -> set[date]:
    new_year = _observed_holiday(date(year, 1, 1))
    juneteenth = _observed_holiday(date(year, 6, 19))
    independence = _observed_holiday(date(year, 7, 4))
    christmas = _observed_holiday(date(year, 12, 25))
    return {
        holiday
        for holiday in {
            new_year,
            _nth_weekday(year, 1, 0, 3),
            _nth_weekday(year, 2, 0, 3),
            _good_friday(year),
            _last_weekday(year, 5, 0),
            juneteenth if year >= 2022 else None,
            independence,
            _nth_weekday(year, 9, 0, 1),
            _nth_weekday(year, 11, 3, 4),
            christmas,
        }
        if holiday is not None and holiday.year == year
    }


def _observed_holiday(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    cursor = date(year, month, 1)
    while cursor.weekday() != weekday:
        cursor += timedelta(days=1)
    return cursor + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    cursor = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while cursor.weekday() != weekday:
        cursor -= timedelta(days=1)
    return cursor


def _good_friday(year: int) -> date:
    # Anonymous Gregorian algorithm for Easter Sunday, then subtract two days.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day) - timedelta(days=2)


def _with_session_context(bar: ReplayBar, message: str) -> str:
    return f"{_session_label(bar.timestamp)} ET: {message}"


def _session_label(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    local_time = value.astimezone(EASTERN).time()
    if local_time < time(9, 30):
        return "Pre-market"
    if local_time < time(16, 0):
        return "Regular session"
    return "After-hours"


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_price(value: Any) -> float | None:
    number = _float(value)
    return round(number, 2) if number is not None else None
