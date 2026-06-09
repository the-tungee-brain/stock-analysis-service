from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Protocol

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.intraday_trading_bias_engine import EASTERN
from app.models.trade_replay_models import (
    TradeReplayEvent,
    TradeReplayRefreshResponse,
    TradeReplayResponse,
    TradeReplaySource,
    TradeReplayWorkflow,
)
from app.services.intraday_trading_bias_service import build_intraday_trading_bias
from app.services.trader_playbook_service import build_trader_playbook
from models.prediction_service import LoadedModel

SOURCE_FRESHNESS_DELAYED = "Educational / delayed — not for live execution."


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


class InMemoryTradeReplayStore:
    """Small test/local store with the same idempotency semantics as Oracle."""

    def __init__(self) -> None:
        self.plans: list[TradePlanRecord] = []
        self.events: list[TradeReplayEvent] = []
        self._event_keys: set[tuple[str, date, str, str, str]] = set()

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
    ) -> TradeReplayResponse:
        symbol_upper = symbol.strip().upper()
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
        return TradeReplayResponse(
            symbol=symbol_upper,
            date=event_date,
            workflow=workflow,
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
    ) -> TradeReplayRefreshResponse:
        symbol_upper = symbol.strip().upper()
        plan = self._build_plan(symbol_upper, workflow, event_date)
        saved = self.store.save_plan_if_changed(plan)
        bars = self._load_bars(
            symbol_upper,
            workflow,
            event_date,
            saved.plan.generated_at,
        )
        events = generate_replay_events(saved.plan, bars)
        created = self.store.append_events(events)
        return TradeReplayRefreshResponse(
            success=True,
            symbol=symbol_upper,
            date=event_date,
            workflow=workflow,
            plan_id=saved.plan.plan_id,
            plan_created=saved.created,
            events_created=created,
            source=saved.plan.source,
            source_freshness_label=saved.plan.source_freshness_label,
        )

    def _build_plan(
        self,
        symbol: str,
        workflow: TradeReplayWorkflow,
        event_date: date,
    ) -> TradePlanRecord:
        if workflow == "day_trade":
            return self._build_day_trade_plan(symbol, event_date)
        return self._build_swing_trade_plan(symbol, event_date)

    def _build_day_trade_plan(self, symbol: str, event_date: date) -> TradePlanRecord:
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
        width = _round_price(
            (or_high - or_low)
            if or_high is not None and or_low is not None
            else None
        )
        buffer = 0.01
        plan_levels: dict[str, float | str | None] = {
            "long_entry": (
                _round_price(or_high + buffer) if or_high is not None else None
            ),
            "long_stop": vwap,
            "long_target_1": (
                _round_price(or_high + buffer + width)
                if or_high is not None and width is not None
                else None
            ),
            "long_target_2": _round_price(levels.resistance),
            "short_entry": (
                _round_price(or_low - buffer) if or_low is not None else None
            ),
            "short_stop": vwap,
            "short_target_1": (
                _round_price(or_low - buffer - width)
                if or_low is not None and width is not None
                else None
            ),
            "short_target_2": _round_price(levels.support),
            "open_range_high": or_high,
            "open_range_low": or_low,
            "vwap": vwap,
            "setup_type": bias.setup_type,
        }
        generated_at = _day_plan_start(event_date)
        payload = {
            "workflow": "day_trade",
            "bias": bias.bias,
            "confidence": bias.confidence,
            "action": bias.action,
            "levels": plan_levels,
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
) -> list[TradeReplayEvent]:
    if plan.workflow == "day_trade":
        return _generate_day_trade_events(plan, bars)
    return _generate_swing_trade_events(plan, bars)


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
) -> list[TradeReplayEvent]:
    levels = plan.levels
    long_entry = _float(levels.get("long_entry"))
    long_stop = _float(levels.get("long_stop"))
    long_target = _float(levels.get("long_target_1"))
    long_target_2 = _float(levels.get("long_target_2"))
    short_entry = _float(levels.get("short_entry"))
    short_stop = _float(levels.get("short_stop"))
    short_target = _float(levels.get("short_target_1"))
    short_target_2 = _float(levels.get("short_target_2"))
    or_high = _float(levels.get("open_range_high"))
    or_low = _float(levels.get("open_range_low"))
    vwap = _float(levels.get("vwap"))

    events: list[TradeReplayEvent] = []
    long_active = False
    short_active = False
    long_target_1_hit = False
    short_target_1_hit = False
    was_above_vwap: bool | None = None

    for bar in sorted(bars, key=lambda item: item.timestamp):
        if long_entry is not None and not long_active and bar.high >= long_entry:
            long_active = True
            events.append(
                _event(
                    plan,
                    event_type="long_trigger_activated",
                    bar=bar,
                    level_price=long_entry,
                    observed_price=max(bar.close, long_entry),
                    message=f"{plan.symbol} broke above the opening range trigger near ${long_entry:.2f}.",
                    severity="important",
                    actionability="active",
                    dedupe_key=f"long-trigger:{long_entry:.2f}",
                )
            )
        if short_entry is not None and not short_active and bar.low <= short_entry:
            short_active = True
            events.append(
                _event(
                    plan,
                    event_type="short_trigger_activated",
                    bar=bar,
                    level_price=short_entry,
                    observed_price=min(bar.close, short_entry),
                    message=f"{plan.symbol} broke below the opening range trigger near ${short_entry:.2f}.",
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
                continue
            long_target_1_hit = True
            events.append(
                _event(
                    plan,
                    event_type="target_1_hit",
                    bar=bar,
                    level_price=long_target,
                    observed_price=max(bar.close, long_target),
                    message=f"Long target 1 was reached near ${long_target:.2f}. Planned expansion delivered.",
                    severity="important",
                    actionability="missed",
                    dedupe_key=f"long-target-1:{long_target:.2f}",
                )
            )
            if long_target_2 is None:
                long_active = False
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
                continue
            short_target_1_hit = True
            events.append(
                _event(
                    plan,
                    event_type="target_1_hit",
                    bar=bar,
                    level_price=short_target,
                    observed_price=min(bar.close, short_target),
                    message=f"Short target 1 was reached near ${short_target:.2f}. Planned expansion delivered.",
                    severity="important",
                    actionability="missed",
                    dedupe_key=f"short-target-1:{short_target:.2f}",
                )
            )
            if short_target_2 is None:
                short_active = False
        if long_active and long_target_2 is not None and bar.high >= long_target_2:
            events.append(
                _event(
                    plan,
                    event_type="target_2_hit",
                    bar=bar,
                    level_price=long_target_2,
                    observed_price=max(bar.close, long_target_2),
                    message=f"Long target 2 was reached near ${long_target_2:.2f}. Extended plan objective was reached.",
                    severity="important",
                    actionability="missed",
                    dedupe_key=f"long-target-2:{long_target_2:.2f}",
                )
            )
            long_active = False
        if short_active and short_target_2 is not None and bar.low <= short_target_2:
            events.append(
                _event(
                    plan,
                    event_type="target_2_hit",
                    bar=bar,
                    level_price=short_target_2,
                    observed_price=min(bar.close, short_target_2),
                    message=f"Short target 2 was reached near ${short_target_2:.2f}. Extended plan objective was reached.",
                    severity="important",
                    actionability="missed",
                    dedupe_key=f"short-target-2:{short_target_2:.2f}",
                )
            )
            short_active = False
        if long_active and long_stop is not None and bar.low <= long_stop:
            events.append(
                _event(
                    plan,
                    event_type="stop_hit",
                    bar=bar,
                    level_price=long_stop,
                    observed_price=min(bar.close, long_stop),
                    message=f"Long plan control level was lost near ${long_stop:.2f}.",
                    severity="warning",
                    actionability="invalidated",
                    dedupe_key=f"long-stop:{long_stop:.2f}",
                )
            )
            long_active = False
        if short_active and short_stop is not None and bar.high >= short_stop:
            events.append(
                _event(
                    plan,
                    event_type="stop_hit",
                    bar=bar,
                    level_price=short_stop,
                    observed_price=max(bar.close, short_stop),
                    message=f"Short plan control level was reclaimed near ${short_stop:.2f}.",
                    severity="warning",
                    actionability="invalidated",
                    dedupe_key=f"short-stop:{short_stop:.2f}",
                )
            )
            short_active = False
        if vwap is not None:
            above = bar.close >= vwap
            if was_above_vwap is not None and above != was_above_vwap:
                event_type = "vwap_reclaimed" if above else "vwap_lost"
                events.append(
                    _event(
                        plan,
                        event_type=event_type,
                        bar=bar,
                        level_price=vwap,
                        observed_price=bar.close,
                        message=(
                            f"Price {'reclaimed' if above else 'lost'} VWAP near ${vwap:.2f}, "
                            "the intraday control line."
                        ),
                        severity="info",
                        actionability="active" if above else "invalidated",
                        dedupe_key=f"{event_type}:{bar.timestamp.isoformat()}",
                    )
                )
            was_above_vwap = above
        if long_active and long_target is not None and long_stop is not None:
            if (long_target - bar.close) <= max(bar.close - long_stop, 0):
                events.append(
                    _event(
                        plan,
                        event_type="setup_extended",
                        bar=bar,
                        level_price=long_target,
                        observed_price=bar.close,
                        message="Long setup became extended as remaining reward compressed versus current risk.",
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
        if short_active and short_target is not None and short_stop is not None:
            if (bar.close - short_target) <= max(short_stop - bar.close, 0):
                events.append(
                    _event(
                        plan,
                        event_type="setup_extended",
                        bar=bar,
                        level_price=short_target,
                        observed_price=bar.close,
                        message="Short setup became extended as remaining reward compressed versus current risk.",
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
        if long_active and or_high is not None and bar.close < or_high:
            events.append(
                _event(
                    plan,
                    event_type="setup_invalidated",
                    bar=bar,
                    level_price=or_high,
                    observed_price=bar.close,
                    message="Long breakout failed by closing back inside the opening range.",
                    severity="warning",
                    actionability="invalidated",
                    dedupe_key=f"long-invalidated:{or_high:.2f}",
                )
            )
            long_active = False
        if short_active and or_low is not None and bar.close > or_low:
            events.append(
                _event(
                    plan,
                    event_type="setup_invalidated",
                    bar=bar,
                    level_price=or_low,
                    observed_price=bar.close,
                    message="Short breakdown failed by closing back inside the opening range.",
                    severity="warning",
                    actionability="invalidated",
                    dedupe_key=f"short-invalidated:{or_low:.2f}",
                )
            )
            short_active = False

    return _dedupe_events(events)


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
                    message=f"{side} swing entry triggered near ${entry:.2f}.",
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
                    message=f"Swing target was reached near ${target:.2f}.",
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
                    message=f"Swing setup invalidated at the stop level near ${stop:.2f}.",
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
                        message="Current swing R/R degraded below 1R as price moved away from the original trigger.",
                        severity="warning",
                        actionability="missed",
                        dedupe_key=f"rr-degraded:{side.lower()}:{entry:.2f}",
                    )
                )
                active = False
    return _dedupe_events(events)


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
) -> TradeReplayEvent:
    return TradeReplayEvent(
        id=f"evt-{uuid.uuid4().hex}",
        plan_id=plan.plan_id,
        symbol=plan.symbol,
        event_date=plan.plan_date,
        workflow=plan.workflow,
        event_type=event_type,
        event_time=bar.timestamp,
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
