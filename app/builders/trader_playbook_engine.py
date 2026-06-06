from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.trade_decision_models import TradeDecision
from app.models.trader_playbook_models import (
    TraderPlaybookAlignment,
    TraderPlaybookConditions,
    TraderPlaybookLevels,
    TraderPlaybookResponse,
    TraderPlaybookRisk,
)
from app.models.trading_bias_models import TradingBiasResponse


@dataclass(frozen=True)
class TraderPlaybookInputs:
    symbol: str
    trading_bias: TradingBiasResponse | None = None
    trade_decision: TradeDecision | None = None
    pattern_intelligence: dict[str, Any] | None = None
    catalyst: str = "none"
    data_gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ExtractedPattern:
    close: float | None
    trend_bias: str
    support: float | None
    resistance: float | None
    failed_breakout: bool
    confirmed_breakout: bool
    volume_confirmed: bool
    price_structure_alignment: str
    reasons: tuple[str, ...] = ()


def evaluate_trader_playbook(inputs: TraderPlaybookInputs) -> TraderPlaybookResponse:
    data_gaps = _dedupe(inputs.data_gaps)
    warnings = _dedupe(inputs.warnings)
    direction = inputs.trading_bias.bias if inputs.trading_bias else "Neutral"
    confidence = inputs.trading_bias.confidence if inputs.trading_bias else "Low"
    pattern = _extract_pattern(inputs.pattern_intelligence)

    if inputs.trading_bias is None:
        data_gaps = _with_gap(data_gaps, "Trading bias unavailable")
    if inputs.trade_decision is None:
        data_gaps = _with_gap(data_gaps, "Execution readiness unavailable")
    if inputs.pattern_intelligence is None:
        data_gaps = _with_gap(data_gaps, "Pattern analysis unavailable")
        return _no_setup_response(
            direction=direction,
            confidence=confidence,
            data_gaps=data_gaps,
            warnings=warnings,
            alignment=_alignment(inputs, pattern),
            reasons=[],
        )

    setup = _select_setup(direction=direction, pattern=pattern)
    levels, conditions, reasons = _build_plan(
        setup=setup,
        direction=direction,
        pattern=pattern,
    )
    levels = _enforce_level_guardrails(levels)
    risk = _risk_from_levels(levels, direction=_risk_direction(setup, direction, levels))
    status = _status_for_plan(
        setup=setup,
        direction=direction,
        levels=levels,
        risk=risk,
        pattern=pattern,
    )

    if risk.risk_reward_label == "poor":
        warnings = _with_gap(warnings, "Risk/reward is poor for this daily setup.")
        if status == "Valid":
            status = "Waiting"
    if setup == "None":
        status = "NoSetup"
    if levels.entry is None or levels.stop is None:
        status = "NoSetup" if setup == "None" else "Waiting"
        if setup != "None":
            warnings = _with_gap(
                warnings,
                "No complete entry/stop plan is available from daily levels.",
            )

    return TraderPlaybookResponse(
        direction=direction,
        confidence=confidence,
        best_setup=setup,
        status=status,
        conditions=conditions,
        levels=levels,
        risk=risk,
        alignment=_alignment(inputs, pattern),
        reasons=_dedupe(list(reasons) + list(pattern.reasons))[:5],
        warnings=warnings,
        data_gaps=data_gaps,
    )


def _no_setup_response(
    *,
    direction: str,
    confidence: str,
    data_gaps: list[str],
    warnings: list[str],
    alignment: TraderPlaybookAlignment,
    reasons: list[str],
) -> TraderPlaybookResponse:
    return TraderPlaybookResponse(
        direction=direction,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        best_setup="None",
        status="NoSetup",
        conditions=TraderPlaybookConditions(
            valid_if=[],
            invalid_if=["Pattern analysis must be available before forming a plan."],
        ),
        levels=TraderPlaybookLevels(),
        risk=TraderPlaybookRisk(),
        alignment=alignment,
        reasons=reasons,
        warnings=warnings,
        data_gaps=data_gaps,
    )


def _select_setup(*, direction: str, pattern: _ExtractedPattern) -> str:
    if pattern.failed_breakout and direction in {"Bearish", "Neutral"}:
        return "FailedBreakout"
    if direction == "Bullish":
        if pattern.resistance is not None and pattern.close is not None:
            if pattern.close >= pattern.resistance:
                return "BreakoutContinuation"
            if _near(pattern.close, pattern.resistance, pct=0.03):
                return "BreakoutContinuation"
        if pattern.support is not None and pattern.close is not None:
            if _near(pattern.close, pattern.support, pct=0.05):
                return "PullbackToSupport"
        if pattern.trend_bias == "uptrend":
            return "TrendContinuation"
    if direction == "Bearish":
        if pattern.failed_breakout:
            return "FailedBreakout"
        if pattern.trend_bias == "downtrend":
            return "TrendContinuation"
    if direction == "Neutral":
        if pattern.support is not None and pattern.resistance is not None:
            return "RangeDay"
    return "None"


def _build_plan(
    *,
    setup: str,
    direction: str,
    pattern: _ExtractedPattern,
) -> tuple[TraderPlaybookLevels, TraderPlaybookConditions, list[str]]:
    close = pattern.close
    support = pattern.support
    resistance = pattern.resistance
    breakout = resistance
    reasons: list[str] = []
    valid_if: list[str] = []
    invalid_if: list[str] = []
    entry = None
    stop = None
    target1 = None
    target2 = None

    if setup == "BreakoutContinuation" and breakout is not None:
        entry = breakout
        stop = support if support is not None else _below(breakout, 0.03)
        target1 = _target_from_r(entry, stop, direction="Bullish", r=2.0)
        target2 = _target_from_r(entry, stop, direction="Bullish", r=3.0)
        valid_if.append(f"Daily close holds above breakout level ${breakout:.2f}.")
        invalid_if.append(f"Setup invalidates on a close back below ${stop:.2f}.")
        reasons.append("Best daily setup is a breakout continuation.")
        if close is not None and close < breakout:
            reasons.append("Breakout trigger has not crossed yet.")

    elif setup == "PullbackToSupport" and support is not None:
        entry = support
        stop = _below(support, 0.03)
        if resistance is not None:
            target1 = resistance
        else:
            target1 = _target_from_r(entry, stop, direction="Bullish", r=2.0)
            target2 = _target_from_r(entry, stop, direction="Bullish", r=3.0)
        valid_if.append(f"Price holds or reclaims support near ${support:.2f}.")
        invalid_if.append(f"Setup invalidates on a close below ${stop:.2f}.")
        reasons.append("Best daily setup is a pullback into support.")

    elif setup == "FailedBreakout" and resistance is not None:
        entry = resistance
        stop = _above(resistance, 0.03)
        if support is not None:
            target1 = support
        else:
            target1 = _target_from_r(entry, stop, direction="Bearish", r=2.0)
            target2 = _target_from_r(entry, stop, direction="Bearish", r=3.0)
        valid_if.append(f"Price remains below rejected resistance ${resistance:.2f}.")
        invalid_if.append(f"Setup invalidates if price reclaims ${stop:.2f}.")
        reasons.append("Pattern evidence flags a failed breakout/rejection.")

    elif setup == "RangeDay" and support is not None and resistance is not None:
        entry = support if close is not None and close <= (support + resistance) / 2 else resistance
        if entry == support:
            stop = _below(support, 0.025)
            target1 = resistance
            valid_if.append(f"Range support holds near ${support:.2f}.")
            invalid_if.append(f"Range invalidates below ${stop:.2f}.")
        else:
            stop = _above(resistance, 0.025)
            target1 = support
            valid_if.append(f"Range resistance rejects near ${resistance:.2f}.")
            invalid_if.append(f"Range invalidates above ${stop:.2f}.")
        reasons.append("Mixed evidence favors a range-bound daily plan.")

    elif setup == "TrendContinuation":
        if direction == "Bearish":
            entry = close or resistance
            stop = resistance
            target1 = support
            if entry is not None and stop is not None and target1 is None:
                target2 = _target_from_r(entry, stop, direction="Bearish", r=2.0)
            valid_if.append("Downtrend continues below recent daily structure.")
            if stop is not None:
                invalid_if.append(f"Bearish plan invalidates above ${stop:.2f}.")
        else:
            entry = close or support
            stop = support
            target1 = resistance
            if entry is not None and stop is not None and target1 is None:
                target2 = _target_from_r(entry, stop, direction="Bullish", r=2.0)
            valid_if.append("Uptrend continues while price holds higher lows.")
            if stop is not None:
                invalid_if.append(f"Bullish plan invalidates below ${stop:.2f}.")
        reasons.append("Trend structure supports continuation.")

    conditions = TraderPlaybookConditions(valid_if=valid_if, invalid_if=invalid_if)
    levels = TraderPlaybookLevels(
        entry=_round(entry),
        stop=_round(stop),
        target1=_round(target1),
        target2=_round(target2),
        support=_round(support),
        resistance=_round(resistance),
        breakout_level=_round(breakout),
    )
    return levels, conditions, reasons


def _status_for_plan(
    *,
    setup: str,
    direction: str,
    levels: TraderPlaybookLevels,
    risk: TraderPlaybookRisk,
    pattern: _ExtractedPattern,
) -> str:
    close = pattern.close
    if setup == "None":
        return "NoSetup"
    if close is None or levels.entry is None or levels.stop is None:
        return "Waiting"
    if direction == "Bullish" and close <= levels.stop:
        return "Invalid"
    if direction == "Bearish" and close >= levels.stop:
        return "Invalid"
    if risk.risk_reward_label in {"poor", "unavailable"}:
        return "Waiting"
    if setup == "BreakoutContinuation":
        return "Valid" if levels.breakout_level is not None and close >= levels.breakout_level else "Waiting"
    if setup == "FailedBreakout":
        return "Valid" if levels.resistance is not None and close < levels.resistance else "Waiting"
    if setup == "PullbackToSupport":
        return "Valid" if levels.support is not None and close >= levels.support else "Waiting"
    if setup == "TrendContinuation":
        return "Valid"
    return "Waiting"


def _risk_from_levels(levels: TraderPlaybookLevels, *, direction: str) -> TraderPlaybookRisk:
    if levels.entry is None or levels.stop is None:
        return TraderPlaybookRisk()
    target1 = levels.target1
    target2 = levels.target2
    if target1 is None and target2 is None:
        return TraderPlaybookRisk()

    risk_per_share = abs(levels.entry - levels.stop)
    if risk_per_share <= 0:
        return TraderPlaybookRisk()

    reward1 = _reward(levels.entry, target1, direction)
    reward2 = _reward(levels.entry, target2, direction)
    r1 = reward1 / risk_per_share if reward1 is not None else None
    r2 = reward2 / risk_per_share if reward2 is not None else None
    best_r = max([value for value in (r1, r2) if value is not None], default=None)
    if best_r is None:
        label = "unavailable"
    elif best_r >= 2.0:
        label = "favorable"
    elif best_r >= 1.2:
        label = "mixed"
    else:
        label = "poor"

    return TraderPlaybookRisk(
        risk_per_share=_round(risk_per_share),
        reward_to_target1=_round(reward1),
        reward_to_target2=_round(reward2),
        r_multiple_target1=_round(r1),
        r_multiple_target2=_round(r2),
        risk_reward_label=label,  # type: ignore[arg-type]
    )


def _risk_direction(
    setup: str,
    direction: str,
    levels: TraderPlaybookLevels,
) -> str:
    if setup == "FailedBreakout":
        return "Bearish"
    if setup == "RangeDay" and levels.entry is not None and levels.target1 is not None:
        return "Bearish" if levels.target1 < levels.entry else "Bullish"
    return direction


def _extract_pattern(payload: dict[str, Any] | None) -> _ExtractedPattern:
    if not payload:
        return _ExtractedPattern(
            close=None,
            trend_bias="",
            support=None,
            resistance=None,
            failed_breakout=False,
            confirmed_breakout=False,
            volume_confirmed=False,
            price_structure_alignment="unavailable",
        )

    trend = _get(payload, "trend_context", "trendContext") or {}
    scores = _get(payload, "scores") or {}
    chart = _get(payload, "chart_intelligence", "chartIntelligence") or {}
    support = _zone_price(_get(chart, "support_zones", "supportZones") or [], "support")
    resistance = _zone_price(
        _get(chart, "resistance_zones", "resistanceZones") or [],
        "resistance",
    )
    events = _get(chart, "breakout_events", "breakoutEvents") or []
    failed = _has_breakout_event(events, "failed_breakout")
    confirmed = _has_breakout_event(events, "confirmed_breakout")
    trend_bias = str(_get(trend, "trend_bias", "trendBias") or "")
    volume_score = _float(_get(scores, "volume_confirmation", "volumeConfirmation"))
    volume_confirmed = (volume_score or 0.0) >= 0.65
    alignment = _price_structure_alignment(trend_bias, payload)
    reasons = []
    if confirmed:
        reasons.append("Pattern evidence includes a confirmed breakout.")
    if failed:
        reasons.append("Pattern evidence includes a failed breakout.")
    if volume_confirmed:
        reasons.append("Daily volume confirmation is constructive.")

    return _ExtractedPattern(
        close=_float(_get(trend, "close")),
        trend_bias=trend_bias.lower(),
        support=support,
        resistance=resistance,
        failed_breakout=failed,
        confirmed_breakout=confirmed,
        volume_confirmed=volume_confirmed,
        price_structure_alignment=alignment,
        reasons=tuple(reasons),
    )


def _alignment(
    inputs: TraderPlaybookInputs,
    pattern: _ExtractedPattern,
) -> TraderPlaybookAlignment:
    bias_alignment = "mixed"
    market = "unavailable"
    rs = "unavailable"
    catalyst = inputs.catalyst
    if inputs.trading_bias is not None:
        bias_alignment = "aligned" if inputs.trading_bias.bias in {"Bullish", "Bearish"} else "mixed"
        market = inputs.trading_bias.alignment.market_regime
        rs = inputs.trading_bias.alignment.relative_strength
        catalyst = inputs.trading_bias.alignment.catalyst

    execution = "watch"
    if inputs.trade_decision is not None:
        if inputs.trade_decision.action == "ENTER":
            execution = "ready"
        elif inputs.trade_decision.action == "AVOID":
            execution = "avoid"

    return TraderPlaybookAlignment(
        daily_bias=bias_alignment,  # type: ignore[arg-type]
        execution_readiness=execution,  # type: ignore[arg-type]
        market_regime=market,  # type: ignore[arg-type]
        relative_strength=rs,  # type: ignore[arg-type]
        price_structure=pattern.price_structure_alignment,  # type: ignore[arg-type]
        catalyst=catalyst,  # type: ignore[arg-type]
    )


def _price_structure_alignment(trend_bias: str, payload: dict[str, Any]) -> str:
    chart = _get(payload, "chart_intelligence", "chartIntelligence") or {}
    summary = _get(chart, "summary") or {}
    outlook = str(_get(_get(summary, "outlook") or {}, "label") or "").lower()
    trend = trend_bias.lower()
    if trend == "uptrend" or "bullish" in outlook or "constructive" in outlook:
        return "aligned"
    if trend == "downtrend" or "bearish" in outlook:
        return "against"
    return "mixed"


def _enforce_level_guardrails(levels: TraderPlaybookLevels) -> TraderPlaybookLevels:
    if levels.entry is None or levels.stop is None:
        return levels.model_copy(update={"target1": None, "target2": None})
    return levels


def _reward(entry: float, target: float | None, direction: str) -> float | None:
    if target is None:
        return None
    reward = target - entry if direction != "Bearish" else entry - target
    return reward if reward > 0 else 0.0


def _target_from_r(entry: float, stop: float, *, direction: str, r: float) -> float:
    risk = abs(entry - stop)
    return entry - risk * r if direction == "Bearish" else entry + risk * r


def _below(value: float, pct: float) -> float:
    return value * (1.0 - pct)


def _above(value: float, pct: float) -> float:
    return value * (1.0 + pct)


def _near(price: float, level: float, *, pct: float) -> bool:
    if price <= 0 or level <= 0:
        return False
    return abs(price - level) / price <= pct


def _zone_price(zones: list[Any], kind: str) -> float | None:
    if not zones:
        return None
    zone = zones[0]
    if not isinstance(zone, dict):
        return None
    key = ("price_high", "priceHigh") if kind == "support" else ("price_low", "priceLow")
    value = _float(_get(zone, *key))
    if value is None:
        value = _float(_get(zone, "price", "level"))
    return _round(value)


def _has_breakout_event(events: list[Any], kind: str) -> bool:
    for event in events[-5:]:
        if not isinstance(event, dict):
            continue
        event_kind = str(_get(event, "kind", "breakout_kind", "breakoutKind") or "")
        if event_kind.lower() == kind:
            return True
    return False


def _get(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _with_gap(values: list[str], value: str) -> list[str]:
    if value not in values:
        return values + [value]
    return values
