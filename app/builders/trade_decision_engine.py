from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.models.trade_decision_models import (
    ScoreBucket,
    TradeAction,
    TradeDecision,
    TradeDecisionRegime,
    TradeEnvironment,
    TradeVerdict,
)
from ranking_pipeline.regime.constants import (
    REGIME_HIGH_VOL_CHOP,
    REGIME_RISK_OFF,
    REGIME_RISK_ON_CHOP,
    REGIME_RISK_ON_TREND,
)

PatternReliability = Literal["low", "medium", "high"]

WEIGHT_RS = 0.30
WEIGHT_EARLY_MOVER = 0.20
WEIGHT_BREAKOUT = 0.25
WEIGHT_SR = 0.15
WEIGHT_PATTERN = 0.10

REJECTION_REGIME = "Neutral or unfavorable regime"
REJECTION_BREAKOUT = "Weak breakout quality"
REJECTION_RS = "Weak relative strength"
REJECTION_SR = "Poor S/R structure"
REJECTION_PATTERN = "Low pattern reliability"
REJECTION_EARLY = "Weak early-mover / trend inflection signals"
REJECTION_SCORE = "Trade quality below actionable threshold"


@dataclass(frozen=True)
class TradeDecisionInputs:
    symbol: str
    as_of_date: str | None
    regime_id: str | None
    market_breadth_pct: float | None
    rs_percentile: float | None
    rs_score_0_1: float | None
    rs_21d: float | None
    rs_63d: float | None
    vol_ratio_20d: float | None
    dist_52w_high_pct: float | None
    near_52w_high: bool
    trend_acceleration: bool
    breakout_quality_score: int
    support_resistance_confidence: int
    pattern_reliability: PatternReliability
    ranking_rank: int | None
    universe_rank_count: int | None


@dataclass(frozen=True)
class _ScoreComponents:
    rs: float
    early_mover: float
    breakout: float
    support_resistance: float
    pattern: float


def evaluate_trade_decision(inputs: TradeDecisionInputs) -> TradeDecision:
    regime = _regime_gate(inputs.regime_id, inputs.market_breadth_pct)
    symbol = inputs.symbol.upper()

    if regime.trade_environment == "AVOID":
        return _compile(
            symbol=symbol,
            as_of_date=inputs.as_of_date,
            regime=regime,
            score=0,
            components=None,
            inputs=inputs,
            rs_pct=None,
        )

    rs_pct = _resolve_rs_percentile(inputs)
    components = _score_components(inputs, rs_pct)
    score = _compute_trade_quality_score(components)
    return _compile(
        symbol=symbol,
        as_of_date=inputs.as_of_date,
        regime=regime,
        score=score,
        components=components,
        inputs=inputs,
        rs_pct=rs_pct,
    )


def _compile(
    *,
    symbol: str,
    as_of_date: str | None,
    regime: TradeDecisionRegime,
    score: int,
    components: _ScoreComponents | None,
    inputs: TradeDecisionInputs,
    rs_pct: float | None,
) -> TradeDecision:
    bucket = _bucket_from_score(score)
    verdict = _verdict_from_score(score) if regime.trade_environment != "AVOID" else "NO_TRADE"
    if regime.trade_environment == "AVOID":
        bucket = "NO_TRADE"
        verdict = "NO_TRADE"
        score = 0

    action = _action_from_verdict(verdict)
    rejection = _primary_rejection_reason(
        regime=regime,
        verdict=verdict,
        inputs=inputs,
        rs_pct=rs_pct,
        components=components,
        score=score,
    )

    return TradeDecision(
        symbol=symbol,
        as_of_date=as_of_date,
        regime=regime,
        trade_quality_score=score,
        score_bucket=bucket,
        verdict=verdict,
        action=action,
        primary_rejection_reason=rejection,
    )


def _regime_gate(
    regime_id: str | None,
    breadth_pct: float | None,
) -> TradeDecisionRegime:
    rid = (regime_id or "").lower()
    if rid == REGIME_RISK_ON_TREND:
        env: TradeEnvironment = "FAVORABLE"
    elif rid in {REGIME_RISK_ON_CHOP, ""}:
        env = "NEUTRAL"
    elif rid in {REGIME_RISK_OFF, REGIME_HIGH_VOL_CHOP}:
        env = "AVOID"
    else:
        env = "NEUTRAL"

    if breadth_pct is not None and breadth_pct < 35 and env == "FAVORABLE":
        env = "NEUTRAL"
    if breadth_pct is not None and breadth_pct < 25:
        env = "AVOID"

    return TradeDecisionRegime(regime_id=regime_id, trade_environment=env)


def _resolve_rs_percentile(inputs: TradeDecisionInputs) -> float | None:
    if inputs.rs_percentile is not None:
        return max(0.0, min(100.0, inputs.rs_percentile))
    if inputs.ranking_rank is not None and inputs.universe_rank_count:
        n = max(1, inputs.universe_rank_count)
        rank = max(1, min(inputs.ranking_rank, n))
        return round(100.0 * (1.0 - (rank - 1) / n), 1)
    if inputs.rs_score_0_1 is not None:
        return round(max(0.0, min(1.0, inputs.rs_score_0_1)) * 100.0, 1)
    return None


def _rs_component(inputs: TradeDecisionInputs, rs_pct: float | None) -> float:
    if rs_pct is not None:
        return rs_pct
    if inputs.rs_score_0_1 is not None:
        return max(0.0, min(1.0, inputs.rs_score_0_1)) * 100.0
    return 50.0


def _early_mover_score(inputs: TradeDecisionInputs) -> float:
    score = 0.0
    rs21 = inputs.rs_21d
    rs63 = inputs.rs_63d
    if rs21 is not None and rs63 is not None and rs21 > rs63 + 0.02:
        score += 0.35
    elif rs21 is not None and rs21 > 0.03:
        score += 0.2

    vol = inputs.vol_ratio_20d
    if vol is not None and vol >= 1.35:
        score += 0.3
    elif vol is not None and vol >= 1.15:
        score += 0.15

    if inputs.trend_acceleration:
        score += 0.2
    if inputs.near_52w_high and (inputs.dist_52w_high_pct or 1) > 0.03:
        score += 0.15

    return min(1.0, score)


def _pattern_component(reliability: PatternReliability) -> float:
    return {"high": 85.0, "medium": 60.0, "low": 35.0}[reliability]


def _score_components(
    inputs: TradeDecisionInputs,
    rs_pct: float | None,
) -> _ScoreComponents:
    return _ScoreComponents(
        rs=_rs_component(inputs, rs_pct),
        early_mover=_early_mover_score(inputs) * 100.0,
        breakout=float(max(0, min(100, inputs.breakout_quality_score))),
        support_resistance=float(max(0, min(100, inputs.support_resistance_confidence))),
        pattern=_pattern_component(inputs.pattern_reliability),
    )


def _compute_trade_quality_score(components: _ScoreComponents) -> int:
    raw = (
        WEIGHT_RS * components.rs
        + WEIGHT_EARLY_MOVER * components.early_mover
        + WEIGHT_BREAKOUT * components.breakout
        + WEIGHT_SR * components.support_resistance
        + WEIGHT_PATTERN * components.pattern
    )
    return int(round(max(0.0, min(100.0, raw))))


def _bucket_from_score(score: int) -> ScoreBucket:
    if score >= 80:
        return "TRADE"
    if score >= 60:
        return "SETUP"
    if score >= 40:
        return "WATCHLIST"
    return "NO_TRADE"


def _verdict_from_score(score: int) -> TradeVerdict:
    if score >= 80:
        return "TRADE"
    if score >= 40:
        return "WATCHLIST"
    return "NO_TRADE"


def _action_from_verdict(verdict: TradeVerdict) -> TradeAction:
    if verdict == "TRADE":
        return "ENTER"
    if verdict == "WATCHLIST":
        return "WAIT_FOR_SETUP"
    return "AVOID"


def _primary_rejection_reason(
    *,
    regime: TradeDecisionRegime,
    verdict: TradeVerdict,
    inputs: TradeDecisionInputs,
    rs_pct: float | None,
    components: _ScoreComponents | None,
    score: int,
) -> str | None:
    if verdict == "TRADE":
        return None

    if regime.trade_environment == "AVOID":
        return REJECTION_REGIME

    candidates: list[tuple[float, str]] = []

    breakout = float(max(0, min(100, inputs.breakout_quality_score)))
    if breakout < 45:
        gap = (45.0 - breakout) * WEIGHT_BREAKOUT
        candidates.append((gap, REJECTION_BREAKOUT))

    rs_val = rs_pct if rs_pct is not None else (components.rs if components else 50.0)
    if rs_val < 55:
        gap = (55.0 - rs_val) * WEIGHT_RS
        candidates.append((gap, REJECTION_RS))

    sr = float(max(0, min(100, inputs.support_resistance_confidence)))
    if sr < 50:
        gap = (50.0 - sr) * WEIGHT_SR
        candidates.append((gap, REJECTION_SR))

    if inputs.pattern_reliability == "low":
        candidates.append((8.0, REJECTION_PATTERN))

    early = components.early_mover if components else _early_mover_score(inputs) * 100.0
    if early < 45:
        gap = (45.0 - early) * WEIGHT_EARLY_MOVER
        candidates.append((gap, REJECTION_EARLY))

    if regime.trade_environment == "NEUTRAL":
        candidates.append((6.0, REJECTION_REGIME))

    if score < 40 and not candidates:
        return REJECTION_SCORE

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    return REJECTION_SCORE


def compute_breakout_quality_score(
    *,
    volume_confirmed: bool,
    confirmed_breakout: bool,
    failed_breakout: bool,
    volume_ratio: float | None,
    volume_confirmation_score: float | None,
) -> int:
    score = 40.0
    if confirmed_breakout:
        score += 28.0
    if volume_confirmed:
        score += 15.0
    if volume_ratio is not None:
        if volume_ratio >= 1.5:
            score += 12.0
        elif volume_ratio >= 1.2:
            score += 6.0
    if volume_confirmation_score is not None:
        score += (volume_confirmation_score - 0.5) * 20.0
    if failed_breakout:
        score -= 35.0
    return int(max(0, min(100, round(score))))


def compute_sr_confidence(
    *,
    nearest_zone_strength: float | None,
    nearest_touches: int | None,
) -> int:
    if nearest_zone_strength is None:
        return 45
    base = nearest_zone_strength * 70.0
    touch_bonus = min(25, (nearest_touches or 0) * 5)
    return int(max(0, min(100, round(base + touch_bonus))))


def pattern_reliability_from_setup(
    occurrence_count: int | None,
    win_rate_5d: float | None,
) -> PatternReliability:
    n = occurrence_count or 0
    wr = win_rate_5d
    if n >= 8 and wr is not None and wr >= 0.55:
        return "high"
    if n >= 4 and wr is not None and wr >= 0.48:
        return "medium"
    return "low"


def inputs_from_chart_payload(chart: dict[str, Any] | None) -> tuple[int, int, bool, bool]:
    """Derive breakout score and S/R confidence from chart intelligence dict."""
    if not chart:
        return 35, 45, False, False

    events = chart.get("breakout_events") or chart.get("breakoutEvents") or []
    confirmed = any(
        (e.get("kind") or e.get("breakout_kind") or "") == "confirmed_breakout"
        for e in events
    )
    failed = any(
        (e.get("kind") or e.get("breakout_kind") or "") == "failed_breakout"
        for e in events
    )
    vol_ratio = None
    if events:
        last = events[-1]
        vol_ratio = last.get("volume_ratio") or last.get("volumeRatio")

    zones = (chart.get("support_zones") or chart.get("supportZones") or []) + (
        chart.get("resistance_zones") or chart.get("resistanceZones") or []
    )
    best_strength = None
    best_touches = None
    for z in zones:
        s = z.get("strength")
        t = z.get("touches")
        if s is None:
            continue
        if best_strength is None or float(s) > best_strength:
            best_strength = float(s)
            best_touches = int(t) if t is not None else 0

    sr = compute_sr_confidence(
        nearest_zone_strength=best_strength,
        nearest_touches=best_touches,
    )
    breakout = compute_breakout_quality_score(
        volume_confirmed=vol_ratio is not None and vol_ratio >= 1.2,
        confirmed_breakout=confirmed,
        failed_breakout=failed,
        volume_ratio=vol_ratio,
        volume_confirmation_score=None,
    )
    return breakout, sr, confirmed, failed
