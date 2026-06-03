from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.models.trade_decision_models import (
    ScoreBucket,
    TradeAction,
    TradeDecision,
    TradeDecisionReasonBreakdown,
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

TrendStage = Literal["early", "mid", "late", "unknown"]


@dataclass(frozen=True)
class _ExecutionFactor:
    factor_id: str
    weighted_gap: float
    line: str


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
    rs_pct = _resolve_rs_percentile(inputs)
    components = _score_components(inputs, rs_pct)

    if regime.trade_environment == "AVOID":
        return _compile(
            symbol=symbol,
            as_of_date=inputs.as_of_date,
            regime=regime,
            score=0,
            components=components,
            inputs=inputs,
            rs_pct=rs_pct,
        )

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
    if components is None:
        components = _score_components(inputs, rs_pct)
    reason_breakdown = _build_reason_breakdown(
        regime=regime,
        score=score,
        components=components,
        inputs=inputs,
        rs_pct=rs_pct,
    )

    return TradeDecision(
        symbol=symbol,
        as_of_date=as_of_date,
        regime=regime,
        trade_quality_score=score,
        score_bucket=bucket,
        verdict=verdict,
        action=action,
        reason_breakdown=reason_breakdown,
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


def _regime_label(regime_id: str | None) -> str:
    mapping = {
        REGIME_RISK_ON_TREND: "risk_on_trend",
        REGIME_RISK_ON_CHOP: "risk_on_chop",
        REGIME_RISK_OFF: "risk_off",
        REGIME_HIGH_VOL_CHOP: "high_vol_chop",
    }
    return mapping.get(regime_id or "", regime_id or "unknown")


def _trend_stage(inputs: TradeDecisionInputs, rs_pct: float | None) -> TrendStage:
    early = _early_mover_score(inputs) >= 0.45
    at_highs = (inputs.dist_52w_high_pct is not None and inputs.dist_52w_high_pct <= 0.02) or (
        inputs.near_52w_high and not early
    )
    if early and rs_pct is not None and rs_pct < 80:
        return "early"
    if at_highs or (rs_pct is not None and rs_pct >= 85):
        return "late"
    if rs_pct is not None and rs_pct >= 55:
        return "mid"
    return "unknown"


def _execution_factors(
    components: _ScoreComponents,
    inputs: TradeDecisionInputs,
    rs_pct: float | None,
) -> list[_ExecutionFactor]:
    rs_val = rs_pct if rs_pct is not None else components.rs
    factors: list[_ExecutionFactor] = []

    breakout = components.breakout
    breakout_adj = "Weak" if breakout < 55 else "Moderate"
    factors.append(
        _ExecutionFactor(
            "breakout",
            WEIGHT_BREAKOUT * (100.0 - breakout),
            f"{breakout_adj} breakout quality ({int(round(breakout))}/100)",
        )
    )

    rs_adj = "Weak" if rs_val < 50 else "Moderate"
    if rs_pct is not None:
        rs_line = f"{rs_adj} relative strength ({rs_pct:.0f}th percentile)"
    else:
        rs_line = f"{rs_adj} relative strength (score {rs_val:.0f}/100)"
    factors.append(
        _ExecutionFactor("rs", WEIGHT_RS * (100.0 - rs_val), rs_line),
    )

    sr = components.support_resistance
    sr_adj = "Low" if sr < 50 else "Moderate"
    factors.append(
        _ExecutionFactor(
            "sr",
            WEIGHT_SR * (100.0 - sr),
            f"{sr_adj} support/resistance strength ({int(round(sr))}/100)",
        )
    )

    pattern = components.pattern
    reliability = inputs.pattern_reliability
    pattern_adj = "Poor" if reliability == "low" else "Moderate"
    factors.append(
        _ExecutionFactor(
            "pattern",
            WEIGHT_PATTERN * (100.0 - pattern),
            f"{pattern_adj} pattern reliability ({reliability})",
        )
    )

    early = components.early_mover
    early_adj = "Weak" if early < 45 else "Moderate"
    factors.append(
        _ExecutionFactor(
            "early_mover",
            WEIGHT_EARLY_MOVER * (100.0 - early),
            f"{early_adj} early-mover / trend inflection ({int(round(early))}/100)",
        )
    )

    return factors


def _primary_weakness(
    components: _ScoreComponents,
    inputs: TradeDecisionInputs,
    rs_pct: float | None,
) -> str:
    factors = _execution_factors(components, inputs, rs_pct)
    core = [f for f in factors if f.factor_id in {"breakout", "rs", "sr", "pattern"}]
    return max(core, key=lambda item: item.weighted_gap).line


def _hard_blockers(regime: TradeDecisionRegime, score: int) -> list[str]:
    blockers: list[str] = []
    if regime.trade_environment == "AVOID":
        blockers.append(
            f"Unfavorable regime gate ({_regime_label(regime.regime_id)})"
        )
    if score < 40:
        blockers.append(f"Trade quality score below threshold ({score}/100)")
    return blockers


def _secondary_factors(
    *,
    regime: TradeDecisionRegime,
    primary_id: str,
    components: _ScoreComponents,
    inputs: TradeDecisionInputs,
    rs_pct: float | None,
    score: int,
) -> list[str]:
    secondary: list[str] = []

    if regime.trade_environment == "NEUTRAL":
        secondary.append(f"Neutral regime ({_regime_label(regime.regime_id)})")
    elif regime.trade_environment == "FAVORABLE":
        secondary.append(f"Favorable regime ({_regime_label(regime.regime_id)})")

    stage = _trend_stage(inputs, rs_pct)
    stage_lines = {
        "early": "Early-trend / inflection structure",
        "mid": "Mid-trend structure",
        "late": "Late-stage extension structure",
    }
    if stage in stage_lines and primary_id not in {"early_mover", "rs"}:
        secondary.append(stage_lines[stage])
    elif stage == "mid" and primary_id == "rs" and stage_lines["mid"] not in secondary:
        secondary.append(stage_lines["mid"])

    if rs_pct is not None and primary_id != "rs":
        if rs_pct >= 70:
            secondary.append(f"Strong RS percentile ({rs_pct:.0f}th)")
        elif rs_pct >= 55:
            secondary.append(f"Mid-trend RS strength ({rs_pct:.0f}th percentile)")
        elif rs_pct >= 45:
            secondary.append(f"Moderate RS percentile ({rs_pct:.0f}th)")

    if primary_id != "pattern" and inputs.pattern_reliability == "medium":
        secondary.append("Moderate pattern reliability")

    if primary_id != "breakout" and 55 <= components.breakout < 70:
        secondary.append(
            f"Breakout not yet confirming ({int(round(components.breakout))}/100)"
        )

    if primary_id != "early_mover" and components.early_mover < 55:
        secondary.append(
            f"Limited early-mover signals ({int(round(components.early_mover))}/100)"
        )

    if primary_id != "sr" and components.support_resistance >= 55:
        secondary.append(
            f"Support/resistance acceptable ({int(round(components.support_resistance))}/100)"
        )

    if score >= 80 and not secondary:
        secondary.append("Execution factors align across RS, breakout, and structure")

    deduped: list[str] = []
    for item in secondary:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]


def _build_reason_breakdown(
    *,
    regime: TradeDecisionRegime,
    score: int,
    components: _ScoreComponents,
    inputs: TradeDecisionInputs,
    rs_pct: float | None,
) -> TradeDecisionReasonBreakdown:
    factors = _execution_factors(components, inputs, rs_pct)
    core_factors = [f for f in factors if f.factor_id in {"breakout", "rs", "sr", "pattern"}]
    primary_factor = max(core_factors, key=lambda item: item.weighted_gap)

    return TradeDecisionReasonBreakdown(
        hard_blockers=_hard_blockers(regime, score),
        primary_weakness=_primary_weakness(components, inputs, rs_pct),
        secondary_factors=_secondary_factors(
            regime=regime,
            primary_id=primary_factor.factor_id,
            components=components,
            inputs=inputs,
            rs_pct=rs_pct,
            score=score,
        ),
    )


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
