from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.models.trade_decision_models import (
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
TrendStage = Literal["early", "mid", "late", "unknown"]

WEIGHT_RS = 0.30
WEIGHT_EARLY_MOVER = 0.20
WEIGHT_BREAKOUT = 0.25
WEIGHT_SR = 0.15
WEIGHT_PATTERN = 0.10


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


def evaluate_trade_decision(inputs: TradeDecisionInputs) -> TradeDecision:
    regime = _regime_gate(inputs.regime_id, inputs.market_breadth_pct)
    symbol = inputs.symbol.upper()

    if regime.trade_environment == "AVOID":
        explanation = _build_explanation(
            inputs=inputs,
            regime=regime,
            score=0,
            trend_stage=_trend_stage(inputs, None),
            rs_pct=None,
            verdict="NO_TRADE",
        )
        return TradeDecision(
            symbol=symbol,
            as_of_date=inputs.as_of_date,
            regime=regime,
            trade_quality_score=0,
            verdict="NO_TRADE",
            action="AVOID",
            explanation=explanation[:5],
        )

    rs_pct = _resolve_rs_percentile(inputs)
    trend_stage = _trend_stage(inputs, rs_pct)
    score = _compute_trade_quality_score(inputs, rs_pct)
    verdict = _verdict_from_score(score)
    action = _action_from_verdict(verdict)
    explanation = _build_explanation(
        inputs=inputs,
        regime=regime,
        score=score,
        trend_stage=trend_stage,
        rs_pct=rs_pct,
        verdict=verdict,
    )

    return TradeDecision(
        symbol=symbol,
        as_of_date=inputs.as_of_date,
        regime=regime,
        trade_quality_score=score,
        verdict=verdict,
        action=action,
        explanation=explanation[:5],
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


def _compute_trade_quality_score(
    inputs: TradeDecisionInputs,
    rs_pct: float | None,
) -> int:
    rs = _rs_component(inputs, rs_pct)
    early = _early_mover_score(inputs) * 100.0
    breakout = float(max(0, min(100, inputs.breakout_quality_score)))
    sr = float(max(0, min(100, inputs.support_resistance_confidence)))
    pattern = _pattern_component(inputs.pattern_reliability)

    raw = (
        WEIGHT_RS * rs
        + WEIGHT_EARLY_MOVER * early
        + WEIGHT_BREAKOUT * breakout
        + WEIGHT_SR * sr
        + WEIGHT_PATTERN * pattern
    )
    return int(round(max(0.0, min(100.0, raw))))


def _verdict_from_score(score: int) -> TradeVerdict:
    if score >= 80:
        return "HIGH_CONVICTION_TRADE"
    if score >= 60:
        return "WATCHLIST"
    return "NO_TRADE"


def _action_from_verdict(verdict: TradeVerdict) -> TradeAction:
    if verdict == "HIGH_CONVICTION_TRADE":
        return "ENTER"
    if verdict == "WATCHLIST":
        return "WAIT_FOR_SETUP"
    return "AVOID"


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


def _regime_label(regime_id: str | None) -> str:
    mapping = {
        REGIME_RISK_ON_TREND: "Risk-on trend",
        REGIME_RISK_ON_CHOP: "Risk-on chop",
        REGIME_RISK_OFF: "Risk-off",
        REGIME_HIGH_VOL_CHOP: "High-volatility chop",
    }
    return mapping.get(regime_id or "", regime_id or "Unknown")


def _breakout_summary(score: int) -> str:
    if score >= 78:
        quality = "strong"
    elif score >= 55:
        quality = "moderate"
    elif score >= 35:
        quality = "weak"
    else:
        quality = "poor / failed"
    return f"Breakout quality: {quality} ({score}/100)."


def _build_explanation(
    *,
    inputs: TradeDecisionInputs,
    regime: TradeDecisionRegime,
    score: int,
    trend_stage: TrendStage,
    rs_pct: float | None,
    verdict: TradeVerdict,
) -> list[str]:
    env = regime.trade_environment
    regime_line = (
        f"Market regime: {_regime_label(regime.regime_id)} — gate {env.lower()}."
    )
    if env == "AVOID":
        regime_line += " New long exposure blocked."

    if rs_pct is not None:
        strength = f"Relative strength: ~{rs_pct:.0f}th percentile vs ranked universe."
    elif inputs.rs_score_0_1 is not None:
        strength = (
            f"Relative strength: model RS score {inputs.rs_score_0_1:.2f} "
            "(percentile rank unavailable)."
        )
    else:
        strength = "Relative strength: limited data for this symbol."

    stage_map = {
        "early": "early leadership / inflection",
        "mid": "confirmed mid-trend",
        "late": "late-stage extension",
        "unknown": "stage unclear",
    }
    stage_line = f"Trend stage: {stage_map[trend_stage]}."

    breakout_line = _breakout_summary(
        max(0, min(100, inputs.breakout_quality_score)),
    )

    risk_parts: list[str] = []
    if env == "AVOID":
        risk_parts.append("risk-off or high-volatility macro regime")
    elif env == "NEUTRAL":
        risk_parts.append("neutral macro regime")
    if inputs.support_resistance_confidence < 40:
        risk_parts.append("weak nearby support/resistance")
    if inputs.pattern_reliability == "low":
        risk_parts.append("low pattern sample reliability")
    if trend_stage == "late":
        risk_parts.append("late-trend extension risk")
    if inputs.breakout_quality_score < 35:
        risk_parts.append("failed or unconfirmed breakout")
    if not risk_parts:
        risk_parts.append("normal execution and structure risk")
    score_note = ""
    if verdict == "NO_TRADE" and env != "AVOID":
        score_note = f" Trade quality {score}/100 — below threshold."
    elif verdict == "WATCHLIST":
        score_note = f" Trade quality {score}/100 — forming, not entry-ready."
    elif verdict == "HIGH_CONVICTION_TRADE":
        score_note = f" Trade quality {score}/100 — passes filter."
    risk_line = f"Key risk: {risk_parts[0]}.{score_note}"

    return [regime_line, strength, stage_line, breakout_line, risk_line]


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
