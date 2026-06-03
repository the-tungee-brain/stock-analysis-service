from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.trade_decision_models import (
    ActionHint,
    BreakoutGrade,
    OpportunityGrade,
    PatternReliability,
    SetupGrade,
    TradeDecision,
    TradeDecisionStageOpportunity,
    TradeDecisionStageRegime,
    TradeDecisionStageSetup,
    TradeEnvironment,
    TradeVerdict,
    TrendStage,
)
from ranking_pipeline.regime.constants import (
    REGIME_HIGH_VOL_CHOP,
    REGIME_RISK_OFF,
    REGIME_RISK_ON_CHOP,
    REGIME_RISK_ON_TREND,
)


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
    regime = _stage_regime(inputs.regime_id, inputs.market_breadth_pct)
    opportunity = _stage_opportunity(inputs)
    setup = _stage_setup(inputs)

    verdict, action = _final_verdict(regime, opportunity, setup)
    explanation = _build_explanation(
        inputs=inputs,
        regime=regime,
        opportunity=opportunity,
        setup=setup,
        verdict=verdict,
    )

    return TradeDecision(
        symbol=inputs.symbol.upper(),
        as_of_date=inputs.as_of_date,
        regime=regime,
        opportunity=opportunity,
        setup=setup,
        verdict=verdict,
        action_hint=action,
        explanation=explanation[:5],
    )


def _stage_regime(
    regime_id: str | None,
    breadth_pct: float | None,
) -> TradeDecisionStageRegime:
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

    return TradeDecisionStageRegime(regime_id=regime_id, trade_environment=env)


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


def _stage_opportunity(inputs: TradeDecisionInputs) -> TradeDecisionStageOpportunity:
    rs_pct = _resolve_rs_percentile(inputs)
    early = _early_mover_score(inputs)
    stage = _trend_stage(inputs, rs_pct)

    if rs_pct is not None and rs_pct < 40 and early < 0.35:
        grade: OpportunityGrade = "D"
    elif early >= 0.5 and stage == "early":
        grade = "A"
    elif rs_pct is not None and rs_pct >= 65 and stage in {"early", "mid"}:
        grade = "B" if stage == "mid" or early < 0.5 else "A"
    elif rs_pct is not None and rs_pct >= 50:
        grade = "C"
    elif early >= 0.4:
        grade = "B"
    else:
        grade = "D"

    if stage == "late" and grade in {"A", "B"}:
        grade = "C"

    return TradeDecisionStageOpportunity(
        opportunity_grade=grade,
        rs_percentile=rs_pct,
        trend_stage=stage,
    )


def _score_to_breakout_grade(score: int) -> BreakoutGrade:
    if score <= 25:
        return "F"
    if score >= 78:
        return "A"
    if score >= 62:
        return "B"
    if score >= 45:
        return "C"
    return "D"


def _stage_setup(inputs: TradeDecisionInputs) -> TradeDecisionStageSetup:
    breakout_score = max(0, min(100, inputs.breakout_quality_score))
    breakout_grade = _score_to_breakout_grade(breakout_score)
    sr_conf = max(0, min(100, inputs.support_resistance_confidence))

    if breakout_grade == "F":
        setup_grade: SetupGrade = "D"
    elif breakout_grade in {"A", "B"} and sr_conf >= 55:
        setup_grade = "A" if breakout_grade == "A" and sr_conf >= 65 else "B"
    elif breakout_grade == "C" and sr_conf >= 50:
        setup_grade = "C"
    elif sr_conf < 40:
        setup_grade = "D" if breakout_grade in {"D", "F"} else "C"
    else:
        setup_grade = "C"

    if breakout_grade in {"D", "F"} and sr_conf < 50:
        setup_grade = "D"

    return TradeDecisionStageSetup(
        setup_grade=setup_grade,
        breakout_quality_score=breakout_score,
        breakout_grade=breakout_grade,
        support_resistance_confidence=sr_conf,
        pattern_reliability=inputs.pattern_reliability,
    )


def _final_verdict(
    regime: TradeDecisionStageRegime,
    opportunity: TradeDecisionStageOpportunity,
    setup: TradeDecisionStageSetup,
) -> tuple[TradeVerdict, ActionHint]:
    if regime.trade_environment == "AVOID":
        return "NO_TRADE", "Avoid"
    if setup.setup_grade == "D" or setup.breakout_grade == "F":
        return "NO_TRADE", "Avoid"

    high_conviction = (
        regime.trade_environment == "FAVORABLE"
        and opportunity.opportunity_grade in {"A", "B"}
        and setup.breakout_grade in {"A", "B"}
        and setup.setup_grade in {"A", "B"}
    )
    if high_conviction:
        return "HIGH_CONVICTION_TRADE", "Buy"

    if opportunity.opportunity_grade == "D":
        return "NO_TRADE", "Avoid"

    if setup.setup_grade in {"A", "B"} or opportunity.opportunity_grade in {"A", "B"}:
        return "MEDIUM_SETUP", "Wait"

    if opportunity.opportunity_grade == "C" and setup.setup_grade == "C":
        return "WATCHLIST", "Wait"

    return "WATCHLIST", "Wait"


def _regime_label(regime_id: str | None) -> str:
    mapping = {
        REGIME_RISK_ON_TREND: "Risk-on trend",
        REGIME_RISK_ON_CHOP: "Risk-on chop",
        REGIME_RISK_OFF: "Risk-off",
        REGIME_HIGH_VOL_CHOP: "High-volatility chop",
    }
    return mapping.get(regime_id or "", regime_id or "Unknown")


def _build_explanation(
    *,
    inputs: TradeDecisionInputs,
    regime: TradeDecisionStageRegime,
    opportunity: TradeDecisionStageOpportunity,
    setup: TradeDecisionStageSetup,
    verdict: TradeVerdict,
) -> list[str]:
    env = regime.trade_environment
    regime_line = (
        f"Market regime: {_regime_label(regime.regime_id)} — trade environment {env.lower()}."
    )

    rs_pct = opportunity.rs_percentile
    if rs_pct is not None:
        strength = (
            f"Stock strength: RS near {rs_pct:.0f}th percentile vs the ranked universe."
        )
    else:
        strength = "Stock strength: relative strength data limited for this symbol."

    stage_map = {
        "early": "early leadership / inflection",
        "mid": "confirmed mid-trend",
        "late": "late-stage extension",
        "unknown": "stage unclear",
    }
    stage_line = (
        f"Stage: {stage_map[opportunity.trend_stage]} "
        f"(opportunity grade {opportunity.opportunity_grade})."
    )

    breakout_line = (
        f"Breakout grade {setup.breakout_grade} "
        f"(quality score {setup.breakout_quality_score}/100)."
    )

    risk_parts: list[str] = []
    if setup.support_resistance_confidence < 40:
        risk_parts.append("weak nearby support/resistance")
    if regime.trade_environment == "NEUTRAL":
        risk_parts.append("neutral macro regime")
    if setup.pattern_reliability == "low":
        risk_parts.append("low pattern sample reliability")
    if opportunity.trend_stage == "late":
        risk_parts.append("late-trend extension risk")
    if not risk_parts:
        risk_parts.append("normal execution and regime risk")
    risk_line = f"Key risk: {risk_parts[0]}."

    bullets = [regime_line, strength, stage_line, breakout_line, risk_line]
    if verdict == "NO_TRADE":
        bullets[0] = regime_line + " Verdict blocks new long exposure."
    return bullets


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
