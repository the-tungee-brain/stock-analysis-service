"""Composite scoring: patterns confirm Model C (RS + trend), not standalone alpha."""

from __future__ import annotations

from dataclasses import dataclass

from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.trend_context import TrendContext


AlignmentState = str  # "confirmed" | "conflict" | "model_only"


@dataclass(frozen=True)
class PatternScoreBreakdown:
    pattern_strength: float
    trend_strength: float
    relative_strength: float
    volume_confirmation: float
    model_alignment: float
    confirmation_score: float
    confidence: str
    alignment_state: AlignmentState


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def score_trend_strength(context: TrendContext) -> float:
    score = 0.5
    if context.above_sma_50 is True:
        score += 0.2
    elif context.above_sma_50 is False:
        score -= 0.2
    if context.above_sma_200 is True:
        score += 0.3
    elif context.above_sma_200 is False:
        score -= 0.3
    return _clamp(score)


def score_relative_strength(context: TrendContext) -> float:
    rs = context.rs_vs_spy_63d
    if rs is None:
        rs = context.rs_vs_spy_21d
    if rs is None:
        return 0.5
    # Map typical RS spread (-0.15..+0.15) into 0..1
    return _clamp(0.5 + rs * 2.5)


def score_volume_confirmation(context: TrendContext) -> float:
    ratio = context.vol_ratio_20d
    if ratio is None:
        return 0.5
    if ratio >= 1.5:
        return 0.9
    if ratio >= 1.1:
        return 0.75
    if ratio >= 0.9:
        return 0.55
    return 0.35


def _pattern_direction_score(direction: str) -> float:
    if direction == "bullish":
        return 1.0
    if direction == "bearish":
        return 0.0
    return 0.5


def score_model_alignment(
    *,
    pattern: CandlestickPatternHit | None,
    model_prediction: int | None,
    model_up_prob: float | None,
) -> float:
    """How well the candlestick pattern aligns with Model C direction."""
    if pattern is None or model_prediction is None:
        return 0.5
    model_bullish = model_prediction == 1
    if pattern.direction == "neutral":
        return 0.5
    pattern_bullish = pattern.direction == "bullish"
    aligned = model_bullish == pattern_bullish
    base = 0.85 if aligned else 0.25
    if model_up_prob is not None:
        edge = abs(model_up_prob - 0.5) * 0.3
        base = _clamp(base + (edge if aligned else -edge))
    return base


def build_pattern_scores(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    model_prediction: int | None = None,
    model_up_prob: float | None = None,
    ranking_score: float | None = None,
) -> PatternScoreBreakdown:
    pattern_strength = pattern.strength if pattern is not None else 0.0
    trend_strength = score_trend_strength(context)
    relative_strength = score_relative_strength(context)
    volume_confirmation = score_volume_confirmation(context)
    model_alignment = score_model_alignment(
        pattern=pattern,
        model_prediction=model_prediction,
        model_up_prob=model_up_prob or ranking_score,
    )

    # Core alpha weight stays on RS + trend model; patterns are a modest overlay.
    confirmation_score = _clamp(
        0.10 * pattern_strength
        + 0.35 * trend_strength
        + 0.35 * relative_strength
        + 0.10 * volume_confirmation
        + 0.10 * model_alignment
    )

    confidence = _confidence_label(confirmation_score, model_alignment, pattern is not None)
    alignment_state = derive_alignment_state(
        pattern=pattern,
        model_prediction=model_prediction,
        model_alignment=model_alignment,
    )
    return PatternScoreBreakdown(
        pattern_strength=round(pattern_strength, 3),
        trend_strength=round(trend_strength, 3),
        relative_strength=round(relative_strength, 3),
        volume_confirmation=round(volume_confirmation, 3),
        model_alignment=round(model_alignment, 3),
        confirmation_score=round(confirmation_score, 3),
        confidence=confidence,
        alignment_state=alignment_state,
    )


def derive_alignment_state(
    *,
    pattern: CandlestickPatternHit | None,
    model_prediction: int | None,
    model_alignment: float,
) -> AlignmentState:
    """Product-facing alignment between candlestick pattern and Model C."""
    del model_alignment
    if pattern is None or model_prediction is None:
        return "model_only"
    if pattern.direction == "neutral":
        return "model_only"
    model_bullish = model_prediction == 1
    pattern_bullish = pattern.direction == "bullish"
    if model_bullish == pattern_bullish:
        return "confirmed"
    return "conflict"


def _confidence_label(score: float, alignment: float, has_pattern: bool) -> str:
    if not has_pattern:
        return "model_only"
    if score >= 0.72 and alignment >= 0.7:
        return "high"
    if score >= 0.58:
        return "moderate"
    if alignment < 0.4:
        return "conflicting"
    return "low"
