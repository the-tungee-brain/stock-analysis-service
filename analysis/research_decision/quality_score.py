"""Composite research quality score (0–100)."""

from __future__ import annotations

from typing import Any

from analysis.research_decision.regime import regime_alignment_score
from analysis.research_decision.trend_labels import TrendLabel


def compute_research_quality_score(
    *,
    ranking_score: float | None,
    daily_trend: TrendLabel,
    weekly_trend: TrendLabel,
    indicators: dict[str, float],
    regime_market: str,
    signal_bias: str,
    chart_intelligence_score: int | None = None,
) -> dict[str, Any]:
    model_confidence = _model_confidence(ranking_score)
    trend_quality = _trend_quality(daily_trend, weekly_trend)
    rs_quality = _relative_strength_quality(indicators)
    regime_quality = regime_alignment_score(
        signal_bias=signal_bias,
        market_regime=regime_market,
    )
    chart_quality = (chart_intelligence_score / 100.0) if chart_intelligence_score is not None else 0.5

    weights = {
        "model_confidence": 0.25,
        "trend_quality": 0.25,
        "relative_strength": 0.20,
        "regime_alignment": 0.15,
        "chart_intelligence": 0.15,
    }
    components = {
        "model_confidence": model_confidence,
        "trend_quality": trend_quality,
        "relative_strength": rs_quality,
        "regime_alignment": regime_quality,
        "chart_intelligence": chart_quality,
    }
    score = sum(components[key] * weights[key] for key in weights) * 100
    score = int(round(max(0, min(100, score))))

    return {
        "score": score,
        "headline": _headline(score),
        "components": {key: round(value * 100) for key, value in components.items()},
        "weights": weights,
    }


def _model_confidence(ranking_score: float | None) -> float:
    if ranking_score is None:
        return 0.5
    return min(1.0, abs(ranking_score - 0.5) * 2)


def _trend_quality(daily: TrendLabel, weekly: TrendLabel) -> float:
    if daily == weekly and daily != "neutral":
        return 1.0
    if daily == weekly:
        return 0.5
    if daily == "neutral" or weekly == "neutral":
        return 0.55
    return 0.3


def _relative_strength_quality(indicators: dict[str, float]) -> float:
    rs21 = indicators.get("rs_vs_spy_21d")
    rs63 = indicators.get("rs_vs_spy_63d")
    if rs21 is None and rs63 is None:
        return 0.5
    score = 0.5
    if rs21 is not None:
        score += max(-0.25, min(0.25, rs21 * 2))
    if rs63 is not None:
        score += max(-0.15, min(0.15, rs63))
    return max(0.0, min(1.0, score))


def _headline(score: int) -> str:
    if score >= 75:
        return "High-conviction setup"
    if score >= 60:
        return "Solid research quality"
    if score >= 45:
        return "Mixed signals — proceed with caution"
    return "Low conviction — verify before acting"
