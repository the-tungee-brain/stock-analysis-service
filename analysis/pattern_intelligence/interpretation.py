"""Compact 3-layer interpretation for Pattern Intelligence (Model C primary)."""

from __future__ import annotations

from typing import Any

from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.historical_analytics import (
    PatternHistoricalStats,
    SetupOutcomeStats,
)
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown
from analysis.pattern_intelligence.trend_context import TrendContext


def build_pattern_interpretation(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    setup_outcome: SetupOutcomeStats | None,
    history: PatternHistoricalStats | None,
    model_prediction: int | None,
    ranking_score: float | None = None,
) -> dict[str, Any]:
    signal_summary = _signal_summary(
        pattern=pattern,
        context=context,
        scores=scores,
        model_prediction=model_prediction,
        ranking_score=ranking_score,
    )
    verdict = _verdict_line(
        pattern=pattern,
        context=context,
        scores=scores,
        model_prediction=model_prediction,
    )
    evidence = _evidence_block(
        pattern=pattern,
        context=context,
        setup_outcome=setup_outcome,
        history=history,
    )
    return {
        "signal_summary": signal_summary,
        "verdict": verdict,
        "evidence": evidence,
    }


def _signal_summary(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, Any]:
    pattern_line, pattern_warning = _pattern_summary_line(pattern)
    return {
        "model_c": _model_c_line(model_prediction, ranking_score),
        "trend": _trend_line(context),
        "relative_strength": _relative_strength_line(context, scores),
        "pattern": pattern_line,
        "pattern_warning": pattern_warning,
    }


def _model_c_line(model_prediction: int | None, ranking_score: float | None) -> str:
    if model_prediction is None:
        return "Unavailable"
    direction = "Bullish" if model_prediction == 1 else "Bearish"
    if ranking_score is not None:
        return f"{direction} ({_pct(ranking_score)} rank score)"
    return direction


def _trend_line(context: TrendContext) -> str:
    bias = context.trend_bias.replace("_", " ").title()
    if context.above_sma_50 is True and context.above_sma_200 is True:
        return f"{bias} (Above SMA50/200)"
    if context.above_sma_50 is False and context.above_sma_200 is False:
        return f"{bias} (Below SMA50/200)"
    if context.above_sma_50 is not None or context.above_sma_200 is not None:
        return f"{bias} (Mixed SMA50/200)"
    return bias


def _relative_strength_line(context: TrendContext, scores: PatternScoreBreakdown) -> str:
    rs = context.rs_vs_spy_63d
    if rs is None:
        rs = context.rs_vs_spy_21d
    qual = _score_qualitative(scores.relative_strength)
    if rs is None:
        return qual
    if rs > 0:
        return f"{qual} vs SPY"
    if rs < 0:
        return "Weak vs SPY"
    return "In line with SPY"


def _pattern_summary_line(
    pattern: CandlestickPatternHit | None,
) -> tuple[str | None, bool]:
    if pattern is None:
        return None, False
    weight = "low weight"
    if pattern.strength >= 0.7:
        weight = "high quality"
    elif pattern.strength >= 0.45:
        weight = "moderate weight"
    warning = pattern.direction == "bearish" or pattern.direction == "bullish"
    if pattern.direction == "neutral":
        return f"{pattern.label} (neutral)", False
    return f"{pattern.label} ({pattern.direction}, {weight})", warning


def _verdict_line(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
) -> str:
    if pattern is None:
        return "No pattern detected — follow Model C ranking signal only."

    if pattern.direction == "neutral":
        return "Pattern is neutral — follow Model C ranking signal."

    if model_prediction is None:
        return "Pattern is informational — Model C unavailable."

    model_bullish = model_prediction == 1
    pattern_bullish = pattern.direction == "bullish"
    trend_bullish = _trend_is_bullish(context, scores.trend_strength)
    trend_bearish = _trend_is_bearish(context, scores.trend_strength)
    trend_weak = not trend_bullish and not trend_bearish

    if model_bullish == pattern_bullish:
        return "Pattern confirms Model C — continue to follow the ranking signal."

    if not pattern_bullish and model_bullish:
        if trend_bullish:
            return (
                "Bullish trend dominates bearish pattern — continue to follow Model C ranking signal."
            )
        if trend_weak:
            return "Bearish pattern confirmed by weak trend — reduce exposure."
        return "Bearish pattern conflicts with Model C — defer to the ranking signal."

    if pattern_bullish and not model_bullish:
        if trend_bearish:
            return (
                "Bearish trend dominates bullish pattern — continue to follow Model C ranking signal."
            )
        if trend_weak:
            return "Bullish pattern in weak trend — do not override Model C."
        return "Bullish pattern conflicts with Model C — defer to the ranking signal."

    return "Pattern conflicts with Model C — defer to the ranking signal."


def _evidence_block(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    setup_outcome: SetupOutcomeStats | None,
    history: PatternHistoricalStats | None,
) -> dict[str, Any]:
    stats = _pick_stats(setup_outcome, history)
    if stats is None:
        return {
            "summary": "Insufficient matched history on this symbol — rely on Model C.",
            "setup_label": setup_outcome.label if setup_outcome else None,
            "occurrence_count": None,
            "win_rate_5d": None,
            "avg_return_5d": None,
            "avg_return_20d": None,
        }

    summary = _evidence_summary(
        stats=stats,
        context=context,
        pattern=pattern,
        filtered=setup_outcome is not None and setup_outcome.avg_return_5d is not None,
    )
    return {
        "summary": summary,
        "setup_label": stats.get("setup_label"),
        "occurrence_count": stats.get("occurrence_count"),
        "win_rate_5d": stats.get("win_rate_5d"),
        "avg_return_5d": stats.get("avg_return_5d"),
        "avg_return_20d": stats.get("avg_return_20d"),
    }


def _pick_stats(
    setup_outcome: SetupOutcomeStats | None,
    history: PatternHistoricalStats | None,
) -> dict[str, Any] | None:
    if (
        setup_outcome is not None
        and setup_outcome.avg_return_5d is not None
        and setup_outcome.occurrence_count >= 3
    ):
        return {
            "setup_label": setup_outcome.label,
            "occurrence_count": setup_outcome.occurrence_count,
            "win_rate_5d": setup_outcome.win_rate_5d,
            "avg_return_5d": setup_outcome.avg_return_5d,
            "avg_return_20d": setup_outcome.avg_return_20d,
            "filtered": True,
        }
    if (
        history is not None
        and history.avg_return_5d is not None
        and history.occurrence_count >= 3
    ):
        return {
            "setup_label": history.label,
            "occurrence_count": history.occurrence_count,
            "win_rate_5d": history.win_rate_5d,
            "avg_return_5d": history.avg_return_5d,
            "avg_return_20d": history.avg_return_20d,
            "filtered": False,
        }
    return None


def _evidence_summary(
    *,
    stats: dict[str, Any],
    context: TrendContext,
    pattern: CandlestickPatternHit | None,
    filtered: bool,
) -> str:
    avg5 = stats["avg_return_5d"]
    avg20 = stats.get("avg_return_20d")
    win = stats.get("win_rate_5d")
    count = stats["occurrence_count"]
    magnitude = _return_magnitude(float(avg5), avg20)

    line1 = (
        f"{count} matched setups produced {magnitude} returns "
        f"({_pct(avg5)} avg 5d"
        f"{f', {_pct(avg20)} avg 20d' if avg20 is not None else ''})."
    )

    if win is not None:
        if win >= 0.55:
            line2 = f"5-day win rate {_pct(win)} — trend context has usually persisted."
        elif win >= 0.45:
            line2 = f"5-day win rate {_pct(win)} — outcomes mixed; pattern is not standalone alpha."
        else:
            line2 = f"5-day win rate {_pct(win)} — pattern alone has been unreliable."
    else:
        line2 = "Use as context only — Model C drives the decision."

    if (
        filtered
        and pattern is not None
        and pattern.direction == "bearish"
        and context.above_sma_200
        and (context.rs_vs_spy_63d or 0) > 0
    ):
        line2 = (
            "In this trend/RS regime, bearish patterns have rarely reversed the core signal."
        )

    return f"{line1} {line2}"


def _trend_is_bullish(context: TrendContext, trend_strength: float) -> bool:
    if context.trend_bias == "uptrend" and trend_strength >= 0.65:
        return True
    if context.above_sma_50 is True and context.above_sma_200 is True:
        return trend_strength >= 0.6
    return False


def _trend_is_bearish(context: TrendContext, trend_strength: float) -> bool:
    if context.trend_bias == "downtrend" and trend_strength <= 0.35:
        return True
    if context.above_sma_50 is False and context.above_sma_200 is False:
        return trend_strength <= 0.4
    return False


def _score_qualitative(score: float) -> str:
    if score >= 0.72:
        return "Strong"
    if score >= 0.55:
        return "Moderate"
    if score >= 0.4:
        return "Weak"
    return "Very weak"


def _return_magnitude(avg5: float, avg20: float | None) -> str:
    ref = avg20 if avg20 is not None else avg5
    if abs(ref) < 0.003:
        return "flat"
    if ref > 0.02:
        return "strongly positive"
    if ref > 0:
        return "mildly positive"
    if ref < -0.02:
        return "strongly negative"
    return "mildly negative"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"
