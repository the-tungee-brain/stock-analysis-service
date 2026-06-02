"""User-facing interpretation for pattern intelligence (Model C remains primary)."""

from __future__ import annotations

from typing import Any

from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.historical_analytics import (
    PatternHistoricalStats,
    SetupOutcomeStats,
)
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown
from analysis.pattern_intelligence.trend_context import TrendContext

SCORE_WEIGHTS: tuple[tuple[str, str, int], ...] = (
    ("trend", "Trend", 35),
    ("relative_strength", "Relative strength", 35),
    ("pattern", "Pattern", 10),
    ("volume", "Volume", 10),
    ("model_alignment", "Model alignment", 10),
)


def build_pattern_interpretation(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    setup_outcome: SetupOutcomeStats | None,
    history: PatternHistoricalStats | None,
    model_prediction: int | None,
) -> dict[str, Any]:
    verdict = _actionable_verdict(
        pattern=pattern,
        context=context,
        scores=scores,
        model_prediction=model_prediction,
    )
    bullets = _final_verdict_bullets(pattern, context, model_prediction)
    conclusion = _final_verdict_conclusion(
        pattern=pattern,
        context=context,
        scores=scores,
        model_prediction=model_prediction,
    )
    contributors = _confidence_contributors(pattern, scores)
    historical_read = _historical_read(
        pattern=pattern,
        context=context,
        setup_outcome=setup_outcome,
        history=history,
    )
    trader_summary = _trader_summary(
        pattern=pattern,
        context=context,
        scores=scores,
        setup_outcome=setup_outcome,
        model_prediction=model_prediction,
        verdict=verdict,
    )

    return {
        "actionable_verdict": verdict,
        "trader_summary": trader_summary,
        "final_verdict": {
            "title": "Final Verdict",
            "bullets": bullets,
            "conclusion": conclusion,
        },
        "confidence_contributors": contributors,
        "historical_read": historical_read,
    }


def _actionable_verdict(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
) -> str:
    if pattern is None:
        return "Model-Only Signal (No Meaningful Pattern)"
    if pattern.direction == "neutral":
        return "Pattern Is Neutral"

    if model_prediction is None:
        return "Pattern Is Informational (Model Unavailable)"

    model_bullish = model_prediction == 1
    pattern_bullish = pattern.direction == "bullish"
    trend_bullish = _trend_is_bullish(context, scores.trend_strength)
    trend_bearish = _trend_is_bearish(context, scores.trend_strength)
    trend_weak = not trend_bullish and not trend_bearish

    if model_bullish == pattern_bullish:
        return "Pattern Confirms Core Signal"

    if not pattern_bullish and model_bullish:
        if trend_bullish:
            return "Bullish Trend Overrides Bearish Pattern"
        if trend_weak:
            return "Bearish Pattern Confirmed By Weak Trend"
        return "Bearish Pattern Conflicts With Core Signal"

    if pattern_bullish and not model_bullish:
        if trend_bearish:
            return "Bearish Trend Overrides Bullish Pattern"
        if trend_weak:
            return "Bullish Pattern Confirmed By Weak Trend"
        return "Bullish Pattern Conflicts With Core Signal"

    return "Pattern Conflicts With Core Signal"


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


def _final_verdict_bullets(
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    model_prediction: int | None,
) -> list[dict[str, str]]:
    bullets: list[dict[str, str]] = []

    if pattern is not None:
        tone = _direction_tone(pattern.direction)
        bullets.append(
            {
                "tone": tone,
                "text": f"{pattern.label} detected ({pattern.direction})",
            }
        )
    else:
        bullets.append(
            {
                "tone": "neutral",
                "text": "No meaningful candlestick pattern in recent sessions",
            }
        )

    if model_prediction is not None:
        model_text = (
            "Model C expects outperformance vs SPY"
            if model_prediction == 1
            else "Model C expects underperformance vs SPY"
        )
        bullets.append({"tone": "positive" if model_prediction == 1 else "negative", "text": model_text})

    trend_tone, trend_text = _trend_bullet(context)
    bullets.append({"tone": trend_tone, "text": trend_text})

    if context.above_sma_50 is not None:
        bullets.append(
            {
                "tone": "positive" if context.above_sma_50 else "negative",
                "text": "Above SMA50" if context.above_sma_50 else "Below SMA50",
            }
        )
    if context.above_sma_200 is not None:
        bullets.append(
            {
                "tone": "positive" if context.above_sma_200 else "negative",
                "text": "Above SMA200" if context.above_sma_200 else "Below SMA200",
            }
        )

    rs = context.rs_vs_spy_63d
    if rs is None:
        rs = context.rs_vs_spy_21d
    if rs is not None:
        if rs > 0:
            bullets.append(
                {"tone": "positive", "text": "Relative strength stronger than SPY"}
            )
        elif rs < 0:
            bullets.append(
                {"tone": "negative", "text": "Relative strength weaker than SPY"}
            )
        else:
            bullets.append({"tone": "neutral", "text": "Relative strength in line with SPY"})

    return bullets


def _trend_bullet(context: TrendContext) -> tuple[str, str]:
    if context.trend_bias == "uptrend":
        return "positive", "Strong uptrend"
    if context.trend_bias == "downtrend":
        return "negative", "Strong downtrend"
    return "neutral", "Mixed trend regime"


def _final_verdict_conclusion(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
) -> str:
    if pattern is None:
        return (
            "No candlestick pattern is present. Rely on Model C (Relative Strength + Trend) "
            "as the sole decision input."
        )

    if pattern.direction == "neutral":
        return (
            "The pattern is neutral and does not add directional conviction. "
            "Model C remains the primary decision engine."
        )

    if scores.alignment_state == "confirmed":
        return (
            "The candlestick pattern supports the core ranking signal. "
            "Treat it as confirmation context — not a standalone trade trigger."
        )

    trend_dominates = scores.trend_strength >= 0.6 or scores.relative_strength >= 0.6
    pattern_bearish = pattern.direction == "bearish"
    pattern_bullish = pattern.direction == "bullish"
    model_bullish = model_prediction == 1 if model_prediction is not None else None

    if trend_dominates and model_bullish is True and pattern_bearish:
        return (
            "The bearish pattern is not strong enough to invalidate the core ranking signal. "
            "Trend and relative strength dominate the confirmation score (70% combined weight). "
            "Treat the pattern as a caution flag rather than a reversal signal."
        )
    if trend_dominates and model_bullish is False and pattern_bullish:
        return (
            "The bullish pattern is not strong enough to invalidate the core ranking signal. "
            "Trend and relative strength dominate the confirmation score (70% combined weight). "
            "Treat the pattern as a counter-trend caution rather than a buy signal."
        )

    if _trend_is_bearish(context, scores.trend_strength) and pattern_bearish:
        return (
            "The bearish pattern aligns with a weak or downtrending backdrop. "
            "Even so, Model C remains the primary alpha source — use the pattern as supporting context."
        )

    return (
        "The candlestick pattern conflicts with Model C. Pattern strength is capped at 10% of the "
        "confirmation score. Defer to the ranking signal for directional decisions."
    )


def _confidence_contributors(
    pattern: CandlestickPatternHit | None,
    scores: PatternScoreBreakdown,
) -> list[dict[str, Any]]:
    qualitative_map = {
        "trend": _score_qualitative(scores.trend_strength),
        "relative_strength": _score_qualitative(scores.relative_strength),
        "pattern": _pattern_qualitative(pattern),
        "volume": _volume_qualitative(scores.volume_confirmation),
        "model_alignment": _alignment_qualitative(scores.model_alignment, scores.alignment_state),
    }
    value_map = {
        "trend": scores.trend_strength,
        "relative_strength": scores.relative_strength,
        "pattern": scores.pattern_strength,
        "volume": scores.volume_confirmation,
        "model_alignment": scores.model_alignment,
    }

    rows: list[dict[str, Any]] = []
    for key, label, weight in SCORE_WEIGHTS:
        rows.append(
            {
                "key": key,
                "label": label,
                "weight_pct": weight,
                "qualitative": qualitative_map[key],
                "emphasized": weight >= 35,
                "score": value_map[key],
            }
        )
    return rows


def _score_qualitative(score: float) -> str:
    if score >= 0.72:
        return "Strong"
    if score >= 0.55:
        return "Moderate"
    if score >= 0.4:
        return "Weak"
    return "Very weak"


def _pattern_qualitative(pattern: CandlestickPatternHit | None) -> str:
    if pattern is None:
        return "None"
    if pattern.direction == "neutral":
        return "Neutral"
    prefix = pattern.direction.capitalize()
    if pattern.strength >= 0.7:
        return prefix
    if pattern.strength >= 0.45:
        return f"Mildly {pattern.direction}"
    return f"Weak {pattern.direction}"


def _volume_qualitative(score: float) -> str:
    if score >= 0.75:
        return "Strong"
    if score >= 0.55:
        return "Neutral"
    return "Weak"


def _alignment_qualitative(score: float, alignment_state: str) -> str:
    if alignment_state == "model_only":
        return "N/A"
    if score >= 0.7:
        return "Strong"
    if score >= 0.45:
        return "Moderate"
    return "Weak"


def _historical_read(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    setup_outcome: SetupOutcomeStats | None,
    history: PatternHistoricalStats | None,
) -> str | None:
    stats = setup_outcome
    if stats is not None and stats.avg_return_5d is not None and stats.occurrence_count >= 3:
        return _format_historical_read(
            pattern_label=stats.pattern_label,
            occurrence_count=stats.occurrence_count,
            avg_return_5d=stats.avg_return_5d,
            avg_return_20d=stats.avg_return_20d,
            win_rate_5d=stats.win_rate_5d,
            context=context,
            pattern=pattern,
            filtered=True,
        )

    if history is not None and history.occurrence_count >= 3 and history.avg_return_5d is not None:
        return _format_historical_read(
            pattern_label=history.label,
            occurrence_count=history.occurrence_count,
            avg_return_5d=history.avg_return_5d,
            avg_return_20d=history.avg_return_20d,
            win_rate_5d=history.win_rate_5d,
            context=context,
            pattern=pattern,
            filtered=False,
        )

    return (
        "Historical Read: Not enough matched setups on this symbol to draw a reliable "
        "conclusion. Defer to Model C."
    )


def _format_historical_read(
    *,
    pattern_label: str,
    occurrence_count: int,
    avg_return_5d: float | None,
    avg_return_20d: float | None,
    win_rate_5d: float | None,
    context: TrendContext,
    pattern: CandlestickPatternHit | None,
    filtered: bool,
) -> str:
    if avg_return_5d is None:
        return (
            "Historical Read: Insufficient forward-return data for this setup. "
            "Do not treat the pattern as a standalone edge."
        )

    magnitude = _return_magnitude(avg_return_5d, avg_return_20d)
    win_text = _win_rate_text(win_rate_5d)
    setup_scope = (
        "this combined setup (pattern + trend + RS)"
        if filtered
        else f"{pattern_label.lower()} occurrences"
    )

    trend_tail = ""
    if context.above_sma_200 is True and (context.rs_vs_spy_63d or 0) > 0:
        trend_tail = (
            " The pattern alone has not been a reliable reversal signal when the stock "
            "remains above SMA200 and is outperforming SPY."
        )
    elif context.above_sma_200 is False and (context.rs_vs_spy_63d or 0) < 0:
        trend_tail = (
            " In this weaker backdrop, bearish patterns have more often coincided with "
            "continued softness — still treat Model C as primary."
        )

    pattern_note = ""
    if pattern is not None and pattern.direction == "bearish" and context.above_sma_200:
        pattern_note = trend_tail or (
            " Counter-trend patterns in this regime have usually failed to reverse the "
            "broader trend."
        )
    elif pattern is not None and pattern.direction == "bullish" and not context.above_sma_200:
        pattern_note = (
            " Bullish patterns in this regime have rarely produced durable reversals "
            "without RS improvement."
        )

    return (
        f"Historical Read: {setup_scope.capitalize()} ({occurrence_count} samples) has "
        f"historically produced {magnitude} returns ({_pct(avg_return_5d)} over 5 days"
        f"{f', {_pct(avg_return_20d)} over 20 days' if avg_return_20d is not None else ''}). "
        f"{win_text}{pattern_note or trend_tail or ' Use this as context only — Model C drives the trade decision.'}"
    )


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


def _win_rate_text(win_rate: float | None) -> str:
    if win_rate is None:
        return ""
    if win_rate >= 0.6:
        return f"The 5-day win rate ({_pct(win_rate)}) has been favorable."
    if win_rate >= 0.48:
        return f"The 5-day win rate ({_pct(win_rate)}) has been roughly coin-flip."
    return f"The 5-day win rate ({_pct(win_rate)}) has been unfavorable."


def _trader_summary(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    setup_outcome: SetupOutcomeStats | None,
    model_prediction: int | None,
    verdict: str,
) -> str:
    if pattern is None:
        return (
            "Trader Summary: No actionable candlestick pattern — follow Model C "
            "(Relative Strength + Trend) for the directional view."
        )

    pattern_clause = f"A short-term {pattern.direction} {pattern.label.lower()} appeared"
    if scores.alignment_state == "confirmed":
        trend_clause = "and aligns with the core Model C signal"
    elif scores.alignment_state == "conflict":
        if _trend_is_bullish(context, scores.trend_strength):
            trend_clause = "but the broader trend and relative strength remain bullish"
        elif _trend_is_bearish(context, scores.trend_strength):
            trend_clause = "and the broader trend backdrop remains soft"
        else:
            trend_clause = "while trend and relative strength are mixed"
    else:
        trend_clause = "with limited directional information"

    history_clause = ""
    if setup_outcome is not None and setup_outcome.avg_return_5d is not None:
        if setup_outcome.avg_return_5d > 0 and model_prediction == 1:
            history_clause = " Historical outcomes suggest the trend has usually continued."
        elif setup_outcome.avg_return_5d < 0 and model_prediction == 0:
            history_clause = " Historical outcomes have tended to follow the weaker path."
        elif abs(setup_outcome.avg_return_5d) < 0.005:
            history_clause = " Historical outcomes for this setup have been mixed."
        else:
            history_clause = " Historical outcomes favor the core model over the pattern alone."

    return f"Trader Summary: {pattern_clause}, {trend_clause}.{history_clause} Model C remains the primary signal."


def _direction_tone(direction: str) -> str:
    if direction == "bullish":
        return "positive"
    if direction == "bearish":
        return "warning"
    return "neutral"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"
