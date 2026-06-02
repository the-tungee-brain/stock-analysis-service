"""Compact interpretation for Pattern Intelligence (Model C primary)."""

from __future__ import annotations

from typing import Any, Literal

from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.historical_analytics import (
    PatternHistoricalStats,
    SetupOutcomeStats,
)
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown
from analysis.pattern_intelligence.trend_context import TrendContext

Side = Literal["bullish", "bearish", "neutral"]
Tone = Literal[
    "strong_bullish",
    "bullish",
    "slight_bullish",
    "neutral",
    "slight_bearish",
    "bearish",
    "strong_bearish",
    "unavailable",
]


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
    probability = _resolve_probability(ranking_score, model_prediction)
    signal_state = _signal_state_block(probability)
    timeframe = _timeframe_block(
        signal_state=signal_state,
        context=context,
        scores=scores,
    )
    model_side = _model_side(probability, model_prediction)
    trend_side = _trend_side(context, scores.trend_strength)
    signal_summary = _signal_summary(
        pattern=pattern,
        context=context,
        scores=scores,
        model_prediction=model_prediction,
        ranking_score=ranking_score,
        signal_state=signal_state,
    )
    verdict = _verdict_line(model_side=model_side, trend_side=trend_side)
    evidence = _evidence_block(
        pattern=pattern,
        context=context,
        setup_outcome=setup_outcome,
        history=history,
        model_side=model_side,
        signal_state=signal_state,
    )
    alignment = _alignment_block(
        pattern=pattern,
        context=context,
        scores=scores,
        signal_state=signal_state,
        timeframe=timeframe,
        model_side=model_side,
        trend_side=trend_side,
        evidence=evidence,
    )
    return {
        "signal_state": signal_state,
        "timeframe": timeframe,
        "alignment": alignment,
        "signal_summary": signal_summary,
        "verdict": verdict,
        "evidence": evidence,
    }


def _resolve_probability(
    ranking_score: float | None,
    model_prediction: int | None,
) -> float | None:
    if ranking_score is not None:
        return ranking_score
    if model_prediction is None:
        return None
    return 0.62 if model_prediction == 1 else 0.38


def _conviction_tier(probability: float) -> tuple[str, Tone]:
    if probability >= 0.65:
        return "Strong Bullish", "strong_bullish"
    if probability >= 0.57:
        return "Bullish", "bullish"
    if probability >= 0.52:
        return "Slight Bullish", "slight_bullish"
    if probability >= 0.48:
        return "Neutral", "neutral"
    if probability >= 0.43:
        return "Slight Bearish", "slight_bearish"
    if probability >= 0.35:
        return "Bearish", "bearish"
    return "Strong Bearish", "strong_bearish"


def _signal_state_block(probability: float | None) -> dict[str, Any]:
    if probability is None:
        return {
            "label": "Unavailable",
            "probability": None,
            "probability_text": "Model C probability unavailable",
            "tone": "unavailable",
        }

    label, tone = _conviction_tier(probability)
    return {
        "label": label,
        "probability": probability,
        "probability_text": f"{_pct(probability)} probability of outperforming SPY",
        "tone": tone,
    }


def _timeframe_block(
    *,
    signal_state: dict[str, Any],
    context: TrendContext,
    scores: PatternScoreBreakdown,
) -> dict[str, Any]:
    trend_label, trend_detail = _long_term_trend_label(context, scores.trend_strength)
    rs_label, rs_detail = _relative_strength_display(context, scores)
    return {
        "short_term": {
            "label": signal_state["label"],
            "caption": "Model C · 5-day horizon",
        },
        "long_term_trend": {
            "label": trend_label,
            "caption": trend_detail,
        },
        "relative_strength": {
            "label": rs_label,
            "caption": rs_detail,
        },
    }


def _long_term_trend_label(
    context: TrendContext,
    trend_strength: float,
) -> tuple[str, str]:
    if _trend_is_bullish(context, trend_strength):
        if context.above_sma_50 is True and context.above_sma_200 is True:
            return "Bullish", "Above SMA50/200"
        return "Bullish", "Uptrend structure"
    if _trend_is_bearish(context, trend_strength):
        if context.above_sma_50 is False and context.above_sma_200 is False:
            return "Bearish", "Below SMA50/200"
        return "Bearish", "Downtrend structure"
    bias = context.trend_bias.replace("_", " ").title()
    return "Neutral", f"{bias} · mixed SMA structure"


def _relative_strength_display(
    context: TrendContext,
    scores: PatternScoreBreakdown,
) -> tuple[str, str]:
    rs = context.rs_vs_spy_63d
    if rs is None:
        rs = context.rs_vs_spy_21d
    score = scores.relative_strength

    if (rs is not None and rs > 0.05) or score >= 0.8:
        return "Strongly Positive", "Outperforming SPY"
    if (rs is not None and rs > 0) or score >= 0.55:
        return "Moderately Positive", "vs SPY"
    if (rs is not None and rs < -0.05) or score <= 0.3:
        return "Strongly Negative", "Underperforming SPY"
    if (rs is not None and rs < 0) or score <= 0.45:
        return "Moderately Negative", "vs SPY"
    return "Neutral", "In line with SPY"


def _model_side(probability: float | None, model_prediction: int | None) -> Side:
    if probability is not None:
        if probability >= 0.52:
            return "bullish"
        if probability <= 0.48:
            return "bearish"
        return "neutral"
    if model_prediction == 1:
        return "bullish"
    if model_prediction == 0:
        return "bearish"
    return "neutral"


def _trend_side(context: TrendContext, trend_strength: float) -> Side:
    if _trend_is_bullish(context, trend_strength):
        return "bullish"
    if _trend_is_bearish(context, trend_strength):
        return "bearish"
    return "neutral"


def _signal_summary(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
    ranking_score: float | None,
    signal_state: dict[str, Any],
) -> dict[str, Any]:
    pattern_line, pattern_warning = _pattern_summary_line(pattern)
    return {
        "model_c": _model_c_line(signal_state, model_prediction, ranking_score),
        "trend": _trend_line(context),
        "relative_strength": _relative_strength_line(context, scores),
        "pattern": pattern_line,
        "pattern_warning": pattern_warning,
    }


def _model_c_line(
    signal_state: dict[str, Any],
    model_prediction: int | None,
    ranking_score: float | None,
) -> str:
    label = signal_state["label"]
    if ranking_score is not None:
        return f"{label} ({_pct(ranking_score)})"
    if model_prediction is None:
        return "Unavailable"
    direction = "Bullish" if model_prediction == 1 else "Bearish"
    return f"{direction} · {label}"


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
    label, detail = _relative_strength_display(context, scores)
    return f"{label} ({detail})"


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


def _verdict_line(*, model_side: Side, trend_side: Side) -> str:
    if model_side == "bearish" and trend_side == "bullish":
        return "Near-term weakness inside a longer-term uptrend."
    if model_side == "bullish" and trend_side == "bullish":
        return "Bullish continuation setup."
    if model_side == "bearish" and trend_side == "bearish":
        return "Bearish trend remains intact."
    if model_side == "bullish" and trend_side == "bearish":
        return "Counter-trend rebound signal."
    if model_side == "neutral":
        return "Mixed near-term signal — weigh the longer-term trend."
    if trend_side == "neutral":
        return "Short-term model read with inconclusive long-term trend."
    return "Follow Model C — patterns provide context only."


def _alignment_block(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    signal_state: dict[str, Any],
    timeframe: dict[str, Any],
    model_side: Side,
    trend_side: Side,
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    short_label = signal_state["label"]
    trend_label = timeframe["long_term_trend"]["label"]
    rs_label = timeframe["relative_strength"]["label"]
    paragraphs: list[str] = []

    if model_side == "bearish" and trend_side == "bullish":
        lead = (
            f"The 5-day model is {short_label.lower()}, but the stock remains "
            f"in a {trend_label.lower()} long-term uptrend"
        )
        if rs_label in {"Moderately Positive", "Strongly Positive"}:
            lead += " and continues to outperform SPY"
        paragraphs.append(f"{lead}.")
    elif model_side == "bullish" and trend_side == "bearish":
        paragraphs.append(
            f"The 5-day model is {short_label.lower()}, but the longer-term trend "
            f"remains {trend_label.lower()}."
        )
    elif model_side != "neutral" and trend_side != "neutral" and model_side != trend_side:
        paragraphs.append(
            f"The short-term model is {short_label.lower()} while the long-term trend "
            f"is {trend_label.lower()}."
        )

    if pattern is not None and pattern.direction not in {None, "neutral"}:
        pattern_bullish = pattern.direction == "bullish"
        model_bullish = model_side == "bullish"
        if pattern_bullish != model_bullish and model_side != "neutral":
            if pattern_bullish:
                paragraphs.append(
                    f"A bullish {pattern.label} pattern conflicts with the "
                    f"{short_label.lower()} model read."
                )
            else:
                paragraphs.append(
                    f"A bearish {pattern.label} pattern conflicts with the "
                    f"{short_label.lower()} model read."
                )

    framing = evidence.get("framing")
    stats_note = evidence.get("stats_note")
    if framing and stats_note and len(paragraphs) >= 1:
        if "despite" in framing.lower() or "disagree" in framing.lower():
            if "historically" not in " ".join(paragraphs).lower():
                paragraphs.append(framing)

    if not paragraphs:
        return None

    state = "conflict" if model_side != trend_side and model_side != "neutral" and trend_side != "neutral" else "mixed"
    headline = "Signal Conflict" if state == "conflict" else "Mixed Signals"
    return {
        "state": state,
        "headline": headline,
        "explanation": " ".join(paragraphs),
    }


def _evidence_block(
    *,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    setup_outcome: SetupOutcomeStats | None,
    history: PatternHistoricalStats | None,
    model_side: Side,
    signal_state: dict[str, Any],
) -> dict[str, Any]:
    stats = _pick_stats(setup_outcome, history)
    if stats is None:
        return {
            "framing": (
                "Insufficient matched history on this symbol to frame pattern risk — rely on Model C."
            ),
            "stats_note": None,
            "insight": (
                "Insufficient matched history on this symbol to frame pattern risk — rely on Model C."
            ),
            "conditional_note": None,
            "summary": "Insufficient matched history on this symbol — rely on Model C.",
            "setup_label": setup_outcome.label if setup_outcome else None,
            "occurrence_count": None,
            "win_rate_5d": None,
            "avg_return_5d": None,
            "avg_return_20d": None,
        }

    filtered = setup_outcome is not None and setup_outcome.avg_return_5d is not None
    insight = _evidence_insight(
        stats=stats,
        context=context,
        pattern=pattern,
        filtered=filtered,
    )
    framing = _evidence_framing(
        stats=stats,
        pattern=pattern,
        model_side=model_side,
        signal_state=signal_state,
        insight=insight,
    )
    stats_note = (
        "Historical statistics describe past outcomes and may disagree with "
        "the current model forecast."
    )
    conditional_note = _conditional_note(stats["occurrence_count"])
    summary = _evidence_summary(
        stats=stats,
        context=context,
        pattern=pattern,
        filtered=filtered,
    )
    return {
        "framing": framing,
        "stats_note": stats_note,
        "insight": insight,
        "conditional_note": conditional_note,
        "summary": summary,
        "setup_label": stats.get("setup_label"),
        "occurrence_count": stats.get("occurrence_count"),
        "win_rate_5d": stats.get("win_rate_5d"),
        "avg_return_5d": stats.get("avg_return_5d"),
        "avg_return_20d": stats.get("avg_return_20d"),
    }


def _evidence_framing(
    *,
    stats: dict[str, Any],
    pattern: CandlestickPatternHit | None,
    model_side: Side,
    signal_state: dict[str, Any],
    insight: str,
) -> str:
    win = stats.get("win_rate_5d")
    avg5 = stats.get("avg_return_5d")
    hist_positive = (win is not None and win >= 0.55) or (
        avg5 is not None and avg5 > 0.003
    )
    hist_negative = (win is not None and win < 0.45) or (
        avg5 is not None and avg5 < -0.003
    )
    pattern_bearish = pattern is not None and pattern.direction == "bearish"
    pattern_bullish = pattern is not None and pattern.direction == "bullish"
    model_bearish = model_side == "bearish"
    model_bullish = model_side == "bullish"

    if hist_positive and (model_bearish or pattern_bearish):
        return (
            "This setup has historically produced positive returns despite "
            "the bearish pattern."
        )
    if hist_positive and model_bullish and pattern_bullish:
        return "Historical outcomes align with the current bullish model and pattern read."
    if hist_positive and model_bullish:
        return "Historical outcomes support the current bullish model read."
    if hist_negative and model_bullish:
        return (
            "Historical outcomes have often been weak in this setup — "
            "the model forecast may be optimistic."
        )
    if hist_negative and (model_bearish or pattern_bearish):
        return "Historical outcomes align with the current cautious read."

    model_label = signal_state.get("label", "the model")
    return (
        f"Past matches for this setup show mixed follow-through — "
        f"use as context alongside the {model_label.lower()} model read."
    )


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


def _regime_phrase(context: TrendContext) -> str:
    trend = "strong uptrend" if context.trend_bias == "uptrend" else (
        "downtrend" if context.trend_bias == "downtrend" else "mixed trend"
    )
    rs = context.rs_vs_spy_63d
    if rs is None:
        rs = context.rs_vs_spy_21d
    if rs is not None and rs > 0:
        return f"{trend} + positive RS"
    if rs is not None and rs < 0:
        return f"{trend} + negative RS"
    return trend


def _evidence_insight(
    *,
    stats: dict[str, Any],
    context: TrendContext,
    pattern: CandlestickPatternHit | None,
    filtered: bool,
) -> str:
    count = int(stats["occurrence_count"])
    win = stats.get("win_rate_5d")
    pattern_label = pattern.label if pattern is not None else stats.get("setup_label", "This setup")
    regime = _regime_phrase(context)

    if pattern is None:
        return (
            f"Historical matches in {regime} regimes ({count} samples) describe past context only — "
            "they do not override Model C."
        )

    if filtered:
        setup_phrase = f"{pattern_label} patterns in {regime} regimes"
    else:
        setup_phrase = f"{pattern_label} patterns on this symbol"

    if pattern.direction == "bearish" and context.trend_bias == "uptrend" and (context.rs_vs_spy_63d or 0) > 0:
        return (
            f"Historically, {setup_phrase} have NOT been reliable reversal signals "
            "when trend and RS remain strong."
        )

    if pattern.direction == "bullish" and context.trend_bias == "downtrend":
        return (
            f"Historically, {setup_phrase} have rarely produced durable reversals "
            "without RS improvement."
        )

    if win is not None and win < 0.45:
        return (
            f"Historically, {setup_phrase} have been unreliable standalone signals "
            "in this conditional setup."
        )

    if win is not None and win >= 0.55 and pattern.direction != "neutral":
        return (
            f"Historically, {setup_phrase} have often continued in the direction "
            "of the prevailing trend/RS backdrop."
        )

    return (
        f"Historically, {setup_phrase} show mixed follow-through — "
        "use as risk context, not a standalone trigger."
    )


def _conditional_note(count: int | None) -> str | None:
    if count is None:
        return None
    sample = "small sample" if count < 12 else "limited historical sample"
    return (
        f"Based on {count} past matches on this symbol ({sample}). "
        "Conditional statistics — descriptive context only, not a predictive guarantee."
    )


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
