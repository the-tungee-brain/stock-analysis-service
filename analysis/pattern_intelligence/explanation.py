"""Natural-language explanations for pattern intelligence (confirmation layer)."""

from __future__ import annotations

from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.historical_analytics import (
    PatternHistoricalStats,
    SetupOutcomeStats,
)
from analysis.pattern_intelligence.interpretation import build_pattern_interpretation
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown
from analysis.pattern_intelligence.trend_context import TrendContext


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def build_pattern_explanation(
    *,
    symbol: str,
    pattern: CandlestickPatternHit | None,
    context: TrendContext,
    scores: PatternScoreBreakdown,
    history: PatternHistoricalStats | None,
    setup_outcome: SetupOutcomeStats | None = None,
    model_label: str | None,
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, str]:
    """Return structured NL fields for API clients."""
    interpretation = build_pattern_interpretation(
        pattern=pattern,
        context=context,
        scores=scores,
        setup_outcome=setup_outcome,
        history=history,
        model_prediction=model_prediction,
    )
    core_line = _core_model_line(model_label, model_prediction, ranking_score)
    if pattern is None:
        return {
            "headline": f"No recent candlestick pattern on {symbol.upper()}",
            "pattern_summary": (
                "No classic candlestick formation was detected in the last few sessions."
            ),
            "trend_context": _trend_context_text(context),
            "historical_context": interpretation.get("historical_read")
            or "No pattern-specific history to cite.",
            "model_context": core_line,
            "confidence_explanation": interpretation["actionable_verdict"],
            "disclaimer": _disclaimer(),
        }

    headline = f"{pattern.label} detected on {pattern.as_of_date}"
    pattern_summary = (
        f"A {pattern.label.lower()} ({pattern.direction}) formed recently with "
        f"formation quality {pattern.strength:.0%}."
    )
    trend_text = _trend_context_text(context)
    history_text = interpretation.get("historical_read") or _historical_text(history, pattern)
    alignment = _alignment_text(pattern, model_prediction, scores.model_alignment)

    return {
        "headline": headline,
        "pattern_summary": pattern_summary,
        "trend_context": trend_text,
        "historical_context": history_text,
        "model_context": f"{core_line} {alignment}".strip(),
        "confidence_explanation": interpretation["actionable_verdict"],
        "disclaimer": _disclaimer(),
    }


def _core_model_line(
    model_label: str | None,
    model_prediction: int | None,
    ranking_score: float | None,
) -> str:
    label = model_label or "Relative strength + trend"
    if model_prediction is None:
        return f"The core Model C signal ({label}) is unavailable for this symbol."
    direction = "outperform SPY" if model_prediction == 1 else "underperform SPY"
    score_text = f" Ranking score {_pct(ranking_score)}." if ranking_score is not None else ""
    return (
        f"The primary Tomcrest signal remains Model C ({label}), expecting the name to "
        f"{direction} over the next 5 sessions.{score_text}"
    )


def _trend_context_text(context: TrendContext) -> str:
    parts = [f"Price is in a {context.trend_bias} regime"]
    if context.above_sma_50 is not None:
        parts.append("above SMA50" if context.above_sma_50 else "below SMA50")
    if context.above_sma_200 is not None:
        parts.append("above SMA200" if context.above_sma_200 else "below SMA200")
    if context.rs_vs_spy_63d is not None:
        parts.append(f"63-day RS vs SPY {_pct(context.rs_vs_spy_63d)}")
    return ", ".join(parts) + "."


def _historical_text(
    history: PatternHistoricalStats | None,
    pattern: CandlestickPatternHit,
) -> str:
    if history is None or history.occurrence_count < 3:
        return (
            f"Insufficient {pattern.label.lower()} history on this symbol for robust "
            "forward-return statistics."
        )
    return (
        f"Historically on this symbol, {pattern.label.lower()} occurred "
        f"{history.occurrence_count} times in the last 5 years with average 5-day return "
        f"{_pct(history.avg_return_5d)}, 20-day return {_pct(history.avg_return_20d)}, "
        f"5-day win rate {_pct(history.win_rate_5d)}, and typical 20-day max drawdown "
        f"{_pct(history.max_drawdown_20d)}."
    )


def _alignment_text(
    pattern: CandlestickPatternHit,
    model_prediction: int | None,
    alignment: float,
) -> str:
    del alignment
    if model_prediction is None or pattern.direction == "neutral":
        return "Pattern direction is informational and does not confirm the core model."
    model_bullish = model_prediction == 1
    pattern_bullish = pattern.direction == "bullish"
    if model_bullish == pattern_bullish:
        return (
            "The candlestick pattern direction aligns with the core model, adding "
            "confirmation rather than a standalone trade signal."
        )
    return (
        "The candlestick pattern conflicts with the core model; treat it as context "
        "only and defer to the ranking signal."
    )


def _confidence_text(scores: PatternScoreBreakdown) -> str:
    mapping = {
        "high": "High confirmation — pattern, trend, relative strength, and model direction agree.",
        "moderate": "Moderate confirmation — mixed but supportive context around the core model.",
        "low": "Low confirmation — pattern is weak or trend/RS context is mixed.",
        "conflicting": "Conflicting — candlestick direction disagrees with the core model.",
        "model_only": "Model only — no recent candlestick pattern detected.",
    }
    return mapping.get(scores.confidence, scores.confidence)


def _disclaimer() -> str:
    return (
        "Pattern intelligence is an explanation and confirmation layer. "
        "The Relative Strength + Trend ranking model remains the primary alpha signal."
    )
