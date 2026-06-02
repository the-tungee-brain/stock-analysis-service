"""Tests for pattern intelligence interpretation layer."""

from __future__ import annotations

from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.historical_analytics import SetupOutcomeStats
from analysis.pattern_intelligence.interpretation import build_pattern_interpretation
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown
from analysis.pattern_intelligence.trend_context import TrendContext


def _context(**overrides) -> TrendContext:
    base = dict(
        as_of_date="2024-06-01",
        close=100.0,
        sma_50=98.0,
        sma_200=95.0,
        above_sma_50=True,
        above_sma_200=True,
        rs_vs_spy_21d=0.02,
        rs_vs_spy_63d=0.04,
        rs_vs_spy_126d=0.03,
        vol_ratio_20d=1.1,
        vol_zscore_20d=0.2,
    )
    base.update(overrides)
    return TrendContext(**base)


def _scores(**overrides) -> PatternScoreBreakdown:
    base = dict(
        pattern_strength=0.85,
        trend_strength=0.9,
        relative_strength=0.75,
        volume_confirmation=0.55,
        model_alignment=0.25,
        confirmation_score=0.72,
        confidence="conflicting",
        alignment_state="conflict",
    )
    base.update(overrides)
    return PatternScoreBreakdown(**base)


def _evening_star() -> CandlestickPatternHit:
    return CandlestickPatternHit(
        pattern_id="evening_star",
        label="Evening Star",
        direction="bearish",
        strength=0.8,
        as_of_date="2024-06-01",
        bar_index=0,
    )


def _bearish_engulfing() -> CandlestickPatternHit:
    return CandlestickPatternHit(
        pattern_id="bearish_engulfing",
        label="Bearish Engulfing",
        direction="bearish",
        strength=0.75,
        as_of_date="2024-06-01",
        bar_index=0,
    )


def test_signal_state_and_timeframe_layers():
    interpretation = build_pattern_interpretation(
        symbol="MSFT",
        pattern=_evening_star(),
        context=_context(),
        scores=_scores(),
        setup_outcome=None,
        history=None,
        model_prediction=0,
        ranking_score=0.453,
    )
    assert interpretation["signal_state"]["label"] == "Slight Bearish"
    assert interpretation["signal_state"]["probability"] == 0.453
    assert "outperforming SPY" in interpretation["signal_state"]["probability_text"]
    assert interpretation["timeframe"]["short_term"]["label"] == "Slight Bearish"
    assert interpretation["timeframe"]["long_term_trend"]["label"] == "Bullish"
    assert interpretation["timeframe"]["relative_strength"]["label"] == "Moderately Positive"


def test_verdict_model_bearish_trend_bullish():
    interpretation = build_pattern_interpretation(
        symbol="MSFT",
        pattern=_bearish_engulfing(),
        context=_context(),
        scores=_scores(),
        setup_outcome=None,
        history=None,
        model_prediction=0,
        ranking_score=0.453,
    )
    assert interpretation["verdict"] == "Near-term weakness inside a longer-term uptrend."
    assert interpretation["alignment"] is not None
    assert interpretation["alignment"]["headline"] == "Signal Conflict"
    assert "5-day model is slight bearish" in interpretation["alignment"]["explanation"].lower()


def test_verdict_bullish_continuation():
    interpretation = build_pattern_interpretation(
        symbol="MSFT",
        pattern=_evening_star(),
        context=_context(),
        scores=_scores(),
        setup_outcome=None,
        history=None,
        model_prediction=1,
        ranking_score=0.62,
    )
    assert interpretation["verdict"] == "Bullish continuation setup."
    assert interpretation["signal_state"]["label"] == "Bullish"


def test_model_only_signal_summary():
    interpretation = build_pattern_interpretation(
        symbol="MSFT",
        pattern=None,
        context=_context(),
        scores=_scores(
            pattern_strength=0.0,
            alignment_state="model_only",
            confidence="model_only",
        ),
        setup_outcome=None,
        history=None,
        model_prediction=1,
        ranking_score=0.62,
    )
    assert interpretation["signal_summary"]["pattern"] is None
    assert "Bullish continuation" in interpretation["verdict"]


def test_evidence_framing_when_history_disagrees_with_model():
    setup = SetupOutcomeStats(
        label="Bearish Engulfing · Above SMA200 · RS leading SPY",
        pattern_label="Bearish Engulfing",
        trend_label="Above SMA200",
        rs_label="RS leading SPY",
        occurrence_count=26,
        pattern_only_count=40,
        avg_return_5d=0.012,
        avg_return_20d=0.018,
        win_rate_5d=0.76,
        win_rate_20d=0.7,
        max_drawdown_20d=-0.04,
    )
    interpretation = build_pattern_interpretation(
        symbol="MSFT",
        pattern=_bearish_engulfing(),
        context=_context(),
        scores=_scores(),
        setup_outcome=setup,
        history=None,
        model_prediction=0,
        ranking_score=0.453,
    )
    evidence = interpretation["evidence"]
    assert evidence["occurrence_count"] == 26
    assert "positive returns despite" in evidence["framing"].lower()
    assert evidence["stats_note"]
    assert "may disagree" in evidence["stats_note"].lower()
    assert evidence.get("conditional_note")


def test_benchmark_symbol_skips_model_c_layers():
    interpretation = build_pattern_interpretation(
        symbol="SPY",
        pattern=_bearish_engulfing(),
        context=_context(),
        scores=_scores(),
        setup_outcome=None,
        history=None,
        model_prediction=0,
        ranking_score=0.453,
    )
    signal_state = interpretation["signal_state"]
    assert signal_state["is_benchmark"] is True
    assert signal_state["probability"] is None
    assert "outperforming SPY" not in signal_state["probability_text"].lower()
    assert interpretation["signal_summary"]["model_c"].startswith("Not applicable")
    assert interpretation["timeframe"]["relative_strength"]["label"] == "Market benchmark"
    assert "Model C" in interpretation["verdict"] or "pattern" in interpretation["verdict"].lower()
