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


def test_three_layer_structure():
    interpretation = build_pattern_interpretation(
        pattern=_evening_star(),
        context=_context(),
        scores=_scores(),
        setup_outcome=None,
        history=None,
        model_prediction=1,
        ranking_score=0.506,
    )
    assert "signal_summary" in interpretation
    assert "verdict" in interpretation
    assert "evidence" in interpretation
    assert "Bullish" in interpretation["signal_summary"]["model_c"]
    assert "50.6%" in interpretation["signal_summary"]["model_c"]
    assert "Evening Star" in interpretation["signal_summary"]["pattern"]
    assert "dominates bearish pattern" in interpretation["verdict"]
    assert "trader_summary" not in interpretation
    assert "confidence_contributors" not in interpretation


def test_model_only_signal_summary():
    interpretation = build_pattern_interpretation(
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
    assert "follow Model C" in interpretation["verdict"]


def test_evidence_summary_compressed():
    setup = SetupOutcomeStats(
        label="Evening Star · Above SMA200 · RS leading SPY",
        pattern_label="Evening Star",
        trend_label="Above SMA200",
        rs_label="RS leading SPY",
        occurrence_count=16,
        pattern_only_count=40,
        avg_return_5d=0.006,
        avg_return_20d=0.008,
        win_rate_5d=0.533,
        win_rate_20d=0.5,
        max_drawdown_20d=-0.04,
    )
    interpretation = build_pattern_interpretation(
        pattern=_evening_star(),
        context=_context(),
        scores=_scores(),
        setup_outcome=setup,
        history=None,
        model_prediction=1,
    )
    evidence = interpretation["evidence"]
    assert evidence["occurrence_count"] == 16
    assert evidence["avg_return_5d"] == 0.006
    assert "insight" in evidence
    assert "NOT been reliable reversal" in evidence["insight"] or "unreliable" in evidence["insight"].lower()
    assert evidence.get("conditional_note")
    assert "not a predictive guarantee" in evidence["conditional_note"].lower()
