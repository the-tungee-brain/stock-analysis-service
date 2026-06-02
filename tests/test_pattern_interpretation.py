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


def test_bullish_trend_overrides_bearish_pattern():
    interpretation = build_pattern_interpretation(
        pattern=_evening_star(),
        context=_context(),
        scores=_scores(),
        setup_outcome=None,
        history=None,
        model_prediction=1,
    )
    assert interpretation["actionable_verdict"] == "Bullish Trend Overrides Bearish Pattern"
    assert "Trader Summary:" in interpretation["trader_summary"]
    assert interpretation["final_verdict"]["conclusion"]
    assert len(interpretation["confidence_contributors"]) == 5
    pattern_row = next(r for r in interpretation["confidence_contributors"] if r["key"] == "pattern")
    assert pattern_row["weight_pct"] == 10
    assert pattern_row["qualitative"] == "Bearish"
    trend_row = next(r for r in interpretation["confidence_contributors"] if r["key"] == "trend")
    assert trend_row["weight_pct"] == 35
    assert trend_row["emphasized"] is True


def test_model_only_verdict():
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
    )
    assert "Model-Only" in interpretation["actionable_verdict"]


def test_historical_read_for_setup():
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
    assert interpretation["historical_read"] is not None
    assert interpretation["historical_read"].startswith("Historical Read:")
    assert "mildly positive" in interpretation["historical_read"]
