from app.builders.trade_decision_engine import TradeDecisionInputs, evaluate_trade_decision
from ranking_pipeline.regime.constants import REGIME_RISK_OFF, REGIME_RISK_ON_TREND


def test_avoid_regime_forces_no_trade():
    inputs = TradeDecisionInputs(
        symbol="TEST",
        as_of_date="2026-01-01",
        regime_id=REGIME_RISK_OFF,
        market_breadth_pct=None,
        rs_percentile=90.0,
        rs_score_0_1=0.9,
        rs_21d=0.05,
        rs_63d=0.02,
        vol_ratio_20d=1.5,
        dist_52w_high_pct=0.05,
        near_52w_high=False,
        trend_acceleration=True,
        breakout_quality_score=85,
        support_resistance_confidence=70,
        pattern_reliability="high",
        ranking_rank=1,
        universe_rank_count=100,
    )
    result = evaluate_trade_decision(inputs)
    assert result.verdict == "NO_TRADE"
    assert result.action == "AVOID"
    assert result.regime.trade_environment == "AVOID"
    assert result.trade_quality_score == 0


def test_high_conviction_when_unified_score_high():
    inputs = TradeDecisionInputs(
        symbol="TEST",
        as_of_date="2026-01-01",
        regime_id=REGIME_RISK_ON_TREND,
        market_breadth_pct=None,
        rs_percentile=92.0,
        rs_score_0_1=0.92,
        rs_21d=0.08,
        rs_63d=0.02,
        vol_ratio_20d=1.5,
        dist_52w_high_pct=0.08,
        near_52w_high=False,
        trend_acceleration=True,
        breakout_quality_score=88,
        support_resistance_confidence=72,
        pattern_reliability="high",
        ranking_rank=2,
        universe_rank_count=100,
    )
    result = evaluate_trade_decision(inputs)
    assert result.trade_quality_score >= 80
    assert result.verdict == "HIGH_CONVICTION_TRADE"
    assert result.action == "ENTER"
    assert len(result.explanation) <= 5


def test_weak_breakout_yields_no_trade():
    inputs = TradeDecisionInputs(
        symbol="TEST",
        as_of_date="2026-01-01",
        regime_id=REGIME_RISK_ON_TREND,
        market_breadth_pct=None,
        rs_percentile=55.0,
        rs_score_0_1=0.55,
        rs_21d=None,
        rs_63d=None,
        vol_ratio_20d=None,
        dist_52w_high_pct=None,
        near_52w_high=False,
        trend_acceleration=False,
        breakout_quality_score=15,
        support_resistance_confidence=50,
        pattern_reliability="low",
        ranking_rank=40,
        universe_rank_count=50,
    )
    result = evaluate_trade_decision(inputs)
    assert result.trade_quality_score < 60
    assert result.verdict == "NO_TRADE"
    assert result.action == "AVOID"


def test_mid_score_watchlist():
    inputs = TradeDecisionInputs(
        symbol="TEST",
        as_of_date="2026-01-01",
        regime_id=REGIME_RISK_ON_TREND,
        market_breadth_pct=None,
        rs_percentile=72.0,
        rs_score_0_1=0.72,
        rs_21d=0.05,
        rs_63d=0.02,
        vol_ratio_20d=1.25,
        dist_52w_high_pct=0.1,
        near_52w_high=False,
        trend_acceleration=True,
        breakout_quality_score=65,
        support_resistance_confidence=60,
        pattern_reliability="medium",
        ranking_rank=15,
        universe_rank_count=100,
    )
    result = evaluate_trade_decision(inputs)
    assert 60 <= result.trade_quality_score < 80
    assert result.verdict == "WATCHLIST"
    assert result.action == "WAIT_FOR_SETUP"
