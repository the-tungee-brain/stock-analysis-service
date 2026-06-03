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
    assert result.score_bucket == "NO_TRADE"
    assert result.action == "AVOID"
    assert result.regime.trade_environment == "AVOID"
    assert result.trade_quality_score == 0
    assert result.primary_rejection_reason == "Neutral or unfavorable regime"


def test_trade_verdict_when_score_high():
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
    assert result.score_bucket == "TRADE"
    assert result.verdict == "TRADE"
    assert result.action == "ENTER"
    assert result.primary_rejection_reason is None


def test_weak_inputs_no_trade_bucket():
    inputs = TradeDecisionInputs(
        symbol="TEST",
        as_of_date="2026-01-01",
        regime_id=REGIME_RISK_ON_TREND,
        market_breadth_pct=None,
        rs_percentile=35.0,
        rs_score_0_1=0.35,
        rs_21d=None,
        rs_63d=None,
        vol_ratio_20d=None,
        dist_52w_high_pct=None,
        near_52w_high=False,
        trend_acceleration=False,
        breakout_quality_score=12,
        support_resistance_confidence=40,
        pattern_reliability="low",
        ranking_rank=40,
        universe_rank_count=50,
    )
    result = evaluate_trade_decision(inputs)
    assert result.trade_quality_score < 40
    assert result.score_bucket == "NO_TRADE"
    assert result.verdict == "NO_TRADE"
    assert result.action == "AVOID"
    assert result.primary_rejection_reason is not None


def test_setup_bucket_watchlist_verdict():
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
    assert result.score_bucket == "SETUP"
    assert result.verdict == "WATCHLIST"
    assert result.action == "WAIT_FOR_SETUP"
    assert result.primary_rejection_reason is not None


def test_watchlist_bucket_weak_setup():
    inputs = TradeDecisionInputs(
        symbol="TEST",
        as_of_date="2026-01-01",
        regime_id=REGIME_RISK_ON_TREND,
        market_breadth_pct=None,
        rs_percentile=52.0,
        rs_score_0_1=0.52,
        rs_21d=0.02,
        rs_63d=0.01,
        vol_ratio_20d=1.1,
        dist_52w_high_pct=0.12,
        near_52w_high=False,
        trend_acceleration=False,
        breakout_quality_score=48,
        support_resistance_confidence=52,
        pattern_reliability="medium",
        ranking_rank=25,
        universe_rank_count=80,
    )
    result = evaluate_trade_decision(inputs)
    assert 40 <= result.trade_quality_score < 60
    assert result.score_bucket == "WATCHLIST"
    assert result.verdict == "WATCHLIST"
    assert result.action == "WAIT_FOR_SETUP"
