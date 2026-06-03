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
    assert result.action_hint == "Avoid"
    assert result.regime.trade_environment == "AVOID"


def test_high_conviction_when_layers_align():
    inputs = TradeDecisionInputs(
        symbol="TEST",
        as_of_date="2026-01-01",
        regime_id=REGIME_RISK_ON_TREND,
        market_breadth_pct=None,
        rs_percentile=75.0,
        rs_score_0_1=0.75,
        rs_21d=0.06,
        rs_63d=0.02,
        vol_ratio_20d=1.4,
        dist_52w_high_pct=0.08,
        near_52w_high=False,
        trend_acceleration=True,
        breakout_quality_score=82,
        support_resistance_confidence=68,
        pattern_reliability="medium",
        ranking_rank=3,
        universe_rank_count=100,
    )
    result = evaluate_trade_decision(inputs)
    assert result.verdict == "HIGH_CONVICTION_TRADE"
    assert result.action_hint == "Buy"
    assert len(result.explanation) <= 5


def test_breakout_f_forces_no_trade():
    inputs = TradeDecisionInputs(
        symbol="TEST",
        as_of_date="2026-01-01",
        regime_id=REGIME_RISK_ON_TREND,
        market_breadth_pct=None,
        rs_percentile=80.0,
        rs_score_0_1=0.8,
        rs_21d=None,
        rs_63d=None,
        vol_ratio_20d=None,
        dist_52w_high_pct=None,
        near_52w_high=False,
        trend_acceleration=False,
        breakout_quality_score=15,
        support_resistance_confidence=70,
        pattern_reliability="high",
        ranking_rank=2,
        universe_rank_count=50,
    )
    result = evaluate_trade_decision(inputs)
    assert result.setup.breakout_grade == "F"
    assert result.verdict == "NO_TRADE"
