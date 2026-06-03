from app.builders.trade_decision_engine import TradeDecisionInputs, evaluate_trade_decision
from ranking_pipeline.regime.constants import (
    REGIME_RISK_OFF,
    REGIME_RISK_ON_CHOP,
    REGIME_RISK_ON_TREND,
)


def test_avoid_regime_hard_blocker_not_primary_weakness():
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
    assert any("regime gate" in b.lower() for b in result.reason_breakdown.hard_blockers)
    assert result.reason_breakdown.primary_weakness is not None
    assert "regime gate" not in result.reason_breakdown.primary_weakness.lower()


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
    assert result.verdict == "TRADE"
    assert result.reason_breakdown.hard_blockers == []
    assert result.reason_breakdown.primary_weakness is not None


def test_weak_inputs_score_hard_blocker():
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
    assert any("below threshold" in b for b in result.reason_breakdown.hard_blockers)
    assert "breakout" in result.reason_breakdown.primary_weakness.lower()


def test_neutral_regime_is_secondary_not_primary():
    inputs = TradeDecisionInputs(
        symbol="TEST",
        as_of_date="2026-01-01",
        regime_id=REGIME_RISK_ON_CHOP,
        market_breadth_pct=None,
        rs_percentile=78.0,
        rs_score_0_1=0.78,
        rs_21d=0.04,
        rs_63d=0.03,
        vol_ratio_20d=1.1,
        dist_52w_high_pct=0.1,
        near_52w_high=False,
        trend_acceleration=False,
        breakout_quality_score=45,
        support_resistance_confidence=58,
        pattern_reliability="medium",
        ranking_rank=12,
        universe_rank_count=100,
    )
    result = evaluate_trade_decision(inputs)
    assert result.regime.trade_environment == "NEUTRAL"
    assert "breakout" in result.reason_breakdown.primary_weakness.lower()
    assert any(
        "neutral regime" in f.lower() and "risk_on_chop" in f
        for f in result.reason_breakdown.secondary_factors
    )
    assert "regime" not in result.reason_breakdown.primary_weakness.lower()


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
    assert result.verdict == "WATCHLIST"
    assert len(result.reason_breakdown.secondary_factors) <= 3
