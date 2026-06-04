"""Driver-aligned explanations, long-option calibration, relative-risk ranking."""

from __future__ import annotations

from app.builders.equity_exit_guidance_engine import (
    EquityExitGuidanceInputs,
    evaluate_equity_exit_guidance,
)
from app.builders.guidance_scoring_types import MEANINGFUL_LOSS_PCT
from app.builders.option_position_guidance_engine import (
    LongOptionGuidanceInputs,
    evaluate_long_option,
)
from app.builders.position_guidance_relative_risk import compute_relative_risk_rank
from app.models.intelligence_models import IntelligenceSignal
from app.models.trade_decision_models import (
    TradeDecision,
    TradeDecisionReasonBreakdown,
    TradeDecisionRegime,
)


def _trade(*, score: int = 72, env: str = "FAVORABLE") -> TradeDecision:
    return TradeDecision(
        symbol="AMZN",
        as_of_date="2026-06-01",
        regime=TradeDecisionRegime(regimeId="risk_on_trend", tradeEnvironment=env),
        tradeQualityScore=score,
        scoreBucket="SETUP",
        verdict="WATCHLIST",
        action="WAIT_FOR_SETUP",
        reasonBreakdown=TradeDecisionReasonBreakdown(
            hardBlockers=[],
            primaryWeakness=None,
            secondaryFactors=[],
        ),
    )


def test_equity_small_loss_does_not_label_large_drawdown():
    """Drawdown signal alone must not become primary driver when P/L is tiny."""
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="AMZN",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=75, env="NEUTRAL"),
            signals=[
                IntelligenceSignal(
                    kind="drawdown",
                    severity="warning",
                    message="AMZN drawdown watch",
                    symbol="AMZN",
                )
            ],
            alert_reasons=[],
            position_weight_pct=3.0,
            open_profit_loss_pct=-1.8,
        )
    )
    assert result.primary_driver.code != "LARGE_DRAWDOWN"
    assert "Large drawdown" not in result.primary_reason


def test_equity_explanation_matches_top_contributor():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="AMZN",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=75, env="NEUTRAL"),
            signals=[],
            alert_reasons=[],
            position_weight_pct=22.0,
            open_profit_loss_pct=5.0,
        )
    )
    top = max(result.contributors, key=lambda c: c.points)
    assert result.primary_driver.code == top.driver
    assert result.primary_driver.label in result.primary_reason


def test_long_option_minus_20_gets_review_close_floor():
    result = evaluate_long_option(
        LongOptionGuidanceInputs(
            position_kind="LONG_CALL",
            thesis="BULLISH",
            dte=30,
            pnl_pct=-21.6,
            moneyness="OTM",
            alert_reasons=[],
        )
    )
    assert result.verdict in {"REVIEW_CLOSE", "CLOSE"}
    assert result.primary_driver.code in {"LARGE_DRAWDOWN", "THETA_DECAY", "THESIS_CONFLICT"}
    if result.primary_driver.code == "LARGE_DRAWDOWN":
        assert "drawdown" in result.primary_driver.label.lower()


def test_amzn_like_long_option_more_urgent_than_small_equity_loss():
    long_opt = evaluate_long_option(
        LongOptionGuidanceInputs(
            position_kind="LONG_CALL",
            thesis="BULLISH",
            dte=25,
            pnl_pct=-21.6,
            moneyness="OTM",
            alert_reasons=[],
        )
    )
    equity = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="AMZN",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=85, env="FAVORABLE"),
            signals=[],
            alert_reasons=[],
            position_weight_pct=2.0,
            open_profit_loss_pct=-1.8,
        )
    )
    long_rank = compute_relative_risk_rank(
        position_kind="LONG_CALL",
        verdict=long_opt.verdict,
        urgency=long_opt.urgency,
        open_profit_loss_pct=-21.6,
    )
    equity_rank = compute_relative_risk_rank(
        position_kind="EQUITY_LONG",
        verdict=equity.verdict,
        urgency=equity.exit_urgency,
        open_profit_loss_pct=-1.8,
        position_weight_pct=2.0,
    )
    assert long_rank > equity_rank


def test_meaningful_loss_threshold_documented():
    assert MEANINGFUL_LOSS_PCT == -5.0


def test_long_option_theta_tiers_contribute():
    result = evaluate_long_option(
        LongOptionGuidanceInputs(
            position_kind="LONG_CALL",
            thesis="BULLISH",
            dte=5,
            pnl_pct=-5.0,
            moneyness="ATM",
            alert_reasons=[],
        )
    )
    buckets = {c.bucket for c in result.contributors}
    assert "theta" in buckets
    assert any(c.driver == "THETA_DECAY" for c in result.contributors)
