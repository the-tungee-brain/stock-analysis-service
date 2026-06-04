"""Position Guidance consistency: drivers, calibration, ranking, TRIM guard."""

from __future__ import annotations

from app.builders.equity_exit_guidance_engine import (
    EquityExitGuidanceInputs,
    evaluate_equity_exit_guidance,
)
from app.builders.guidance_scoring_types import LARGE_DRAWDOWN_MIN_PCT
from app.builders.option_position_guidance_engine import (
    LongOptionGuidanceInputs,
    evaluate_long_option,
)
from app.builders.position_guidance_relative_risk import compute_relative_risk_rank
from app.builders.position_guidance_verdict_normalize import normalize_equity_verdict
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


def _top_contributor(result) -> object:
    return max(result.contributors, key=lambda c: c.points)


def test_equity_small_loss_no_large_drawdown_label():
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
            open_profit_loss_pct=-2.0,
            position_quantity=10.0,
        )
    )
    assert result.primary_driver.code != "LARGE_DRAWDOWN"
    assert "Large drawdown" not in result.primary_reason
    top = _top_contributor(result)
    assert result.primary_driver.points == top.points
    assert top.label in result.primary_reason


def test_equity_explanation_matches_highest_point_contributor():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="AMZN",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=75, env="NEUTRAL"),
            signals=[],
            alert_reasons=[],
            position_weight_pct=22.0,
            open_profit_loss_pct=5.0,
            position_quantity=50.0,
        )
    )
    top = _top_contributor(result)
    assert result.primary_driver.points == top.points
    assert top.label in result.primary_reason


def test_one_share_equity_never_trim():
    raw = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="AMZN",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=75),
            signals=[
                IntelligenceSignal(
                    kind="position_size",
                    severity="watch",
                    message="AMZN position weight is 22.0% of portfolio.",
                    symbol="AMZN",
                )
            ],
            alert_reasons=[],
            position_weight_pct=22.0,
            open_profit_loss_pct=5.0,
            position_quantity=1.0,
        )
    )
    assert raw.verdict != "TRIM"
    assert normalize_equity_verdict("TRIM", 1.0) == "EXIT"


def test_long_option_minus_20_minimum_review_close():
    result = evaluate_long_option(
        LongOptionGuidanceInputs(
            position_kind="LONG_CALL",
            thesis="BULLISH",
            dte=30,
            pnl_pct=-21.0,
            moneyness="OTM",
            alert_reasons=[],
        )
    )
    assert result.verdict in {"REVIEW_CLOSE", "CLOSE"}
    top = _top_contributor(result)
    assert result.primary_driver.points == top.points
    assert top.label in result.primary_reason


def test_long_option_minus_35_close():
    result = evaluate_long_option(
        LongOptionGuidanceInputs(
            position_kind="LONG_CALL",
            thesis="BULLISH",
            dte=30,
            pnl_pct=-36.0,
            moneyness="OTM",
            alert_reasons=[],
        )
    )
    assert result.verdict == "CLOSE"


def test_option_minus_20_higher_urgency_than_equity_minus_2():
    long_opt = evaluate_long_option(
        LongOptionGuidanceInputs(
            position_kind="LONG_CALL",
            thesis="BULLISH",
            dte=25,
            pnl_pct=-21.0,
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
            open_profit_loss_pct=-2.0,
            position_quantity=10.0,
        )
    )
    assert long_opt.urgency > equity.exit_urgency
    long_rank = compute_relative_risk_rank(
        position_kind="LONG_CALL",
        verdict=long_opt.verdict,
        urgency=long_opt.urgency,
        open_profit_loss_pct=-21.0,
    )
    equity_rank = compute_relative_risk_rank(
        position_kind="EQUITY_LONG",
        verdict=equity.verdict,
        urgency=equity.exit_urgency,
        open_profit_loss_pct=-2.0,
        position_weight_pct=2.0,
    )
    assert long_rank > equity_rank


def test_large_drawdown_requires_unrealized_loss_dominant_and_pnl_10():
    assert LARGE_DRAWDOWN_MIN_PCT == -10.0
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="AMZN",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=40, env="AVOID"),
            signals=[],
            alert_reasons=[],
            position_weight_pct=5.0,
            open_profit_loss_pct=-25.0,
            position_quantity=10.0,
        )
    )
    top = _top_contributor(result)
    if result.primary_driver.code == "LARGE_DRAWDOWN":
        assert top.bucket == "unrealized_loss"
        assert top.points == result.primary_driver.points
