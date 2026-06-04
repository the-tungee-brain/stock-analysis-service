"""Tests for equity exit guidance engine."""

from __future__ import annotations

from app.builders.equity_exit_guidance_engine import (
    EquityExitGuidanceInputs,
    evaluate_equity_exit_guidance,
)
from app.models.intelligence_models import IntelligenceSignal
from app.models.trade_decision_models import (
    TradeDecision,
    TradeDecisionReasonBreakdown,
    TradeDecisionRegime,
)


def _trade(
    *,
    score: int = 72,
    env: str = "FAVORABLE",
    regime_id: str = "risk_on_trend",
) -> TradeDecision:
    return TradeDecision(
        symbol="TEST",
        as_of_date="2026-06-01",
        regime=TradeDecisionRegime(regimeId=regime_id, tradeEnvironment=env),
        tradeQualityScore=score,
        scoreBucket="TRADE" if score >= 80 else "SETUP",
        verdict="TRADE" if score >= 80 else "WATCHLIST",
        action="ENTER" if score >= 80 else "WAIT_FOR_SETUP",
        reasonBreakdown=TradeDecisionReasonBreakdown(
            hardBlockers=[],
            primaryWeakness=None,
            secondaryFactors=["Mid-trend structure"],
        ),
    )


def test_hold_healthy_position():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="TEST",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=85),
            signals=[],
            alert_reasons=[],
            position_weight_pct=8.0,
            open_profit_loss_pct=12.0,
        )
    )
    assert result.verdict == "HOLD"
    assert result.exit_urgency <= 24


def test_trim_on_concentration():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="TEST",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=75),
            signals=[
                IntelligenceSignal(
                    kind="position_size",
                    severity="watch",
                    message="TEST position weight is 22.0% of portfolio.",
                    symbol="TEST",
                )
            ],
            alert_reasons=[],
            position_weight_pct=22.0,
            open_profit_loss_pct=5.0,
        )
    )
    assert result.verdict in {"TRIM", "REVIEW_SELL", "EXIT"}


def test_review_sell_on_deep_drawdown():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="TEST",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=45, env="NEUTRAL"),
            signals=[
                IntelligenceSignal(
                    kind="drawdown",
                    severity="critical",
                    message="Unrealized loss ~-32% on TEST — urgent risk review recommended.",
                    symbol="TEST",
                )
            ],
            alert_reasons=[],
            position_weight_pct=12.0,
            open_profit_loss_pct=-32.0,
        )
    )
    assert result.verdict in {"REVIEW_SELL", "EXIT"}


def test_exit_regime_avoid_and_large_loss():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="TEST",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=30, env="AVOID", regime_id="risk_off"),
            signals=[],
            alert_reasons=["Macro risk-off regime"],
            position_weight_pct=18.0,
            open_profit_loss_pct=-28.0,
        )
    )
    assert result.verdict == "EXIT"
