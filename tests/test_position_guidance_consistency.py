"""Single-source Position Guidance: consistency, cross-leg, AI guardrails."""

from __future__ import annotations

import pytest

from app.builders.equity_exit_guidance_engine import (
    EquityExitGuidanceInputs,
    evaluate_equity_exit_guidance,
)
from app.builders.guidance_scoring_types import (
    EQUITY_LARGE_DRAWDOWN_MIN_PCT,
    OPTION_LARGE_DRAWDOWN_MIN_PCT,
)
from app.builders.option_position_guidance_engine import (
    LongOptionGuidanceInputs,
    evaluate_long_option,
)
from app.builders.position_guidance_consistency import (
    GuidanceConsistencyError,
    validate_cross_leg_ordering,
    validate_equity_tiny_trim_guard,
)
from app.builders.position_guidance_cross_leg import apply_cross_leg_sanity
from app.builders.position_guidance_loss_severity import (
    equity_loss_severity,
    option_loss_severity,
)
from app.builders.position_guidance_verdict_normalize import normalize_equity_verdict
from app.models.intelligence_models import IntelligenceSignal
from app.models.position_guidance_models import PositionGuidanceItem, PositionKind
from app.models.trade_decision_models import (
    TradeDecision,
    TradeDecisionReasonBreakdown,
    TradeDecisionRegime,
)
from app.models.position_guidance_models import GuidanceDriverModel


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


def _top_contributor(result):
    return max(result.contributors, key=lambda c: c.points)


def test_equity_one_share_minus_2_pct_is_hold_not_trim():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="AMZN",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=75, env="NEUTRAL"),
            signals=[],
            alert_reasons=[],
            position_weight_pct=0.5,
            open_profit_loss_pct=-2.0,
            position_quantity=1.0,
        )
    )
    assert result.verdict == "HOLD"
    assert result.verdict != "TRIM"
    top = _top_contributor(result)
    assert result.primary_driver.points == top.points
    assert top.label in result.primary_reason
    assert "Large drawdown" not in result.primary_reason


def test_long_option_minus_22_review_close_minimum():
    result = evaluate_long_option(
        LongOptionGuidanceInputs(
            position_kind="LONG_CALL",
            thesis="BULLISH",
            dte=30,
            pnl_pct=-22.0,
            moneyness="OTM",
            alert_reasons=[],
        )
    )
    assert result.verdict in {"REVIEW_CLOSE", "CLOSE"}
    assert result.verdict != "HOLD"
    top = _top_contributor(result)
    assert result.primary_driver.points == top.points


def test_option_more_urgent_than_tiny_equity_loss():
    equity = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="AMZN",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=85, env="FAVORABLE"),
            signals=[],
            alert_reasons=[],
            position_weight_pct=0.5,
            open_profit_loss_pct=-2.0,
            position_quantity=1.0,
        )
    )
    option = evaluate_long_option(
        LongOptionGuidanceInputs(
            position_kind="LONG_CALL",
            thesis="BULLISH",
            dte=25,
            pnl_pct=-22.0,
            moneyness="OTM",
            alert_reasons=[],
        )
    )
    assert equity.verdict == "HOLD"
    assert option.urgency > equity.exit_urgency
    validate_cross_leg_ordering(
        equity_verdict=equity.verdict,
        equity_urgency=equity.exit_urgency,
        equity_rank=50,
        option_verdict=option.verdict,
        option_urgency=option.urgency,
        option_rank=80,
        equity_pnl_pct=-2.0,
        option_pnl_pct=-22.0,
    )


def test_loss_severity_cross_leg_thresholds():
    assert equity_loss_severity(-2.0) <= 25
    assert option_loss_severity(-22.0, position_kind="LONG_CALL") >= 70


def test_validate_tiny_trim_fails():
    with pytest.raises(GuidanceConsistencyError):
        validate_equity_tiny_trim_guard(
            verdict="TRIM",
            open_profit_loss_pct=-2.0,
            position_weight_pct=0.5,
        )


def test_normalize_single_share_trim_to_exit():
    assert normalize_equity_verdict("TRIM", 1.0) == "EXIT"


def test_large_drawdown_thresholds():
    assert EQUITY_LARGE_DRAWDOWN_MIN_PCT == -10.0
    assert OPTION_LARGE_DRAWDOWN_MIN_PCT == -20.0


def _guidance_item(
    *,
    kind: PositionKind,
    verdict: str,
    urgency: int,
    rank: int,
    pnl: float,
) -> PositionGuidanceItem:
    driver = GuidanceDriverModel(
        code="STABLE_POSITION",
        label="Stable position",
        points=1.0,
        detail="test",
    )
    return PositionGuidanceItem(
        positionKey=f"k-{kind}",
        positionKind=kind,
        displayLabel=kind,
        instrumentSymbol="AMZN",
        underlyingSymbol="AMZN",
        quantity=1.0,
        marketValue=100.0,
        openProfitLossPct=pnl,
        verdict=verdict,
        confidence="medium",
        urgency=urgency,
        relativeRiskRank=rank,
        justification="Stable position",
        primaryDriver=driver,
        primaryReason="Stable position: test",
    )


def test_cross_leg_sanity_boosts_option_ranking():
    items = [
        _guidance_item(
            kind="EQUITY_LONG",
            verdict="TRIM",
            urgency=30,
            rank=40,
            pnl=-2.0,
        ),
        _guidance_item(
            kind="LONG_CALL",
            verdict="REVIEW_CLOSE",
            urgency=55,
            rank=35,
            pnl=-22.0,
        ),
    ]
    out = apply_cross_leg_sanity(items)
    opt = next(i for i in out if i.position_kind == "LONG_CALL")
    eq = next(i for i in out if i.position_kind == "EQUITY_LONG")
    assert opt.cross_leg_sanity is True
    assert opt.relative_risk_rank > eq.relative_risk_rank
