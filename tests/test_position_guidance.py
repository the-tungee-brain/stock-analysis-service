"""Tests for position-level guidance engines."""

from __future__ import annotations

from app.builders.option_position_guidance_engine import (
    LongOptionGuidanceInputs,
    ShortOptionGuidanceInputs,
    evaluate_long_option,
    evaluate_short_option,
)
from app.builders.position_guidance_support import classify_position_kind
from app.builders.symbol_thesis_engine import evaluate_symbol_thesis
from app.models.schwab_models import Instrument, Position
from app.models.trade_decision_models import (
    TradeDecision,
    TradeDecisionReasonBreakdown,
    TradeDecisionRegime,
)


def _trade(*, score: int = 72, env: str = "FAVORABLE") -> TradeDecision:
    return TradeDecision(
        symbol="TEST",
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


def _option_position(
    *,
    underlying: str,
    put_call: str,
    long_qty: float = 0,
    short_qty: float = 0,
) -> Position:
    return Position(
        shortQuantity=short_qty,
        averagePrice=1.0,
        currentDayProfitLoss=0,
        currentDayProfitLossPercentage=0,
        longQuantity=long_qty,
        settledLongQuantity=long_qty,
        settledShortQuantity=short_qty,
        instrument=Instrument(
            assetType="OPTION",
            cusip="OPT",
            symbol=f"{underlying}_{put_call}",
            putCall=put_call,
            underlyingSymbol=underlying,
            strikePrice=100.0,
            expirationDate="2026-06-20",
        ),
        marketValue=500.0,
        maintenanceRequirement=0,
        currentDayCost=0,
        openProfitLossPct=-25.0,
    )


def test_symbol_thesis_bullish():
    result = evaluate_symbol_thesis(_trade(score=80, env="FAVORABLE"))
    assert result.thesis == "BULLISH"


def test_symbol_thesis_bearish():
    result = evaluate_symbol_thesis(_trade(score=30, env="AVOID"))
    assert result.thesis == "BEARISH"


def test_long_call_conflicts_with_bearish_thesis():
    result = evaluate_long_option(
        LongOptionGuidanceInputs(
            position_kind="LONG_CALL",
            thesis="BEARISH",
            dte=4,
            pnl_pct=-22.0,
            moneyness="OTM",
            alert_reasons=[],
        )
    )
    assert result.verdict in {"REVIEW_CLOSE", "CLOSE"}


def test_short_option_assignment_risk():
    result = evaluate_short_option(
        ShortOptionGuidanceInputs(
            position_kind="SHORT_PUT",
            thesis="NEUTRAL",
            dte=3,
            pnl_pct=-10.0,
            moneyness="ITM",
            assignment_risk="high",
            option_strategy="cash_secured_put",
            alert_reasons=[],
        )
    )
    assert result.verdict in {"REVIEW_ASSIGNMENT_RISK", "CLOSE", "ROLL"}


def test_classify_option_kinds():
    long_call = _option_position(underlying="AAPL", put_call="CALL", long_qty=1)
    short_put = _option_position(underlying="AAPL", put_call="PUT", short_qty=1)
    assert classify_position_kind(long_call, "AAPL") == "LONG_CALL"
    assert classify_position_kind(short_put, "AAPL") == "SHORT_PUT"
