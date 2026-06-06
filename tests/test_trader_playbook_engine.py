from __future__ import annotations

from copy import deepcopy

from app.builders.trader_playbook_engine import (
    TraderPlaybookInputs,
    evaluate_trader_playbook,
)
from app.models.trade_decision_models import (
    TradeDecision,
    TradeDecisionReasonBreakdown,
    TradeDecisionRegime,
)
from app.models.trading_bias_models import (
    TradingBiasAlignment,
    TradingBiasLevels,
    TradingBiasResponse,
)
from tests.test_symbol_intelligence_route import _pattern_intelligence_payload


def _bias(direction: str = "Bullish") -> TradingBiasResponse:
    return TradingBiasResponse(
        symbol="AAPL",
        bias=direction,
        confidence="Medium",
        action="Watch",
        bullish_factors=[],
        bearish_factors=[],
        invalidation=None,
        levels=TradingBiasLevels(support=95, resistance=105, breakout_level=105),
        alignment=TradingBiasAlignment(
            market_regime="aligned",
            relative_strength="aligned",
            pattern_trend="aligned" if direction == "Bullish" else "mixed",
            volume="confirmed",
            catalyst="none",
        ),
        data_gaps=[],
    )


def _decision(action: str = "WAIT_FOR_SETUP") -> TradeDecision:
    return TradeDecision(
        symbol="AAPL",
        as_of_date="2026-06-04",
        regime=TradeDecisionRegime(
            regime_id="risk_on_trend",
            trade_environment="FAVORABLE",
        ),
        trade_quality_score=72,
        score_bucket="SETUP",
        verdict="WATCHLIST",
        action=action,
        reason_breakdown=TradeDecisionReasonBreakdown(),
    )


def _payload(
    *,
    close: float = 103,
    support: float = 99,
    resistance: float = 105,
    trend_bias: str = "uptrend",
) -> dict:
    payload = deepcopy(_pattern_intelligence_payload("AAPL"))
    payload["trend_context"]["close"] = close
    payload["trend_context"]["trend_bias"] = trend_bias
    payload["chart_intelligence"]["support_zones"] = [
        {"price_low": support - 1, "price_high": support}
    ]
    payload["chart_intelligence"]["resistance_zones"] = [
        {"price_low": resistance, "price_high": resistance + 1}
    ]
    payload["chart_intelligence"]["summary"]["outlook"]["label"] = "Bullish"
    return payload


def test_bullish_breakout_not_crossed_returns_waiting_with_valid_condition():
    result = evaluate_trader_playbook(
        TraderPlaybookInputs(
            symbol="AAPL",
            trading_bias=_bias("Bullish"),
            trade_decision=_decision(),
            pattern_intelligence=_payload(close=103, support=99, resistance=105),
        )
    )

    assert result.best_setup == "BreakoutContinuation"
    assert result.status == "Waiting"
    assert result.levels.entry == 105
    assert result.levels.stop == 99
    assert any("above breakout level $105.00" in item for item in result.conditions.valid_if)


def test_bullish_crossed_breakout_favorable_rr_returns_valid():
    result = evaluate_trader_playbook(
        TraderPlaybookInputs(
            symbol="AAPL",
            trading_bias=_bias("Bullish"),
            trade_decision=_decision("ENTER"),
            pattern_intelligence=_payload(close=106, support=100, resistance=105),
        )
    )

    assert result.best_setup == "BreakoutContinuation"
    assert result.status == "Valid"
    assert result.risk.risk_reward_label == "favorable"
    assert result.risk.r_multiple_target1 == 2


def test_bullish_poor_rr_demotes_to_waiting_with_warning():
    result = evaluate_trader_playbook(
        TraderPlaybookInputs(
            symbol="AAPL",
            trading_bias=_bias("Bullish"),
            trade_decision=_decision(),
            pattern_intelligence=_payload(close=98, support=100, resistance=101),
        )
    )

    assert result.best_setup == "PullbackToSupport"
    assert result.status in {"Waiting", "NoSetup"}
    assert result.risk.risk_reward_label == "poor"
    assert "Risk/reward is poor for this daily setup." in result.warnings


def test_failed_breakout_evidence_returns_failed_breakout_plan():
    payload = _payload(close=99, support=95, resistance=105, trend_bias="mixed")
    payload["chart_intelligence"]["summary"]["outlook"]["label"] = "Bearish"
    payload["chart_intelligence"]["breakout_events"] = [
        {"kind": "failed_breakout", "price": 104, "label": "Failed breakout"}
    ]

    result = evaluate_trader_playbook(
        TraderPlaybookInputs(
            symbol="AAPL",
            trading_bias=_bias("Neutral"),
            trade_decision=_decision(),
            pattern_intelligence=payload,
        )
    )

    assert result.best_setup == "FailedBreakout"
    assert result.status == "Valid"
    assert result.levels.entry == 105
    assert result.levels.stop == 108.15
    assert any("rejected resistance" in item for item in result.conditions.valid_if)


def test_missing_pattern_analysis_returns_no_setup_with_data_gap():
    result = evaluate_trader_playbook(
        TraderPlaybookInputs(
            symbol="AAPL",
            trading_bias=_bias("Bullish"),
            trade_decision=_decision(),
            pattern_intelligence=None,
            data_gaps=["Pattern analysis unavailable"],
        )
    )

    assert result.best_setup == "None"
    assert result.status == "NoSetup"
    assert "Pattern analysis unavailable" in result.data_gaps


def test_no_entry_without_stop_and_no_target_without_entry_stop():
    result = evaluate_trader_playbook(
        TraderPlaybookInputs(
            symbol="AAPL",
            trading_bias=_bias("Neutral"),
            trade_decision=_decision(),
            pattern_intelligence=_pattern_intelligence_payload("AAPL"),
        )
    )

    if result.levels.entry is None or result.levels.stop is None:
        assert result.levels.target1 is None
        assert result.levels.target2 is None


def test_response_shape_stable():
    result = evaluate_trader_playbook(
        TraderPlaybookInputs(
            symbol="AAPL",
            trading_bias=_bias("Bullish"),
            trade_decision=_decision(),
            pattern_intelligence=_payload(),
        )
    )
    payload = result.model_dump(mode="json", by_alias=True)

    assert set(payload) == {
        "direction",
        "confidence",
        "horizon",
        "dataMode",
        "bestSetup",
        "status",
        "conditions",
        "levels",
        "risk",
        "alignment",
        "reasons",
        "warnings",
        "dataGaps",
    }
    assert set(payload["conditions"]) == {"validIf", "invalidIf"}
    assert set(payload["levels"]) == {
        "entry",
        "stop",
        "target1",
        "target2",
        "support",
        "resistance",
        "breakoutLevel",
    }
    assert set(payload["risk"]) == {
        "riskPerShare",
        "rewardToTarget1",
        "rewardToTarget2",
        "rMultipleTarget1",
        "rMultipleTarget2",
        "riskRewardLabel",
    }
