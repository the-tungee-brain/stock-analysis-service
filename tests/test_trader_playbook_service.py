from __future__ import annotations

from copy import deepcopy
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user, get_current_user_id
from app.dependencies.service_dependencies import (
    get_pattern_analysis_service,
    get_pattern_loaded_model,
    get_research_events_service,
)
from app.main import app
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
from app.services.pattern_analysis_service import PatternAnalysisSnapshot
from app.services.trader_playbook_service import build_trader_playbook
from tests.test_symbol_intelligence_route import (
    _pattern_intelligence_payload,
    _prediction_payload,
)


def _auth_user():
    class _FakeUser:
        identity_sub = "user-1"

    return _FakeUser()


def _bias() -> TradingBiasResponse:
    return TradingBiasResponse(
        symbol="AAPL",
        bias="Bullish",
        confidence="Medium",
        action="Watch",
        bullish_factors=[],
        bearish_factors=[],
        invalidation=None,
        levels=TradingBiasLevels(support=100, resistance=105, breakout_level=105),
        alignment=TradingBiasAlignment(
            market_regime="aligned",
            relative_strength="aligned",
            pattern_trend="aligned",
            volume="confirmed",
            catalyst="none",
        ),
        data_gaps=[],
    )


def _decision() -> TradeDecision:
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
        action="WAIT_FOR_SETUP",
        reason_breakdown=TradeDecisionReasonBreakdown(),
    )


def _payload() -> dict:
    payload = deepcopy(_pattern_intelligence_payload("AAPL"))
    payload["trend_context"]["close"] = 103
    payload["chart_intelligence"]["support_zones"] = [
        {"price_low": 98, "price_high": 99}
    ]
    payload["chart_intelligence"]["resistance_zones"] = [
        {"price_low": 105, "price_high": 106}
    ]
    return payload


def test_trader_playbook_service_reuses_pattern_analysis_snapshot(monkeypatch):
    monkeypatch.setattr(
        "app.services.trader_playbook_service.build_trading_bias",
        MagicMock(return_value=_bias()),
    )
    monkeypatch.setattr(
        "app.services.trader_playbook_service.build_trade_decision",
        MagicMock(return_value=_decision()),
    )
    monkeypatch.setattr(
        "app.services.trader_playbook_service.build_pattern_intelligence_payload",
        MagicMock(side_effect=AssertionError("fallback builder should not run")),
    )

    pattern_analysis_service = MagicMock()
    pattern_analysis_service.get_or_build.return_value = PatternAnalysisSnapshot(
        cache_key="pattern:AAPL",
        prediction_payload=_prediction_payload("AAPL"),
        pattern_intelligence=_payload(),
    )

    result = build_trader_playbook(
        "aapl",
        loaded_model=MagicMock(),
        pattern_analysis_service=pattern_analysis_service,
        research_events_service=None,
    )

    pattern_analysis_service.get_or_build.assert_called_once()
    assert result.direction == "Bullish"
    assert result.horizon == "1-5 sessions"
    assert result.data_mode == "daily"
    assert result.best_setup == "BreakoutContinuation"


def test_trader_playbook_route_returns_stable_shape(monkeypatch):
    monkeypatch.setattr(
        "app.services.trader_playbook_service.build_trading_bias",
        MagicMock(return_value=_bias()),
    )
    monkeypatch.setattr(
        "app.services.trader_playbook_service.build_trade_decision",
        MagicMock(return_value=_decision()),
    )

    pattern_analysis_service = MagicMock()
    pattern_analysis_service.get_or_build.return_value = PatternAnalysisSnapshot(
        cache_key="pattern:AAPL",
        prediction_payload=_prediction_payload("AAPL"),
        pattern_intelligence=_payload(),
    )
    research_events_service = MagicMock()
    research_events_service.get_events.return_value = []

    app.dependency_overrides[get_current_user] = _auth_user
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_pattern_analysis_service] = (
        lambda: pattern_analysis_service
    )
    app.dependency_overrides[get_pattern_loaded_model] = lambda: MagicMock()
    app.dependency_overrides[get_research_events_service] = (
        lambda: research_events_service
    )

    client = TestClient(app)
    try:
        response = client.get("/api/v1/research/trader-playbook?symbol=aapl")
        assert response.status_code == 200
        payload = response.json()
        assert payload["direction"] in {"Bullish", "Neutral", "Bearish"}
        assert payload["horizon"] == "1-5 sessions"
        assert payload["dataMode"] == "daily"
        assert set(payload["conditions"]) == {"validIf", "invalidIf"}
        assert "riskRewardLabel" in payload["risk"]
    finally:
        app.dependency_overrides.clear()


def test_trader_playbook_service_missing_pattern_returns_no_setup(monkeypatch):
    monkeypatch.setattr(
        "app.services.trader_playbook_service.build_trading_bias",
        MagicMock(return_value=_bias()),
    )
    monkeypatch.setattr(
        "app.services.trader_playbook_service.build_trade_decision",
        MagicMock(return_value=_decision()),
    )
    monkeypatch.setattr(
        "app.services.trader_playbook_service.build_pattern_intelligence_payload",
        MagicMock(return_value=None),
    )

    result = build_trader_playbook(
        "AAPL",
        loaded_model=None,
        pattern_analysis_service=None,
        research_events_service=None,
    )

    assert result.best_setup == "None"
    assert result.status == "NoSetup"
    assert "Pattern analysis unavailable" in result.data_gaps
