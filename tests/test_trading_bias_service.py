from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user, get_current_user_id
from app.dependencies.service_dependencies import (
    get_pattern_analysis_service,
    get_pattern_loaded_model,
    get_research_events_service,
)
from app.main import app
from app.models.intelligence_models import EventTimelineEntry
from app.services.pattern_analysis_service import PatternAnalysisSnapshot
from app.services.trading_bias_service import build_trading_bias
from tests.test_symbol_intelligence_route import (
    _pattern_intelligence_payload,
    _prediction_payload,
)


class _FakeRankingStore:
    def latest_run_id(self):
        return "run-1"

    def get_run_meta(self, _run_id):
        return {"regime_id": "risk_on_trend", "as_of_date": "2026-06-04"}

    def get_symbol_ranking_row(self, _run_id, _symbol):
        return {
            "rank": 12,
            "ml_probability": 0.72,
            "expected_excess_return": 0.02,
            "final_score": 0.78,
        }

    def count_ranking_results(self, _run_id):
        return 2000


def _auth_user():
    class _FakeUser:
        identity_sub = "user-1"

    return _FakeUser()


def test_trading_bias_service_reuses_pattern_analysis_snapshot(monkeypatch):
    monkeypatch.setattr(
        "app.services.trading_bias_service.open_store",
        lambda _cfg: _FakeRankingStore(),
    )
    monkeypatch.setattr(
        "app.services.trading_bias_service.default_config",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "app.services.trading_bias_service.build_pattern_intelligence_payload",
        MagicMock(side_effect=AssertionError("fallback builder should not run")),
    )

    pattern_analysis_service = MagicMock()
    pattern_analysis_service.get_or_build.return_value = PatternAnalysisSnapshot(
        cache_key="pattern:AAPL",
        prediction_payload=_prediction_payload("AAPL"),
        pattern_intelligence=_pattern_intelligence_payload("AAPL"),
    )
    research_events_service = MagicMock()
    research_events_service.get_events.return_value = [
        EventTimelineEntry(
            date="2026-06-04",
            kind="earnings",
            title="Earnings report (beat)",
            detail="EPS surprise +4.0%",
        )
    ]
    loaded_model = MagicMock()

    result = build_trading_bias(
        "aapl",
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
        research_events_service=research_events_service,
    )

    pattern_analysis_service.get_or_build.assert_called_once_with("AAPL", loaded_model)
    research_events_service.get_events.assert_called_once_with(symbol="AAPL")
    payload = result.model_dump(mode="json", by_alias=True)
    assert payload["symbol"] == "AAPL"
    assert payload["horizon"] == "1-5 sessions"
    assert "bullishFactors" in payload
    assert "alignment" in payload


def test_trading_bias_route_returns_stable_shape(monkeypatch):
    monkeypatch.setattr(
        "app.services.trading_bias_service.open_store",
        lambda _cfg: _FakeRankingStore(),
    )
    monkeypatch.setattr(
        "app.services.trading_bias_service.default_config",
        lambda: MagicMock(),
    )

    pattern_analysis_service = MagicMock()
    pattern_analysis_service.get_or_build.return_value = PatternAnalysisSnapshot(
        cache_key="pattern:NVDA",
        prediction_payload=_prediction_payload("NVDA"),
        pattern_intelligence=_pattern_intelligence_payload("NVDA"),
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
        response = client.get("/api/v1/research/trading-bias?symbol=nvda")
        assert response.status_code == 200
        payload = response.json()
        assert payload["symbol"] == "NVDA"
        assert payload["bias"] in {"Bullish", "Neutral", "Bearish"}
        assert payload["confidence"] in {"High", "Medium", "Low"}
        assert payload["horizon"] == "1-5 sessions"
        assert set(payload["alignment"]) == {
            "marketRegime",
            "relativeStrength",
            "patternTrend",
            "volume",
            "catalyst",
        }
    finally:
        app.dependency_overrides.clear()


def test_trading_bias_service_soft_returns_when_pattern_missing(monkeypatch):
    monkeypatch.setattr(
        "app.services.trading_bias_service.open_store",
        lambda _cfg: _FakeRankingStore(),
    )
    monkeypatch.setattr(
        "app.services.trading_bias_service.default_config",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "app.services.trading_bias_service.build_pattern_intelligence_payload",
        lambda *_args, **_kwargs: None,
    )

    result = build_trading_bias(
        "AAPL",
        loaded_model=None,
        pattern_analysis_service=None,
        research_events_service=None,
    )

    assert result.bias == "Neutral"
    assert result.confidence == "Low"
    assert "Pattern analysis unavailable" in result.data_gaps
