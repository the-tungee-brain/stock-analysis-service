from unittest.mock import MagicMock

from app.services.pattern_analysis_service import PatternAnalysisSnapshot
from app.services.trade_decision_service import build_trade_decision
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
        return {"rank": 12}

    def count_ranking_results(self, _run_id):
        return 2000


def test_trade_decision_reuses_pattern_analysis_prediction(monkeypatch):
    monkeypatch.setattr(
        "app.services.trade_decision_service.open_store",
        lambda _cfg: _FakeRankingStore(),
    )
    monkeypatch.setattr(
        "app.services.trade_decision_service.default_config",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "app.services.trade_decision_service.build_pattern_intelligence_payload",
        MagicMock(side_effect=AssertionError("old pattern builder should not run")),
    )
    monkeypatch.setattr(
        "app.services.trade_decision_service._forecast_indicators_payload",
        MagicMock(side_effect=AssertionError("predict_for_symbol should not run")),
    )

    pattern_analysis_service = MagicMock()
    pattern_analysis_service.get_or_build.return_value = PatternAnalysisSnapshot(
        cache_key="pattern:AAPL",
        prediction_payload=_prediction_payload("AAPL"),
        pattern_intelligence=_pattern_intelligence_payload("AAPL"),
    )
    loaded_model = MagicMock()

    decision = build_trade_decision(
        "AAPL",
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
    )

    pattern_analysis_service.get_or_build.assert_called_once_with("AAPL", loaded_model)
    payload = decision.model_dump(mode="json", by_alias=True)
    assert payload["symbol"] == "AAPL"
    assert "tradeQualityScore" in payload
    assert "reasonBreakdown" in payload
    assert payload["regime"]["regimeId"] == "risk_on_trend"
