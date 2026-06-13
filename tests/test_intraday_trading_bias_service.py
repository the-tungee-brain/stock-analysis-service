from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user, get_current_user_id
from app.dependencies.adapter_dependencies import get_yfinance_adapter
from app.dependencies.service_dependencies import (
    get_pattern_analysis_service,
    get_pattern_loaded_model,
    get_research_events_service,
)
from app.builders.intraday_trading_bias_engine import evaluate_intraday_trading_bias
from app.main import app
from app.services import intraday_trading_bias_service as intraday_service_module
from app.services.pattern_analysis_service import PatternAnalysisSnapshot
from tests.test_symbol_intelligence_route import (
    _pattern_intelligence_payload,
    _prediction_payload,
)

ET = ZoneInfo("America/New_York")


class _FakeYFinanceAdapter:
    def __init__(self, *, empty_market: bool = False) -> None:
        self.calls: list[tuple[str, str, str, bool]] = []
        self.empty_market = empty_market

    def get_history(
        self,
        symbol: str,
        *,
        period: str,
        interval: str,
        auto_adjust: bool = True,
        prepost: bool = False,
    ):
        del auto_adjust
        self.calls.append((symbol, period, interval, prepost))
        if self.empty_market and symbol in {"SPY", "QQQ"}:
            return pd.DataFrame()
        return _history_frame()


def _auth_user():
    class _FakeUser:
        identity_sub = "user-1"

    return _FakeUser()


def _history_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            datetime(2026, 6, 5, 8, 30, tzinfo=ET),
            datetime(2026, 6, 5, 9, 0, tzinfo=ET),
            datetime(2026, 6, 5, 9, 30, tzinfo=ET),
            datetime(2026, 6, 5, 9, 35, tzinfo=ET),
            datetime(2026, 6, 5, 9, 40, tzinfo=ET),
            datetime(2026, 6, 5, 9, 45, tzinfo=ET),
            datetime(2026, 6, 5, 9, 50, tzinfo=ET),
            datetime(2026, 6, 5, 9, 55, tzinfo=ET),
            datetime(2026, 6, 5, 10, 0, tzinfo=ET),
            datetime(2026, 6, 5, 10, 5, tzinfo=ET),
            datetime(2026, 6, 5, 10, 10, tzinfo=ET),
        ]
    )
    return pd.DataFrame(
        {
            "Open": [97, 98, 100, 100.5, 101, 101.7, 102, 102.5, 102.8, 103.8, 104.7],
            "High": [98, 100, 101, 101.5, 102, 102.4, 102.8, 103, 104, 105, 106],
            "Low": [96, 97, 99, 100, 100.8, 101.4, 101.8, 102.2, 102.6, 103.5, 104.5],
            "Close": [97.5, 99.5, 100.5, 101, 101.7, 102, 102.5, 102.8, 103.8, 104.7, 105.8],
            "Volume": [500, 600, 800, 900, 950, 1000, 1050, 1100, 1600, 1900, 2200],
        },
        index=index,
    )


def test_intraday_trading_bias_route_returns_stable_shape_and_uses_5m_prepost():
    fake_adapter = _FakeYFinanceAdapter()
    pattern_analysis_service = MagicMock()
    pattern_analysis_service.get_or_build.return_value = PatternAnalysisSnapshot(
        cache_key="pattern:AAPL",
        prediction_payload=_prediction_payload("AAPL"),
        pattern_intelligence=_pattern_intelligence_payload("AAPL"),
    )
    research_events_service = MagicMock()
    research_events_service.get_events.return_value = []

    app.dependency_overrides[get_current_user] = _auth_user
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_yfinance_adapter] = lambda: fake_adapter
    app.dependency_overrides[get_pattern_analysis_service] = (
        lambda: pattern_analysis_service
    )
    app.dependency_overrides[get_pattern_loaded_model] = lambda: MagicMock()
    app.dependency_overrides[get_research_events_service] = (
        lambda: research_events_service
    )

    client = TestClient(app)
    try:
        response = client.get("/api/v1/research/intraday-trading-bias?symbol=aapl")
        assert response.status_code == 200
        payload = response.json()
        assert payload["bias"] in {"Bullish", "Neutral", "Bearish"}
        assert payload["confidence"] in {"High", "Medium", "Low"}
        assert payload["horizon"] == "Intraday"
        assert payload["provider"] == "yfinance"
        assert payload["isRealtime"] is False
        assert "setupType" in payload
        assert "stalenessSeconds" in payload
        assert ("AAPL", "5d", "5m", True) in fake_adapter.calls
        assert ("SPY", "5d", "5m", True) in fake_adapter.calls
    finally:
        app.dependency_overrides.clear()


def test_intraday_trading_bias_route_missing_market_bars_does_not_fail():
    fake_adapter = _FakeYFinanceAdapter(empty_market=True)
    pattern_analysis_service = MagicMock()
    pattern_analysis_service.get_or_build.return_value = PatternAnalysisSnapshot(
        cache_key="pattern:AAPL",
        prediction_payload=_prediction_payload("AAPL"),
        pattern_intelligence=_pattern_intelligence_payload("AAPL"),
    )
    research_events_service = MagicMock()
    research_events_service.get_events.return_value = []

    app.dependency_overrides[get_current_user] = _auth_user
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_yfinance_adapter] = lambda: fake_adapter
    app.dependency_overrides[get_pattern_analysis_service] = (
        lambda: pattern_analysis_service
    )
    app.dependency_overrides[get_pattern_loaded_model] = lambda: MagicMock()
    app.dependency_overrides[get_research_events_service] = (
        lambda: research_events_service
    )

    client = TestClient(app)
    try:
        response = client.get("/api/v1/research/intraday-trading-bias?symbol=aapl")
        assert response.status_code == 200
        payload = response.json()
        assert "SPY/QQQ intraday market bars unavailable" in payload["dataGaps"]
        assert payload["alignment"]["market"] == "mixed"
    finally:
        app.dependency_overrides.clear()


def test_intraday_trading_bias_route_supports_core_etfs_without_company_metadata(
    monkeypatch,
):
    fake_adapter = _FakeYFinanceAdapter()
    pattern_analysis_service = MagicMock()
    pattern_analysis_service.get_or_build.side_effect = RuntimeError(
        "company metadata unavailable"
    )
    research_events_service = MagicMock()
    research_events_service.get_events.side_effect = RuntimeError(
        "events unavailable"
    )

    def evaluate_with_fresh_now(inputs):
        return evaluate_intraday_trading_bias(
            type(inputs)(
                symbol=inputs.symbol,
                bars=inputs.bars,
                market_bars=inputs.market_bars,
                support=inputs.support,
                resistance=inputs.resistance,
                catalyst=inputs.catalyst,
                data_gaps=inputs.data_gaps,
                warnings=inputs.warnings,
                now=datetime(2026, 6, 5, 10, 15, tzinfo=ET),
            )
        )

    monkeypatch.setattr(
        intraday_service_module,
        "evaluate_intraday_trading_bias",
        evaluate_with_fresh_now,
    )

    app.dependency_overrides[get_current_user] = _auth_user
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_yfinance_adapter] = lambda: fake_adapter
    app.dependency_overrides[get_pattern_analysis_service] = (
        lambda: pattern_analysis_service
    )
    app.dependency_overrides[get_pattern_loaded_model] = lambda: MagicMock()
    app.dependency_overrides[get_research_events_service] = (
        lambda: research_events_service
    )

    client = TestClient(app)
    try:
        for symbol in ("SPY", "QQQ", "IWM", "DIA"):
            response = client.get(
                f"/api/v1/research/intraday-trading-bias?symbol={symbol}"
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["horizon"] == "Intraday"
            assert payload["provider"] == "yfinance"
            assert payload["levels"]["openRangeHigh"] is not None
            assert payload["levels"]["openRangeLow"] is not None
            assert payload["levels"]["vwap"] is not None
            assert "Daily support/resistance unavailable" in payload["dataGaps"]
            assert (symbol, "5d", "5m", True) in fake_adapter.calls
    finally:
        app.dependency_overrides.clear()
