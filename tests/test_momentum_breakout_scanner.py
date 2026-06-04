"""Momentum Breakout scanner service and API."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.dependencies.service_dependencies import get_momentum_breakout_scanner_service
from app.main import app
from app.models.momentum_breakout_alert_models import AlertRiskGateResultDto
from app.models.momentum_breakout_scan_models import MomentumBreakoutScanResponse
from app.services.strategy.momentum_breakout_scanner_service import (
    MomentumBreakoutScannerService,
    _candidate_sort_key,
    _ScanCandidate,
)
from data.benchmarks import BENCHMARK_SYMBOL
from tests.test_momentum_breakout_setup import (
    _aligned_trend_series,
    _test_config,
)
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup


def _bars_to_df(bars) -> pd.DataFrame:
    index = pd.DatetimeIndex([b.trading_date for b in bars])
    return pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        },
        index=index,
    )


@pytest.fixture
def scanner() -> MomentumBreakoutScannerService:
    return MomentumBreakoutScannerService(
        setup=MomentumBreakoutSetup(_test_config()),
    )


def test_resolve_explicit_symbols(scanner: MomentumBreakoutScannerService) -> None:
    assert scanner.resolve_symbol_list("aapl,msft,AAPL") == ["AAPL", "MSFT"]


def test_candidates_sort_by_setup_score_then_profit_factor() -> None:
    low = _ScanCandidate(
        symbol="LOW",
        entry_price=10.0,
        stop_price=9.0,
        target_price=12.0,
        risk_reward=2.0,
        historical_win_rate=0.5,
        historical_profit_factor=3.0,
        historical_total_trades=10,
        setup_score=50.0,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="BULL",
        risk_gate=AlertRiskGateResultDto(
            allowed=True,
            action="ALLOW",
            reasons=[],
            recommendedPositionRiskPct=0.01,
            alertPriority="HIGH",
        ),
    )
    high = _ScanCandidate(
        symbol="HIGH",
        entry_price=10.0,
        stop_price=9.0,
        target_price=12.0,
        risk_reward=2.0,
        historical_win_rate=0.5,
        historical_profit_factor=1.0,
        historical_total_trades=10,
        setup_score=90.0,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="BULL",
        risk_gate=AlertRiskGateResultDto(
            allowed=True,
            action="ALLOW",
            reasons=[],
            recommendedPositionRiskPct=0.01,
            alertPriority="HIGH",
        ),
    )
    tie_a = _ScanCandidate(
        symbol="A",
        entry_price=10.0,
        stop_price=9.0,
        target_price=12.0,
        risk_reward=2.0,
        historical_win_rate=0.5,
        historical_profit_factor=2.5,
        historical_total_trades=10,
        setup_score=80.0,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="BULL",
        risk_gate=low.risk_gate,
    )
    tie_b = _ScanCandidate(
        symbol="B",
        entry_price=10.0,
        stop_price=9.0,
        target_price=12.0,
        risk_reward=2.0,
        historical_win_rate=0.5,
        historical_profit_factor=1.5,
        historical_total_trades=10,
        setup_score=80.0,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="BULL",
        risk_gate=low.risk_gate,
    )
    ordered = sorted([low, tie_b, high, tie_a], key=_candidate_sort_key)
    assert [c.symbol for c in ordered] == ["HIGH", "A", "B", "LOW"]


def test_evaluate_symbol_returns_candidate(
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock, bench = _aligned_trend_series(120)
    stock_df = _bars_to_df(stock)
    bench_df = _bars_to_df(bench)

    def _fake_load(symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym == BENCHMARK_SYMBOL:
            return bench_df
        if sym == "NVDA":
            return stock_df
        raise FileNotFoundError(sym)

    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scanner_service.load_symbol",
        _fake_load,
    )

    candidate = scanner.evaluate_symbol("NVDA")
    assert candidate is not None
    assert candidate.symbol == "NVDA"
    assert candidate.entry_price > 0
    assert candidate.setup_score > 0
    assert candidate.risk_gate.action in {"ALLOW", "WARN", "SIZE_DOWN", "BLOCK"}
    assert candidate.market_regime is not None
    assert candidate.volume_ratio is not None


def test_scan_with_explicit_symbols(
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock, bench = _aligned_trend_series(120)
    stock_df = _bars_to_df(stock)
    bench_df = _bars_to_df(bench)

    def _fake_load(symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym == BENCHMARK_SYMBOL:
            return bench_df
        if sym == "NVDA":
            return stock_df
        raise FileNotFoundError(sym)

    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scanner_service.load_symbol",
        _fake_load,
    )

    result = scanner.scan(symbols="NVDA,MISSING", limit=10)
    assert result.total_symbols_scanned == 2
    assert result.candidates_found == 1
    assert result.candidates[0].symbol == "NVDA"


def test_scan_api_returns_candidates(
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_ALERTS_ENABLED", "true")

    stock, bench = _aligned_trend_series(120)
    stock_df = _bars_to_df(stock)
    bench_df = _bars_to_df(bench)

    def _fake_load(symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym == BENCHMARK_SYMBOL:
            return bench_df
        if sym == "NVDA":
            return stock_df
        raise FileNotFoundError(sym)

    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scanner_service.load_symbol",
        _fake_load,
    )

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_momentum_breakout_scanner_service] = lambda: scanner

    client = TestClient(app)
    try:
        response = client.get(
            "/api/v1/strategy/momentum-breakout/scan",
            params={"symbols": "NVDA", "limit": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["totalSymbolsScanned"] == 1
        assert body["candidatesFound"] >= 1
        assert body["candidates"][0]["symbol"] == "NVDA"
        assert "riskGate" in body["candidates"][0]
    finally:
        app.dependency_overrides.clear()


def test_top_candidates_limits_to_twenty(
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_ALERTS_ENABLED", "true")

    captured: list[int] = []

    def _fake_scan(*, symbols: str | None = None, limit: int = 50):
        captured.append(limit)
        return MomentumBreakoutScanResponse(
            scanTime="2025-01-01T00:00:00+00:00",
            totalSymbolsScanned=100,
            candidatesFound=25,
            candidates=[],
        )

    monkeypatch.setattr(scanner, "scan", _fake_scan)

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_momentum_breakout_scanner_service] = lambda: scanner

    client = TestClient(app)
    try:
        response = client.get("/api/v1/strategy/momentum-breakout/top-candidates")
        assert response.status_code == 200
        assert captured == [20]
    finally:
        app.dependency_overrides.clear()
