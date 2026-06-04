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
    build_production_scan_universe,
    compute_stop_distance_pct,
    is_tradable_candidate,
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


def test_build_production_scan_universe_alphabetical_cap_and_raw_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_MAX_UNIVERSE", "2")

    class _FakeStore:
        def active_snapshot_id(self) -> str:
            return "2026-01-01"

        def load_universe_symbols(self, snapshot_id: str | None = None) -> list[str]:
            assert snapshot_id == "2026-01-01"
            return ["ZZZ", "AAA", "BBB", "CCC"]

    def _raw_exists(symbol: str) -> bool:
        return symbol in {"AAA", "BBB", "CCC"}

    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scanner_service.open_store",
        lambda _cfg: _FakeStore(),
    )
    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scanner_service.raw_exists",
        _raw_exists,
    )

    info = build_production_scan_universe(sample_size=50)
    assert info.total_available_symbols == 3
    assert info.scan_cap == 2
    assert info.symbols_scanned == 2
    assert info.excluded_symbols == 1
    assert info.sample_symbols == ["AAA", "BBB"]
    assert "ranking_pipeline.sqlite" in info.universe_source


def test_universe_api_returns_diagnostics(
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_ALERTS_ENABLED", "true")

    class _FakeStore:
        def active_snapshot_id(self) -> str:
            return "snap"

        def load_universe_symbols(self, snapshot_id: str | None = None) -> list[str]:
            return [f"SYM{i:02d}" for i in range(60)]

    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scanner_service.open_store",
        lambda _cfg: _FakeStore(),
    )
    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scanner_service.raw_exists",
        lambda _symbol: True,
    )
    monkeypatch.setenv("MB_SCAN_MAX_UNIVERSE", "500")

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_momentum_breakout_scanner_service] = lambda: scanner

    client = TestClient(app)
    try:
        response = client.get("/api/v1/strategy/momentum-breakout/universe")
        assert response.status_code == 200
        body = response.json()
        assert body["totalAvailableSymbols"] == 60
        assert body["scanCap"] == 500
        assert body["symbolsScanned"] == 60
        assert body["excludedSymbols"] == 0
        assert len(body["sampleSymbols"]) == 50
        assert body["sampleSymbols"][0] == "SYM00"
        assert body["sampleSymbols"][49] == "SYM49"
    finally:
        app.dependency_overrides.clear()


def test_candidates_sort_by_setup_score_then_profit_factor() -> None:
    gate = AlertRiskGateResultDto(
        allowed=True,
        action="ALLOW",
        reasons=[],
        recommendedPositionRiskPct=0.01,
        alertPriority="HIGH",
    )
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
        stop_distance_pct=10.0,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="BULL",
        risk_gate=gate,
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
        stop_distance_pct=10.0,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="BULL",
        risk_gate=gate,
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
        stop_distance_pct=10.0,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="BULL",
        risk_gate=gate,
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
        stop_distance_pct=10.0,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="BULL",
        risk_gate=gate,
    )
    ordered = sorted([low, tie_b, high, tie_a], key=_candidate_sort_key)
    assert [c.symbol for c in ordered] == ["HIGH", "A", "B", "LOW"]


def _tradable_candidate(**overrides) -> _ScanCandidate:
    defaults = dict(
        symbol="OK",
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        risk_reward=2.0,
        historical_win_rate=0.55,
        historical_profit_factor=1.5,
        historical_total_trades=25,
        setup_score=80.0,
        stop_distance_pct=5.0,
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
    defaults.update(overrides)
    return _ScanCandidate(**defaults)


def test_compute_stop_distance_pct() -> None:
    assert compute_stop_distance_pct(100.0, 95.0) == pytest.approx(5.0)


def test_is_tradable_candidate_rules() -> None:
    assert is_tradable_candidate(_tradable_candidate()) is True
    assert (
        is_tradable_candidate(
            _tradable_candidate(
                risk_gate=AlertRiskGateResultDto(
                    allowed=False,
                    action="BLOCK",
                    reasons=["blocked"],
                    recommendedPositionRiskPct=0.0,
                    alertPriority="LOW",
                )
            )
        )
        is False
    )
    assert is_tradable_candidate(_tradable_candidate(historical_profit_factor=1.0)) is False
    assert is_tradable_candidate(_tradable_candidate(historical_total_trades=10)) is False
    assert is_tradable_candidate(_tradable_candidate(stop_distance_pct=7.0)) is True
    assert is_tradable_candidate(_tradable_candidate(stop_distance_pct=8.1)) is False


def test_scan_tradable_only_filters_response(
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = _tradable_candidate(symbol="ALLOWED")
    blocked = _tradable_candidate(
        symbol="BLOCKED",
        risk_gate=AlertRiskGateResultDto(
            allowed=False,
            action="BLOCK",
            reasons=["blocked"],
            recommendedPositionRiskPct=0.0,
            alertPriority="LOW",
        ),
    )
    monkeypatch.setattr(
        scanner,
        "_collect_candidates",
        lambda _symbols: [allowed, blocked],
    )

    all_result = scanner.scan(symbols="X", limit=10, tradable_only=False)
    assert all_result.valid_setups_found == 2
    assert all_result.tradable_candidates_found == 1
    assert all_result.blocked_candidates_count == 1
    assert len(all_result.candidates) == 2

    tradable_result = scanner.scan(symbols="X", limit=10, tradable_only=True)
    assert tradable_result.valid_setups_found == 2
    assert tradable_result.tradable_candidates_found == 1
    assert tradable_result.blocked_candidates_count == 1
    assert len(tradable_result.candidates) == 1
    assert tradable_result.candidates[0].symbol == "ALLOWED"
    assert tradable_result.candidates[0].risk_gate.allowed is True


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
    assert candidate.stop_distance_pct >= 0
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
    assert result.valid_setups_found == 1
    assert result.candidates_found == 1
    assert result.candidates[0].symbol == "NVDA"
    assert result.candidates[0].stop_distance_pct >= 0


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
        assert body["validSetupsFound"] >= 1
        assert body["candidatesFound"] >= 1
        assert body["candidates"][0]["symbol"] == "NVDA"
        assert "stopDistancePct" in body["candidates"][0]
        assert "riskGate" in body["candidates"][0]
    finally:
        app.dependency_overrides.clear()


def test_top_candidates_limits_to_twenty(
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_ALERTS_ENABLED", "true")

    captured: list[int] = []

    def _fake_scan(*, symbols: str | None = None, limit: int = 50, **kwargs):
        captured.append(limit)
        return MomentumBreakoutScanResponse(
            scanTime="2025-01-01T00:00:00+00:00",
            totalSymbolsScanned=100,
            validSetupsFound=25,
            tradableCandidatesFound=10,
            blockedCandidatesCount=15,
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
