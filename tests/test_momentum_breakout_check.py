"""Momentum Breakout single-symbol check API."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.dependencies.service_dependencies import get_momentum_breakout_check_service
from app.main import app
from app.services.strategy.momentum_breakout_check_service import (
    MomentumBreakoutCheckService,
)
from data.benchmarks import BENCHMARK_SYMBOL
from trade_planner.types import OHLCVBar
from tests.test_momentum_breakout_setup import _aligned_trend_series, _test_config
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
def check_service() -> MomentumBreakoutCheckService:
    return MomentumBreakoutCheckService(
        setup=MomentumBreakoutSetup(_test_config()),
    )


def test_check_data_unavailable(check_service: MomentumBreakoutCheckService) -> None:
    result = check_service.check("MISSING")
    assert result.status == "DATA_UNAVAILABLE"
    assert result.can_track_breakout_plan is False


def test_check_tradable_breakout(
    check_service: MomentumBreakoutCheckService,
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
        "app.services.strategy.momentum_breakout_check_service.load_symbol",
        _fake_load,
    )

    result = check_service.check("NVDA")
    assert result.status in {"TRADABLE_BREAKOUT", "REJECTED_BREAKOUT"}
    assert result.entry_price is not None
    assert result.stop_price is not None
    assert result.target_price is not None
    assert result.risk_gate is not None


def test_check_no_breakout_setup(
    check_service: MomentumBreakoutCheckService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock, bench = _aligned_trend_series(120)
    stock_df = _bars_to_df(stock)
    stock_df.iloc[-1, stock_df.columns.get_loc("close")] = 1.0
    bench_df = _bars_to_df(bench)

    def _fake_load(symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym == BENCHMARK_SYMBOL:
            return bench_df
        return stock_df

    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_check_service.load_symbol",
        _fake_load,
    )

    result = check_service.check("WEAK")
    assert result.status == "NO_BREAKOUT_SETUP"
    assert len(result.failed_setup_rules) > 0
    assert result.can_track_breakout_plan is False


def test_check_trims_stock_history_to_benchmark_start(
    check_service: MomentumBreakoutCheckService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock, bench = _aligned_trend_series(120)
    early_stock = tuple(
        OHLCVBar(
            trading_date=stock[0].trading_date - timedelta(days=offset),
            open=stock[0].open * 0.97,
            high=stock[0].high * 0.97,
            low=stock[0].low * 0.97,
            close=stock[0].close * 0.97,
            volume=stock[0].volume,
        )
        for offset in (3, 2, 1)
    )
    stock_df = _bars_to_df((*early_stock, *stock))
    bench_df = _bars_to_df(bench)

    def _fake_load(symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym == BENCHMARK_SYMBOL:
            return bench_df
        if sym == "NVDA":
            return stock_df
        raise FileNotFoundError(sym)

    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_check_service.load_symbol",
        _fake_load,
    )

    result = check_service.check("NVDA")
    assert result.status in {
        "TRADABLE_BREAKOUT",
        "REJECTED_BREAKOUT",
        "NO_BREAKOUT_SETUP",
    }
    assert result.status != "DATA_UNAVAILABLE"


def test_check_api(
    check_service: MomentumBreakoutCheckService,
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
        "app.services.strategy.momentum_breakout_check_service.load_symbol",
        _fake_load,
    )

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_momentum_breakout_check_service] = lambda: check_service

    client = TestClient(app)
    try:
        response = client.get("/api/v1/strategy/momentum-breakout/check/NVDA")
        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "NVDA"
        assert body["status"] in {
            "TRADABLE_BREAKOUT",
            "REJECTED_BREAKOUT",
            "NO_BREAKOUT_SETUP",
        }
        assert "verdictTitle" in body
    finally:
        app.dependency_overrides.clear()
