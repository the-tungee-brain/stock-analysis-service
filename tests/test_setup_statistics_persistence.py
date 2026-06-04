"""Tests for HistoricalTrade persistence, SetupStatistics, and TradePlan UI fields."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trade_planner.backtest.engine import BacktestEngine
from trade_planner.backtest.statistics import aggregate_setup_statistics, sharpe_ratio
from trade_planner.config import BacktestConfig, TrendContinuationConfig
from trade_planner.models import (
    SetupStatistics,
    SimulatedTrade,
    TradeOutcome,
    TradePlan,
    utc_now,
)
from trade_planner.persistence import (
    HistoricalTrade,
    InMemorySetupStatisticsStore,
)
from trade_planner.setups import TrendContinuationSetup
from trade_planner.types import OHLCVBar


def _plan(symbol: str = "TEST", setup: str = "Unit") -> TradePlan:
    return TradePlan(
        symbol=symbol,
        setup_name=setup,
        direction="LONG",
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        risk_reward=2.0,
        confidence_score=50.0,
        generated_at=utc_now(),
    )


def _simulated(
    *,
    ret: float,
    holding: int = 5,
    outcome: TradeOutcome = TradeOutcome.TARGET_HIT,
) -> SimulatedTrade:
    plan = _plan()
    base = date(2024, 3, 1)
    return SimulatedTrade(
        plan=plan,
        signal_date=base,
        entry_date=base + timedelta(days=1),
        exit_date=base + timedelta(days=holding),
        exit_price=100.0 * (1.0 + ret),
        outcome=outcome,
        return_pct=ret,
        holding_days=holding,
    )


class TestAggregateSetupStatistics:
    def test_empty_statistics(self) -> None:
        stats = aggregate_setup_statistics("Momentum", (), symbol="nvda")
        assert stats == SetupStatistics.empty("Momentum", symbol="nvda")

    def test_computes_core_metrics(self) -> None:
        trades = (
            _simulated(ret=0.10),
            _simulated(ret=0.05),
            _simulated(ret=-0.04, outcome=TradeOutcome.STOP_HIT),
            _simulated(ret=-0.02, outcome=TradeOutcome.STOP_HIT),
        )
        stats = aggregate_setup_statistics("Unit", trades, symbol="test")
        assert stats.total_trades == 4
        assert stats.win_rate == pytest.approx(0.5)
        assert stats.average_win == pytest.approx(0.075)
        assert stats.average_loss == pytest.approx(-0.03)
        assert stats.expectancy == pytest.approx(0.0225)
        assert stats.profit_factor == pytest.approx(2.5)
        assert stats.average_holding_days == pytest.approx(5.0)
        assert stats.max_drawdown >= 0.0
        assert stats.sharpe_ratio != 0.0

    def test_profit_factor_infinite_becomes_cap(self) -> None:
        stats = aggregate_setup_statistics("Unit", (_simulated(ret=0.05),), symbol="x")
        assert stats.profit_factor == 999.0


class TestSharpeRatio:
    def test_requires_multiple_trades(self) -> None:
        assert sharpe_ratio([0.05], average_holding_days=5.0) == 0.0

    def test_positive_for_consistent_wins(self) -> None:
        returns = [0.02, 0.03, 0.025, 0.028]
        value = sharpe_ratio(returns, average_holding_days=10.0)
        assert value > 0.0


class TestHistoricalTrade:
    def test_from_simulated_and_roundtrip(self) -> None:
        sim = _simulated(ret=0.08, holding=7)
        historical = HistoricalTrade.from_simulated(sim)
        assert historical.symbol == "TEST"
        assert historical.return_pct == pytest.approx(0.08)
        assert historical.holding_days == 7
        assert historical.trade_id
        roundtrip = historical.to_simulated()
        assert roundtrip.return_pct == historical.return_pct
        assert roundtrip.holding_days == historical.holding_days


class TestInMemorySetupStatisticsStore:
    def test_persists_every_trade(self) -> None:
        store = InMemorySetupStatisticsStore()
        trades = (_simulated(ret=0.05), _simulated(ret=-0.03))
        stats = aggregate_setup_statistics("Unit", trades, symbol="abc")
        from trade_planner.models import BacktestResult

        result = BacktestResult(
            setup_name="Unit",
            symbol="abc",
            trades=trades,
            statistics=stats,
        )
        record = store.save_backtest(result)
        assert len(record.trades) == 2
        assert store.get("Unit", "abc") is record
        assert store.list_trades("Unit", "abc") == record.trades

    def test_multi_setup_per_symbol(self) -> None:
        store = InMemorySetupStatisticsStore()
        for setup in ("Pullback", "Momentum"):
            trades = (_simulated(ret=0.02),)
            stats = aggregate_setup_statistics(setup, trades, symbol="xyz")
            from trade_planner.models import BacktestResult

            store.save_backtest(
                BacktestResult(
                    setup_name=setup,
                    symbol="xyz",
                    trades=trades,
                    statistics=stats,
                )
            )
        records = store.list_for_symbol("xyz")
        assert len(records) == 2
        assert {r.setup_name for r in records} == {"Pullback", "Momentum"}


class TestBacktestEnginePersistence:
    def _bars(self, days: int) -> tuple[OHLCVBar, ...]:
        from tests.test_trade_planner import _uptrend_bars

        return _uptrend_bars(days, daily_return=0.008)

    def test_engine_records_trades_in_store(self) -> None:
        store = InMemorySetupStatisticsStore()
        setup = TrendContinuationSetup(
            TrendContinuationConfig(volume_expansion_ratio=1.0, min_ma_spread_pct=0.0001)
        )
        engine = BacktestEngine(
            BacktestConfig(min_warmup_bars=30, max_holding_days=15),
            store=store,
        )
        result = engine.run(setup, self._bars(120), symbol="up")
        assert result.record_id == (setup.name, "UP")
        record = store.get(setup.name, "UP")
        assert record is not None
        assert len(record.trades) == result.statistics.total_trades
        assert len(record.trades) == len(result.trades)


class TestTradePlanHistoricalUI:
    def test_properties_exposed_when_statistics_attached(self) -> None:
        stats = SetupStatistics(
            setup_name="Unit",
            symbol="NVDA",
            total_trades=10,
            win_rate=0.6,
            expectancy=0.01,
            average_return=0.01,
            average_win=0.05,
            average_loss=-0.03,
            average_holding_days=8.5,
            profit_factor=1.8,
            max_drawdown=0.12,
            sharpe_ratio=1.2,
        )
        plan = _plan().with_historical_statistics(stats)
        assert plan.historical_win_rate == pytest.approx(0.6)
        assert plan.historical_profit_factor == pytest.approx(1.8)
        assert plan.historical_average_holding_days == pytest.approx(8.5)
        assert plan.historical_total_trades == 10

    def test_properties_none_without_statistics(self) -> None:
        plan = _plan()
        assert plan.historical_win_rate is None
        assert plan.historical_total_trades is None
