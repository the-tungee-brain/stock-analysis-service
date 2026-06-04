"""MomentumBreakoutSetup — rules, unit tests, and backtest integration."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trade_planner.backtest.engine import BacktestEngine
from trade_planner.config import BacktestConfig, MomentumBreakoutConfig
from trade_planner.models import TradeOutcome
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from trade_planner.types import OHLCVBar, StockData


def _bar(
    day: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    start: date | None = None,
) -> OHLCVBar:
    base = start or date(2020, 1, 2)
    return OHLCVBar(
        trading_date=base + timedelta(days=day),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _aligned_trend_series(
    days: int,
    *,
    stock_daily: float = 0.006,
    bench_daily: float = 0.001,
    final_volume: float = 3_000_000.0,
    base_volume: float = 800_000.0,
) -> tuple[tuple[OHLCVBar, ...], tuple[OHLCVBar, ...]]:
    """Stock outperforms benchmark with accelerating RS for high percentile rank."""
    stock: list[OHLCVBar] = []
    bench: list[OHLCVBar] = []
    s_price = 40.0
    b_price = 40.0
    for idx in range(days):
        # Accelerate stock returns late so latest RS ranks above the 80th percentile.
        ramp = stock_daily + (idx / max(days - 1, 1)) * 0.012
        s_price *= 1.0 + ramp
        b_price *= 1.0 + bench_daily
        high_s = s_price * 1.002
        low_s = s_price * 0.996
        vol = final_volume if idx == days - 1 else base_volume
        stock.append(
            _bar(
                idx,
                open_=s_price * 0.999,
                high=high_s,
                low=low_s,
                close=s_price,
                volume=vol,
            )
        )
        bench.append(
            _bar(
                idx,
                open_=b_price,
                high=b_price * 1.001,
                low=b_price * 0.999,
                close=b_price,
                volume=1_000_000.0,
            )
        )
    return tuple(stock), tuple(bench)


def _test_config() -> MomentumBreakoutConfig:
    """Shorter lookbacks for fast, deterministic tests."""
    return MomentumBreakoutConfig(
        sma_fast_days=10,
        sma_slow_days=20,
        high_lookback_days=20,
        high_proximity_pct=0.02,
        volume_avg_days=20,
        volume_ratio_min=1.5,
        rs_lookback_days=20,
        rs_percentile_window=40,
        rs_percentile_min=80.0,
        stop_lookback_days=10,
        entry_buffer=0.01,
        target_risk_reward=2.0,
        require_benchmark=True,
    )


class TestMomentumBreakoutRules:
    def test_valid_when_all_conditions_met(self) -> None:
        stock, bench = _aligned_trend_series(120)
        setup = MomentumBreakoutSetup(_test_config())
        data = StockData.from_bars("NVDA", stock, benchmark_bars=bench)
        assert setup.is_valid(data)

    def test_entry_is_buy_stop_above_high(self) -> None:
        stock, bench = _aligned_trend_series(120)
        setup = MomentumBreakoutSetup(_test_config())
        data = StockData.from_bars("NVDA", stock, benchmark_bars=bench)
        entry = setup.generate_entry(data)
        assert entry == pytest.approx(data.current.high + 0.01)

    def test_stop_is_lowest_low_prior_10_days(self) -> None:
        stock, bench = _aligned_trend_series(120)
        setup = MomentumBreakoutSetup(_test_config())
        data = StockData.from_bars("NVDA", stock, benchmark_bars=bench)
        entry = setup.generate_entry(data)
        assert entry is not None
        stop = setup.generate_stop(data, entry)
        window = data.bars[data.index - 10 : data.index]
        expected = min(bar.low for bar in window)
        assert stop == pytest.approx(expected)

    def test_target_is_two_r(self) -> None:
        stock, bench = _aligned_trend_series(120)
        setup = MomentumBreakoutSetup(_test_config())
        data = StockData.from_bars("NVDA", stock, benchmark_bars=bench)
        entry = setup.generate_entry(data)
        assert entry is not None
        stop = setup.generate_stop(data, entry)
        assert stop is not None
        target = setup.generate_target(data, entry, stop)
        risk = entry - stop
        assert target == pytest.approx(entry + 2.0 * risk)
        assert setup.build_plan(data) is not None
        plan = setup.build_plan(data)
        assert plan is not None
        assert plan.risk_reward == pytest.approx(2.0)
        assert plan.entry_is_stop is True

    def test_fails_without_benchmark_when_required(self) -> None:
        stock, _ = _aligned_trend_series(120)
        setup = MomentumBreakoutSetup(_test_config())
        data = StockData.from_bars("NVDA", stock)
        assert not setup.is_valid(data)

    def test_fails_when_close_below_sma50(self) -> None:
        stock, bench = _aligned_trend_series(120)
        last = stock[-1]
        broken = stock[:-1] + (
            _bar(
                len(stock) - 1,
                open_=last.open,
                high=last.high,
                low=last.low * 0.85,
                close=last.close * 0.80,
                volume=last.volume,
            ),
        )
        setup = MomentumBreakoutSetup(_test_config())
        data = StockData.from_bars("NVDA", broken, benchmark_bars=bench)
        assert not setup.is_valid(data)

    def test_fails_when_volume_ratio_low(self) -> None:
        stock, bench = _aligned_trend_series(
            120, final_volume=500_000.0, base_volume=900_000.0
        )
        setup = MomentumBreakoutSetup(_test_config())
        data = StockData.from_bars("NVDA", stock, benchmark_bars=bench)
        assert not setup.is_valid(data)

    def test_fails_when_not_near_20_day_high(self) -> None:
        stock, bench = _aligned_trend_series(120)
        last = stock[-1]
        far = stock[:-1] + (
            _bar(
                len(stock) - 1,
                open_=last.open,
                high=last.high,
                low=last.low,
                close=last.close * 0.90,
                volume=3_500_000.0,
            ),
        )
        setup = MomentumBreakoutSetup(_test_config())
        data = StockData.from_bars("NVDA", far, benchmark_bars=bench)
        assert not setup.is_valid(data)


class TestMomentumBreakoutBacktest:
    def test_backtest_entry_fill_and_target_hit(self) -> None:
        stock, bench = _aligned_trend_series(120)
        setup = MomentumBreakoutSetup(_test_config())
        signal_index = len(stock) - 1
        data = StockData.from_bars("NVDA", stock, index=signal_index, benchmark_bars=bench)
        plan = setup.build_plan(data)
        assert plan is not None

        signal_high = stock[signal_index].high
        entry_trigger = plan.entry_price
        next_day = len(stock)
        fill_day_stock = _bar(
            next_day,
            open_=signal_high,
            high=entry_trigger + 1.0,
            low=signal_high - 0.5,
            close=entry_trigger + 0.5,
            volume=2_000_000.0,
        )
        target_day_stock = _bar(
            next_day + 1,
            open_=plan.target_price - 1.0,
            high=plan.target_price + 2.0,
            low=plan.stop_price + 0.5,
            close=plan.target_price + 1.0,
            volume=2_000_000.0,
        )
        extended = stock + (fill_day_stock, target_day_stock)
        last_bench = bench[-1]
        extended_bench = bench + (
            _bar(
                next_day,
                open_=last_bench.close,
                high=last_bench.close * 1.001,
                low=last_bench.close * 0.999,
                close=last_bench.close,
                volume=1_000_000.0,
            ),
            _bar(
                next_day + 1,
                open_=last_bench.close,
                high=last_bench.close * 1.001,
                low=last_bench.close * 0.999,
                close=last_bench.close,
                volume=1_000_000.0,
            ),
        )

        engine = BacktestEngine(
            BacktestConfig(
                min_warmup_bars=setup.required_warmup_bars(),
                max_holding_days=10,
                slippage_bps=0.0,
            )
        )
        result = engine.run(setup, extended, symbol="NVDA", benchmark_bars=extended_bench)
        targets = [t for t in result.trades if t.outcome == TradeOutcome.TARGET_HIT]
        assert len(targets) >= 1
        trade = targets[-1]
        assert trade.plan.entry_is_stop is True
        assert trade.entry_date >= trade.signal_date
        assert trade.return_pct > 0

    def test_backtest_integration_service_path(self) -> None:
        """End-to-end: scan history, simulate stop entries, aggregate stats."""
        stock, bench = _aligned_trend_series(100, stock_daily=0.007, bench_daily=0.001)
        setup = MomentumBreakoutSetup(_test_config())
        engine = BacktestEngine(BacktestConfig(min_warmup_bars=35, max_holding_days=15))
        result = engine.run(setup, stock, symbol="AAPL", benchmark_bars=bench)

        assert result.setup_name == "Momentum Breakout"
        assert result.symbol == "AAPL"
        assert result.statistics.total_trades >= 0
        if result.trades:
            assert result.statistics.win_rate >= 0.0
            assert result.statistics.average_holding_days >= 0.0
