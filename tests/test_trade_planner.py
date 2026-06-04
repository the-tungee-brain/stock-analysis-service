"""Unit tests for rules-based trade planner engine."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trade_planner.alerts.engine import AlertEngine
from trade_planner.backtest.engine import BacktestEngine
from trade_planner.backtest.statistics import aggregate_setup_statistics
from trade_planner.config import BacktestConfig, TrendContinuationConfig
from trade_planner.models import AlertType, TradeOutcome, TradePlan
from trade_planner.ranking.engine import StockRankingEngine
from trade_planner.persistence import InMemorySetupStatisticsStore
from trade_planner.service import TradePlannerService
from trade_planner.setups import (
    MomentumBreakoutSetup,
    PullbackSetup,
    TrendContinuationSetup,
)
from trade_planner.types import OHLCVBar, StockData

# Momentum-specific tests live in test_momentum_breakout_setup.py


def _bar(
    day_offset: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1_000_000.0,
    start: date | None = None,
) -> OHLCVBar:
    base = start or date(2024, 1, 2)
    return OHLCVBar(
        trading_date=base + timedelta(days=day_offset),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _uptrend_bars(
    days: int,
    *,
    start_price: float = 100.0,
    daily_return: float = 0.003,
    day_offset: int = 0,
) -> tuple[OHLCVBar, ...]:
    bars: list[OHLCVBar] = []
    price = start_price
    for idx in range(days):
        nxt = price * (1.0 + daily_return)
        bars.append(
            _bar(
                day_offset + idx,
                open_=price,
                high=max(price, nxt) * 1.002,
                low=min(price, nxt) * 0.998,
                close=nxt,
                volume=1_200_000.0 + idx * 50_000.0,
            )
        )
        price = nxt
    return tuple(bars)


class TestTradePlan:
    def test_calculate_risk_reward_long(self) -> None:
        rr = TradePlan.calculate_risk_reward(
            direction="LONG",
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
        )
        assert rr == pytest.approx(2.0)


class TestBacktestEngine:
    def test_simulates_trades_with_trend_continuation(self) -> None:
        bars = _uptrend_bars(120, daily_return=0.008)
        setup = TrendContinuationSetup(
            TrendContinuationConfig(volume_expansion_ratio=1.0, min_ma_spread_pct=0.0001)
        )
        engine = BacktestEngine(BacktestConfig(min_warmup_bars=30, max_holding_days=15))
        result = engine.run(setup, bars, symbol="UP")
        assert result.statistics.total_trades > 0

    def test_aggregate_statistics_empty(self) -> None:
        stats = aggregate_setup_statistics("Test", ())
        assert stats.total_trades == 0


class TestStockRankingEngine:
    def test_ranks_higher_trend_first(self) -> None:
        strong = _uptrend_bars(80, daily_return=0.01)
        weak = _uptrend_bars(80, start_price=50.0, daily_return=0.001)
        engine = StockRankingEngine()
        setups = [TrendContinuationSetup(), MomentumBreakoutSetup()]
        ranks = engine.rank_symbols({"STRONG": strong, "WEAK": weak}, setups)
        assert ranks[0].symbol == "STRONG"


class TestAlertEngine:
    def test_entry_triggered_alert(self) -> None:
        plan = TradePlan(
            symbol="NVDA",
            setup_name="Momentum Breakout",
            direction="LONG",
            entry_price=180.25,
            stop_price=175.0,
            target_price=190.75,
            risk_reward=2.1,
            confidence_score=72.0,
            generated_at=__import__("trade_planner.models", fromlist=["utc_now"]).utc_now(),
        )
        bars = (
            _bar(0, open_=179.0, high=179.5, low=178.5, close=179.0),
            _bar(1, open_=180.0, high=181.0, low=179.5, close=180.5),
        )
        alerts = AlertEngine().evaluate(
            plan=plan,
            stock_data=StockData.from_bars("NVDA", bars),
            prior_price=179.0,
        )
        assert AlertType.ENTRY_TRIGGERED in {a.alert_type for a in alerts}


class TestTradePlannerService:
    def test_pullback_setup_warmup_requirement(self) -> None:
        assert PullbackSetup().required_warmup_bars() >= 51

    def test_generate_best_plan_attaches_historical_statistics(self) -> None:
        bars = _uptrend_bars(120, daily_return=0.008)
        service = TradePlannerService(
            statistics_store=InMemorySetupStatisticsStore(),
        )
        enriched = service.generate_best_plan(symbol="UP", bars=bars)
        if enriched is None:
            pytest.skip("no setup triggered on fixture")
        plan = enriched.plan
        assert plan.historical_statistics is not None
        assert plan.historical_total_trades == enriched.statistics.total_trades
        summary = service.format_plan_summary(enriched)
        assert "Historical Win Rate:" in summary
        assert "Profit Factor:" in summary
        assert "Total Historical Trades:" in summary
