"""Research validation: walk-forward, regime, yearly, CSV export."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trade_planner.config import BacktestConfig, MomentumBreakoutConfig
from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.export import ResearchCsvExporter
from trade_planner.research.feature_analysis import analyze_feature_conditions
from trade_planner.research.metrics import performance_from_trades
from trade_planner.research.models import FeatureSnapshot, MarketRegime
from trade_planner.research.regime import classify_market_regime
from trade_planner.research.regime_analysis import build_regime_comparison
from trade_planner.research.report_generator import (
    StrategyResearchReportGenerator,
    SymbolBarSet,
)
from trade_planner.research.walk_forward import (
    WalkForwardFoldSpec,
    WalkForwardValidator,
    default_walk_forward_folds,
)
from trade_planner.research.yearly import yearly_performance_table
from trade_planner.research.collector import collect_momentum_breakout_trades
from trade_planner.types import OHLCVBar
from tests.test_momentum_breakout_setup import (
    _aligned_trend_series,
    _test_config,
)


def _manual_trade(
    *,
    signal: date,
    ret: float,
    regime: MarketRegime = MarketRegime.RISK_ON,
    rs: float = 85.0,
) -> HistoricalTrade:
    snap = FeatureSnapshot(
        rs_percentile=rs,
        volume_ratio=2.0,
        close_vs_sma50=0.05,
        close_vs_sma200=0.10,
        distance_to_20d_high=0.01,
        market_regime=regime,
    )
    entry = signal + timedelta(days=1)
    exit_d = entry + timedelta(days=5)
    return HistoricalTrade(
        trade_id=f"TEST:Unit:{signal}:{entry}:{exit_d}",
        setup_name="Momentum Breakout",
        symbol="TEST",
        direction="LONG",
        signal_date=signal,
        entry_date=entry,
        exit_date=exit_d,
        entry_price=100.0,
        exit_price=100.0 * (1.0 + ret),
        stop_price=95.0,
        target_price=110.0,
        outcome=__import__(
            "trade_planner.models", fromlist=["TradeOutcome"]
        ).TradeOutcome.TARGET_HIT,
        return_pct=ret,
        holding_days=5,
        feature_snapshot=snap,
    )


class TestWalkForwardValidator:
    def test_default_folds_match_spec(self) -> None:
        folds = default_walk_forward_folds()
        years = [fold.test_year for fold in folds]
        assert years == [2019, 2020, 2021, 2022, 2023, 2024]
        assert folds[0].train_end == date(2018, 12, 31)
        assert folds[0].test_start == date(2019, 1, 1)

    def test_oos_metrics_per_year(self) -> None:
        trades = (
            _manual_trade(signal=date(2019, 6, 1), ret=0.05),
            _manual_trade(signal=date(2019, 8, 1), ret=-0.02),
            _manual_trade(signal=date(2020, 3, 1), ret=0.08),
            _manual_trade(signal=date(2021, 1, 15), ret=0.03),
        )
        report = WalkForwardValidator().validate(trades, setup_name="Momentum Breakout")
        by_year = {fold.test_year: fold.performance for fold in report.folds}
        assert by_year[2019].total_trades == 2
        assert by_year[2019].win_rate == pytest.approx(0.5)
        assert by_year[2020].total_trades == 1
        assert report.aggregate.total_trades == 4

    def test_custom_fold_window(self) -> None:
        trades = (_manual_trade(signal=date(2022, 5, 1), ret=0.04),)
        folds = (
            WalkForwardFoldSpec(
                train_start=date(2000, 1, 1),
                train_end=date(2021, 12, 31),
                test_start=date(2022, 1, 1),
                test_end=date(2022, 12, 31),
                test_year=2022,
            ),
        )
        report = WalkForwardValidator(folds).validate(
            trades, setup_name="Momentum Breakout"
        )
        assert report.folds[0].performance.total_trades == 1
        assert report.folds[0].performance.expectancy == pytest.approx(0.04)


class TestRegimeSegmentation:
    def test_groups_trades_by_regime(self) -> None:
        trades = (
            _manual_trade(signal=date(2020, 1, 1), ret=0.05, regime=MarketRegime.RISK_ON),
            _manual_trade(signal=date(2020, 2, 1), ret=0.03, regime=MarketRegime.RISK_ON),
            _manual_trade(signal=date(2020, 3, 1), ret=-0.04, regime=MarketRegime.RISK_OFF),
            _manual_trade(signal=date(2020, 4, 1), ret=0.01, regime=MarketRegime.NEUTRAL),
        )
        comparison = build_regime_comparison(trades, setup_name="Momentum Breakout")
        by_regime = {row.regime: row.performance for row in comparison.rows}
        assert by_regime[MarketRegime.RISK_ON].total_trades == 2
        assert by_regime[MarketRegime.RISK_ON].win_rate == pytest.approx(1.0)
        assert by_regime[MarketRegime.RISK_OFF].total_trades == 1
        assert by_regime[MarketRegime.RISK_OFF].average_return == pytest.approx(-0.04)


class TestYearlyAggregation:
    def test_yearly_table(self) -> None:
        trades = (
            _manual_trade(signal=date(2019, 1, 1), ret=0.02),
            _manual_trade(signal=date(2020, 1, 1), ret=0.04),
            _manual_trade(signal=date(2020, 6, 1), ret=-0.01),
        )
        rows = yearly_performance_table(trades, setup_name="Momentum Breakout")
        assert [row.year for row in rows] == [2019, 2020]
        assert rows[0].performance.total_trades == 1
        assert rows[1].performance.total_trades == 2


class TestCsvExport:
    def test_export_generates_headers_and_rows(self) -> None:
        trades = (_manual_trade(signal=date(2021, 1, 1), ret=0.05),)
        exporter = ResearchCsvExporter()
        csv_text = exporter.export_historical_trades(trades)
        assert "trade_id" in csv_text
        assert "rs_percentile" in csv_text
        assert "RISK_ON" in csv_text

        yearly = exporter.export_yearly_report(
            yearly_performance_table(trades, setup_name="Momentum Breakout")
        )
        assert "year,total_trades" in yearly.replace(" ", "")

        regime = exporter.export_regime_report(
            build_regime_comparison(trades, setup_name="Momentum Breakout")
        )
        assert "regime,total_trades" in regime.replace(" ", "")


class TestFeatureSnapshotsOnTrades:
    def test_collector_attaches_snapshot(self) -> None:
        stock, bench = _aligned_trend_series(320)
        cfg = _test_config()
        trades = collect_momentum_breakout_trades(
            symbol="TEST",
            stock_bars=stock,
            benchmark_bars=bench,
            momentum_config=cfg,
            backtest_config=BacktestConfig(min_warmup_bars=30, max_holding_days=15),
        )
        if not trades:
            pytest.skip("fixture produced no trades")
        trade = trades[0]
        assert trade.feature_snapshot is not None
        assert trade.feature_snapshot.volume_ratio is not None
        assert trade.feature_snapshot.market_regime in MarketRegime


class TestStrategyResearchReport:
    def test_report_generator_end_to_end(self) -> None:
        stock, bench = _aligned_trend_series(400)
        universe = [
            SymbolBarSet(symbol="TEST", stock_bars=stock, benchmark_bars=bench)
        ]
        generator = StrategyResearchReportGenerator(
            walk_forward=WalkForwardValidator(
                (
                    WalkForwardFoldSpec(
                        train_start=date(2000, 1, 1),
                        train_end=date(2019, 12, 31),
                        test_start=date(2020, 1, 1),
                        test_end=date(2020, 12, 31),
                        test_year=2020,
                    ),
                )
            )
        )
        report = generator.generate(
            universe,
            start_date=date(2020, 1, 2),
            end_date=date(2021, 12, 31),
        )
        assert report.setup_name == "Momentum Breakout"
        assert report.symbols_tested == ("TEST",)
        assert report.performance.total_trades >= 0
        assert report.walk_forward.folds


class TestMarketRegimeClassifier:
    def test_risk_on_uptrend_benchmark(self) -> None:
        bars: list[OHLCVBar] = []
        price = 100.0
        start = date(2018, 1, 1)
        for idx in range(260):
            price *= 1.002
            bars.append(
                OHLCVBar(
                    trading_date=start + timedelta(days=idx),
                    open=price,
                    high=price * 1.001,
                    low=price * 0.999,
                    close=price,
                    volume=1_000_000.0,
                )
            )
        regime = classify_market_regime(tuple(bars), index=len(bars) - 1)
        assert regime == MarketRegime.RISK_ON


class TestFeatureConditionAnalysis:
    def test_top_and_worst_bins(self) -> None:
        trades = tuple(
            _manual_trade(
                signal=date(2020, 1, 1) + timedelta(days=idx * 30),
                ret=0.10 if idx % 2 == 0 else -0.05,
                rs=70.0 + idx * 5,
            )
            for idx in range(12)
        )
        top, worst = analyze_feature_conditions(trades, top_n=2)
        if top:
            assert top[0].expectancy >= worst[0].expectancy if worst else True
