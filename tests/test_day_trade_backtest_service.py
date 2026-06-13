from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.dependencies.adapter_dependencies import get_yfinance_adapter
from app.main import app
from app.services.day_trade_backtest_service import (
    DayTradeBacktestService,
    DayTradeBacktestDataError,
    DayTradeEntryFilterConfig,
    IntradayBacktestCandle,
    RULE_ALIGNMENT_NOTES,
    CLOSE_CONFIRMED_ENTRY_FILTERS,
    build_day_trade_comparison_table,
    build_multi_symbol_backtest_report,
    intraday_provider_availability,
    simulate_day_trade_backtest,
    summarize_day_trade_backtest,
    top_day_trade_losers,
    top_day_trade_winners,
)
from app.services.trade_replay_service import (
    ReplayBar,
    TradePlanRecord,
    generate_replay_events,
    plan_signature,
)

ET = ZoneInfo("America/New_York")


class _FakeYFinanceAdapter:
    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars
        self.calls: list[dict[str, object]] = []

    def get_history(
        self,
        symbol: str,
        *,
        period: str | None = None,
        interval: str,
        auto_adjust: bool = True,
        prepost: bool = False,
        start: date | datetime | None = None,
        end: date | datetime | None = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "auto_adjust": auto_adjust,
                "prepost": prepost,
                "start": start,
                "end": end,
            }
        )
        return self.bars


def _dt(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ET)


def _candle(
    day: date,
    hour: int,
    minute: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> IntradayBacktestCandle:
    return IntradayBacktestCandle(
        timestamp=_dt(day.year, day.month, day.day, hour, minute),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100,
    )


def _frame(candles: list[IntradayBacktestCandle]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [candle.open for candle in candles],
            "High": [candle.high for candle in candles],
            "Low": [candle.low for candle in candles],
            "Close": [candle.close for candle in candles],
            "Volume": [candle.volume for candle in candles],
        },
        index=pd.DatetimeIndex([candle.timestamp for candle in candles]),
    )


def _opening_range(day: date) -> list[IntradayBacktestCandle]:
    return [
        _candle(day, 9, 30, 100.0, 100.5, 99.5, 100.0),
        _candle(day, 9, 35, 100.0, 101.0, 99.0, 100.5),
    ]


def test_day_trade_backtest_long_target_win() -> None:
    day = date(2026, 6, 5)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.8, 101.2),
            _candle(day, 10, 5, 101.2, 103.5, 101.1, 103.1),
        ],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.setup_direction == "long"
    assert row.entry_time == _dt(2026, 6, 5, 10, 0)
    assert row.entry_price == 101.01
    assert row.target_1 == 103.01
    assert row.outcome == "win"
    assert row.exit_reason == "target_1_hit"
    assert row.target_reason == "target_1_hit_one_opening_range_width"
    assert row.stop_reason is None
    assert row.or_width == 2.0
    assert row.stop_distance == 0.93
    assert row.target_distance == 2.0
    assert row.r_achieved == 1.0
    assert row.r_multiple == 1.0
    assert row.dollar_pl == 100.0
    assert row.mfe >= 1.0
    assert row.mae >= 0.0
    assert row.hold_minutes == 5.0
    assert row.max_favorable_excursion >= 1.0


def test_day_trade_backtest_uses_stop_first_when_stop_and_target_hit_same_candle() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=250,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 103.5, 100.0, 102.0),
        ],
    )

    row = rows[0]
    assert row.setup_direction == "long"
    assert row.outcome == "loss"
    assert row.exit_reason == "stop_hit"
    assert row.stop_reason == "long_vwap_stop_hit"
    assert row.target_reason is None
    assert row.r_multiple == -1.0
    assert row.dollar_pl == -250.0


def test_day_trade_backtest_entry_candle_close_back_inside_does_not_invalidate() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.8, 100.95),
            _candle(day, 10, 5, 101.2, 103.5, 101.1, 103.1),
        ],
    )

    row = rows[0]
    assert row.entry_candle_closed_inside_or is True
    assert row.entry_time == _dt(2026, 6, 8, 10, 0)
    assert row.exit_reason == "target_1_hit"
    assert row.outcome == "win"


def test_day_trade_backtest_one_close_back_inside_does_not_invalidate() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.8, 101.2),
            _candle(day, 10, 5, 101.2, 101.4, 100.8, 100.95),
        ],
    )

    row = rows[0]
    assert row.entry_candle_closed_inside_or is False
    assert row.exit_reason == "close_exit"
    assert row.outcome == "loss"
    assert row.exit_time == _dt(2026, 6, 8, 10, 5)


def test_day_trade_backtest_two_consecutive_closes_back_inside_invalidates() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.8, 101.2),
            _candle(day, 10, 5, 101.2, 101.4, 100.8, 100.95),
            _candle(day, 10, 10, 100.95, 101.2, 100.7, 100.9),
        ],
    )

    row = rows[0]
    assert row.exit_reason == "invalidated"
    assert row.outcome == "invalidated"
    assert row.exit_time == _dt(2026, 6, 8, 10, 10)


def test_day_trade_backtest_short_two_consecutive_closes_back_inside_invalidates() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 99.0, 99.2, 98.8, 98.9),
            _candle(day, 10, 5, 98.9, 99.2, 98.8, 99.05),
            _candle(day, 10, 10, 99.05, 99.3, 98.9, 99.1),
        ],
    )

    row = rows[0]
    assert row.setup_direction == "short"
    assert row.exit_reason == "invalidated"
    assert row.outcome == "invalidated"
    assert row.exit_time == _dt(2026, 6, 8, 10, 10)


def test_day_trade_backtest_custom_invalidation_confirmation_closes() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        invalidation_confirmation_closes=3,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.8, 101.2),
            _candle(day, 10, 5, 101.2, 101.4, 100.8, 100.95),
            _candle(day, 10, 10, 100.95, 101.2, 100.7, 100.9),
        ],
    )

    row = rows[0]
    assert row.exit_reason == "close_exit"
    assert row.outcome == "loss"


def test_day_trade_backtest_summary_counts_same_candle_close_back_inside() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.8, 100.95),
            _candle(day, 10, 5, 101.2, 103.5, 101.1, 103.1),
        ],
    )

    summary = summarize_day_trade_backtest(rows)

    assert summary.same_candle_invalidation_count == 1
    assert summary.invalidation_pct == 0.0


def test_day_trade_backtest_target_still_exits_immediately_before_invalidation() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 103.5, 100.8, 100.95),
            _candle(day, 10, 5, 101.2, 101.4, 100.8, 100.95),
        ],
    )

    row = rows[0]
    assert row.entry_candle_closed_inside_or is True
    assert row.exit_reason == "target_1_hit"
    assert row.outcome == "win"


def test_day_trade_backtest_long_close_confirmed_breakout_filter() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        entry_filters=DayTradeEntryFilterConfig(
            require_close_confirmed_breakout=True,
        ),
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.8, 100.95),
            _candle(day, 10, 5, 100.95, 101.2, 100.8, 100.9),
        ],
    )

    assert rows[0].outcome == "no_trade"


def test_day_trade_backtest_short_close_confirmed_breakout_filter() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        entry_filters=DayTradeEntryFilterConfig(
            require_close_confirmed_breakout=True,
        ),
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 99.0, 99.2, 98.8, 99.05),
            _candle(day, 10, 5, 99.05, 99.2, 98.9, 99.1),
        ],
    )

    assert rows[0].outcome == "no_trade"


def test_day_trade_backtest_long_vwap_alignment_filter() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        entry_filters=DayTradeEntryFilterConfig(require_vwap_alignment=True),
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.8, 100.0),
            _candle(day, 10, 5, 100.0, 100.9, 100.5, 100.9),
        ],
    )

    assert rows[0].outcome == "no_trade"


def test_day_trade_backtest_short_vwap_alignment_filter() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        entry_filters=DayTradeEntryFilterConfig(require_vwap_alignment=True),
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 99.0, 99.2, 98.8, 100.2),
            _candle(day, 10, 5, 99.05, 99.2, 99.05, 99.1),
        ],
    )

    assert rows[0].outcome == "no_trade"


def test_day_trade_backtest_or_width_filter_skips_too_narrow_range() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        entry_filters=DayTradeEntryFilterConfig(min_or_width_pct=0.01),
        candles=[
            _candle(day, 9, 30, 100.0, 100.2, 99.8, 100.0),
            _candle(day, 9, 35, 100.0, 100.4, 99.6, 100.0),
            _candle(day, 10, 0, 100.4, 101.0, 100.3, 100.8),
        ],
    )

    assert rows[0].or_width == 0.8
    assert rows[0].outcome == "no_trade"


def test_day_trade_backtest_or_width_filter_skips_too_wide_range() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        entry_filters=DayTradeEntryFilterConfig(max_or_width_pct=0.035),
        candles=[
            _candle(day, 9, 30, 100.0, 104.0, 96.0, 100.0),
            _candle(day, 9, 35, 100.0, 103.0, 97.0, 100.0),
            _candle(day, 10, 0, 104.0, 105.0, 103.5, 104.5),
        ],
    )

    assert rows[0].or_width == 8.0
    assert rows[0].outcome == "no_trade"


def test_day_trade_backtest_no_trade_after_noon_filter() -> None:
    day = date(2026, 6, 8)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        entry_filters=DayTradeEntryFilterConfig(no_trade_after_noon=True),
        candles=[
            *_opening_range(day),
            _candle(day, 12, 0, 101.0, 101.5, 100.8, 101.2),
            _candle(day, 12, 5, 101.2, 103.5, 101.1, 103.1),
        ],
    )

    assert rows[0].outcome == "no_trade"


def test_day_trade_backtest_comparison_table_contains_filter_scenarios() -> None:
    day = date(2026, 6, 8)
    comparison = build_day_trade_comparison_table(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.8, 101.2),
            _candle(day, 10, 5, 101.2, 103.5, 101.1, 103.1),
        ],
    )

    assert [row.scenario for row in comparison] == [
        "baseline",
        "close-confirmed breakout",
        "VWAP-aligned",
        "OR-width-filtered",
        "all filters combined",
    ]
    assert all(row.total_trades == 1 for row in comparison)
    assert comparison[0].target_1_hit_pct == 1.0


def test_day_trade_backtest_multi_symbol_aggregation() -> None:
    nvda_day = date(2026, 6, 8)
    aapl_day = date(2026, 6, 9)
    candidate_rows = {
        "NVDA": simulate_day_trade_backtest(
            symbol="NVDA",
            risk_per_trade=100,
            entry_filters=CLOSE_CONFIRMED_ENTRY_FILTERS,
            candles=[
                *_opening_range(nvda_day),
                _candle(nvda_day, 10, 0, 101.0, 101.5, 100.8, 101.2),
                _candle(nvda_day, 10, 5, 101.2, 103.5, 101.1, 103.1),
            ],
        ),
        "AAPL": simulate_day_trade_backtest(
            symbol="AAPL",
            risk_per_trade=100,
            entry_filters=CLOSE_CONFIRMED_ENTRY_FILTERS,
            candles=[
                *_opening_range(aapl_day),
                _candle(aapl_day, 10, 0, 101.0, 101.5, 100.8, 101.2),
                _candle(aapl_day, 10, 5, 101.2, 101.4, 100.0, 100.5),
            ],
        ),
    }
    baseline_rows = {
        symbol: rows
        for symbol, rows in candidate_rows.items()
    }

    report = build_multi_symbol_backtest_report(
        candidate_rows_by_symbol=candidate_rows,
        baseline_rows_by_symbol=baseline_rows,
    )

    assert report.candidate_scenario == "close-confirmed breakout"
    assert report.entry_filters.require_close_confirmed_breakout is True
    assert [row.symbol for row in report.aggregate_comparison] == ["NVDA", "AAPL"]
    assert report.aggregate_comparison[0].total_trades == 1
    assert report.aggregate_comparison[0].total_r == 1.0
    assert report.aggregate_comparison[0].target_1_hit_pct == 1.0
    assert report.aggregate_comparison[1].total_trades == 1
    assert report.aggregate_comparison[1].total_r == -1.0
    assert report.aggregate_comparison[1].stop_hit_pct == 1.0
    assert report.portfolio_summary.total_trades == 2
    assert report.portfolio_summary.total_r == 0.0
    assert report.portfolio_summary.average_r == 0.0
    assert report.portfolio_summary.profit_factor == 1.0
    assert report.portfolio_summary.max_drawdown == -1.0
    assert report.portfolio_summary.best_symbol == "NVDA"
    assert report.portfolio_summary.worst_symbol == "AAPL"
    assert len(report.baseline_comparison) == 2


def test_day_trade_backtest_no_trade_when_opening_range_never_breaks() -> None:
    day = date(2026, 6, 9)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 100.2, 100.8, 99.2, 100.0),
            _candle(day, 15, 55, 100.0, 100.6, 99.4, 100.2),
        ],
    )

    row = rows[0]
    assert row.outcome == "no_trade"
    assert row.entry_time is None
    assert row.exit_reason == "no_trade"
    assert row.or_width == 2.0
    assert row.stop_distance is None
    assert row.target_distance is None
    assert row.r_multiple == 0.0


def test_day_trade_backtest_final_exit_near_close_if_still_open() -> None:
    day = date(2026, 6, 10)
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=[
            *_opening_range(day),
            _candle(day, 10, 0, 101.0, 101.5, 100.9, 101.2),
            _candle(day, 15, 55, 101.2, 102.2, 101.1, 102.0),
        ],
    )

    row = rows[0]
    assert row.exit_time == _dt(2026, 6, 10, 15, 55)
    assert row.exit_price == 102.0
    assert row.exit_reason == "close_exit"
    assert row.hold_minutes == 355.0
    assert row.outcome == "win"
    assert row.exit_price < row.target_1
    assert row.r_multiple > 0


def test_day_trade_backtest_summary_metrics() -> None:
    rows = []
    for day, candles in [
        (
            date(2026, 6, 5),
            [
                *_opening_range(date(2026, 6, 5)),
                _candle(date(2026, 6, 5), 10, 0, 101.0, 101.5, 100.8, 101.2),
                _candle(date(2026, 6, 5), 10, 5, 101.2, 103.5, 101.1, 103.1),
            ],
        ),
        (
            date(2026, 6, 8),
            [
                *_opening_range(date(2026, 6, 8)),
                _candle(date(2026, 6, 8), 10, 0, 101.0, 103.5, 100.0, 102.0),
            ],
        ),
        (
            date(2026, 6, 9),
            [
                *_opening_range(date(2026, 6, 9)),
                _candle(date(2026, 6, 9), 10, 0, 100.2, 100.8, 99.2, 100.0),
            ],
        ),
    ]:
        rows.extend(
            simulate_day_trade_backtest(
                symbol="NVDA",
                risk_per_trade=100,
                candles=candles,
            )
        )

    summary = summarize_day_trade_backtest(rows)

    assert summary.total_trading_days_tested == 3
    assert summary.total_trades == 2
    assert summary.win_rate == 0.5
    assert summary.total_r == 0.0
    assert summary.profit_factor == 1.0
    assert summary.max_drawdown == -1.0
    assert summary.best_day == 1.0
    assert summary.worst_day == -1.0
    assert summary.stop_hit_pct == 0.5
    assert summary.target_1_hit_pct == 0.5
    assert summary.target_2_hit_pct == 0.0
    assert summary.close_exit_pct == 0.0
    assert summary.invalidation_pct == 0.0
    assert summary.same_candle_invalidation_count == 0
    assert summary.average_stop_distance == 0.93
    assert summary.average_or_width == 2.0
    assert summary.average_hold_minutes == 2.5


def test_day_trade_backtest_top_winners_and_losers() -> None:
    rows = [
        *simulate_day_trade_backtest(
            symbol="NVDA",
            risk_per_trade=100,
            candles=[
                *_opening_range(date(2026, 6, 1)),
                _candle(date(2026, 6, 1), 10, 0, 101.0, 101.5, 100.8, 101.2),
                _candle(date(2026, 6, 1), 10, 5, 101.2, 106.0, 101.1, 105.0),
            ],
        ),
        *simulate_day_trade_backtest(
            symbol="NVDA",
            risk_per_trade=100,
            candles=[
                *_opening_range(date(2026, 6, 2)),
                _candle(date(2026, 6, 2), 10, 0, 101.0, 101.5, 100.8, 101.2),
                _candle(date(2026, 6, 2), 15, 55, 101.2, 102.2, 101.1, 102.0),
            ],
        ),
        *simulate_day_trade_backtest(
            symbol="NVDA",
            risk_per_trade=100,
            candles=[
                *_opening_range(date(2026, 6, 3)),
                _candle(date(2026, 6, 3), 10, 0, 101.0, 101.5, 100.8, 101.2),
                _candle(date(2026, 6, 3), 10, 5, 101.2, 101.3, 100.4, 100.5),
                _candle(date(2026, 6, 3), 10, 10, 100.5, 101.3, 100.4, 100.5),
            ],
        ),
        *simulate_day_trade_backtest(
            symbol="NVDA",
            risk_per_trade=100,
            candles=[
                *_opening_range(date(2026, 6, 4)),
                _candle(date(2026, 6, 4), 10, 0, 101.0, 101.5, 100.0, 101.2),
            ],
        ),
    ]

    top_winners = top_day_trade_winners(rows)
    top_losers = top_day_trade_losers(rows)

    assert len(top_winners) == 2
    assert top_winners[0].exit_reason == "target_2_hit"
    assert top_winners[0].r_multiple == 2.0
    assert top_winners[1].exit_reason == "close_exit"
    assert len(top_losers) == 2
    assert top_losers[0].exit_reason == "stop_hit"
    assert top_losers[0].r_multiple == -1.0
    assert top_losers[-1].exit_reason == "invalidated"


def test_day_trade_backtest_top_lists_are_limited_to_10() -> None:
    rows = []
    for offset in range(12):
        day = date(2026, 6, 1) + timedelta(days=offset)
        rows.extend(
            simulate_day_trade_backtest(
                symbol="NVDA",
                risk_per_trade=100,
                candles=[
                    *_opening_range(day),
                    _candle(day, 10, 0, 101.0, 101.5, 100.8, 101.2),
                    _candle(day, 10, 5, 101.2, 103.5, 101.1, 103.1),
                ],
            )
        )
    for offset in range(12):
        day = date(2026, 7, 1) + timedelta(days=offset)
        rows.extend(
            simulate_day_trade_backtest(
                symbol="NVDA",
                risk_per_trade=100,
                candles=[
                    *_opening_range(day),
                    _candle(day, 10, 0, 101.0, 101.5, 100.0, 101.2),
                ],
            )
        )

    assert len(top_day_trade_winners(rows)) == 10
    assert len(top_day_trade_losers(rows)) == 10


def test_day_trade_backtest_service_loads_historical_intraday_candles() -> None:
    day = date(2026, 6, 5)
    adapter = _FakeYFinanceAdapter(
        _frame(
            [
                *_opening_range(day),
                _candle(day, 10, 0, 101.0, 101.5, 100.8, 101.2),
                _candle(day, 10, 5, 101.2, 103.5, 101.1, 103.1),
            ]
        )
    )
    service = DayTradeBacktestService(adapter)  # type: ignore[arg-type]

    result = service.run_backtest(
        symbol="nvda",
        start=day,
        end=day,
        risk_per_trade=100,
    )

    assert adapter.calls[0]["symbol"] == "NVDA"
    assert adapter.calls[0]["interval"] == "5m"
    assert adapter.calls[0]["prepost"] is True
    assert adapter.calls[0]["start"] == day
    assert adapter.calls[0]["end"] == date(2026, 6, 6)
    assert result.symbol == "NVDA"
    assert result.available_start_date == intraday_provider_availability().available_start_date
    assert result.available_end_date == intraday_provider_availability().available_end_date
    assert "Yahoo Finance 5-minute" in result.provider_limit_reason
    assert result.invalidation_confirmation_closes == 2
    assert result.entry_filters.require_close_confirmed_breakout is False
    assert [row.scenario for row in result.comparison_table] == [
        "baseline",
        "close-confirmed breakout",
        "VWAP-aligned",
        "OR-width-filtered",
        "all filters combined",
    ]
    assert result.rows[0].outcome == "win"


def test_day_trade_backtest_rejects_request_before_yahoo_intraday_window() -> None:
    availability = intraday_provider_availability()
    start = availability.available_start_date - timedelta(days=1)
    adapter = _FakeYFinanceAdapter(pd.DataFrame())
    service = DayTradeBacktestService(adapter)  # type: ignore[arg-type]

    try:
        service.run_backtest(
            symbol="nvda",
            start=start,
            end=availability.available_end_date,
            risk_per_trade=100,
        )
    except DayTradeBacktestDataError as exc:
        detail = exc.to_detail()
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected provider history limit error")

    assert adapter.calls == []
    assert detail["code"] == "provider_history_limit_exceeded"
    assert detail["available_start_date"] == availability.available_start_date.isoformat()
    assert detail["available_end_date"] == availability.available_end_date.isoformat()
    assert "60 calendar days" in detail["provider_limit_reason"]


def test_day_trade_backtest_empty_provider_response_is_not_zero_trade_result() -> None:
    availability = intraday_provider_availability()
    adapter = _FakeYFinanceAdapter(pd.DataFrame())
    service = DayTradeBacktestService(adapter)  # type: ignore[arg-type]

    try:
        service.run_backtest(
            symbol="nvda",
            start=availability.available_end_date,
            end=availability.available_end_date,
            risk_per_trade=100,
        )
    except DayTradeBacktestDataError as exc:
        detail = exc.to_detail()
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected no market data error")

    assert len(adapter.calls) == 1
    assert detail["code"] == "no_market_data_available"
    assert detail["available_start_date"] == availability.available_start_date.isoformat()
    assert detail["available_end_date"] == availability.available_end_date.isoformat()


def test_day_trade_backtest_route_returns_frontend_contract() -> None:
    day = date(2026, 6, 5)
    adapter = _FakeYFinanceAdapter(
        _frame(
            [
                *_opening_range(day),
                _candle(day, 10, 0, 101.0, 101.5, 100.8, 101.2),
                _candle(day, 10, 5, 101.2, 103.5, 101.1, 103.1),
            ]
        )
    )

    class _FakeUser:
        identity_sub = "user-1"

    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    app.dependency_overrides[get_yfinance_adapter] = lambda: adapter
    client = TestClient(app)
    try:
        response = client.get(
            "/api/v1/research/day-trade/backtest",
            params={
                "symbol": "nvda",
                "start": "2026-06-05",
                "end": "2026-06-05",
                "risk_per_trade": "100",
                "invalidation_confirmation_closes": "2",
                "require_close_confirmed_breakout": "true",
                "require_vwap_alignment": "true",
                "min_or_width_pct": "0.01",
                "max_or_width_pct": "0.035",
                "no_trade_after_noon": "true",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "NVDA"
        assert body["risk_per_trade"] == 100
        assert body["invalidation_confirmation_closes"] == 2
        assert body["entry_filters"] == {
            "require_close_confirmed_breakout": True,
            "require_vwap_alignment": True,
            "min_or_width_pct": 0.01,
            "max_or_width_pct": 0.035,
            "no_trade_after_noon": True,
        }
        assert body["available_start_date"]
        assert body["available_end_date"]
        assert "Yahoo Finance 5-minute" in body["provider_limit_reason"]
        assert body["rows"][0]["entry_price"] == 101.01
        assert body["rows"][0]["or_width"] == 2.0
        assert body["rows"][0]["stop_distance"] == 0.93
        assert body["rows"][0]["target_distance"] == 2.0
        assert body["rows"][0]["r_achieved"] == 1.0
        assert body["rows"][0]["mfe"] >= 1.0
        assert body["rows"][0]["mae"] >= 0.0
        assert body["rows"][0]["exit_reason"] == "target_1_hit"
        assert body["rows"][0]["target_reason"] == "target_1_hit_one_opening_range_width"
        assert body["rows"][0]["outcome"] == "win"
        assert body["summary"]["total_trades"] == 1
        assert body["summary"]["target_1_hit_pct"] == 1.0
        assert body["summary"]["invalidation_pct"] == 0.0
        assert body["summary"]["same_candle_invalidation_count"] == 0
        assert body["summary"]["average_or_width"] == 2.0
        assert body["top_winners"][0]["date"] == "2026-06-05"
        assert body["top_losers"] == []
        assert [row["scenario"] for row in body["comparison_table"]] == [
            "baseline",
            "close-confirmed breakout",
            "VWAP-aligned",
            "OR-width-filtered",
            "all filters combined",
        ]
        assert set(body["comparison_table"][0]) == {
            "scenario",
            "total_trades",
            "win_rate",
            "average_r",
            "total_r",
            "profit_factor",
            "max_drawdown",
            "target_1_hit_pct",
            "invalidation_pct",
        }
    finally:
        app.dependency_overrides.clear()


def test_day_trade_backtest_route_returns_structured_provider_limit_error() -> None:
    availability = intraday_provider_availability()
    adapter = _FakeYFinanceAdapter(pd.DataFrame())

    class _FakeUser:
        identity_sub = "user-1"

    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    app.dependency_overrides[get_yfinance_adapter] = lambda: adapter
    client = TestClient(app)
    try:
        response = client.get(
            "/api/v1/research/day-trade/backtest",
            params={
                "symbol": "nvda",
                "start": (availability.available_start_date - timedelta(days=1)).isoformat(),
                "end": availability.available_end_date.isoformat(),
                "risk_per_trade": "100",
            },
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "provider_history_limit_exceeded"
        assert detail["available_start_date"] == availability.available_start_date.isoformat()
        assert detail["available_end_date"] == availability.available_end_date.isoformat()
        assert "60 calendar days" in detail["provider_limit_reason"]
    finally:
        app.dependency_overrides.clear()


def test_day_trade_backtest_rules_are_validated_against_live_replay_levels() -> None:
    day = date(2026, 6, 5)
    candles = [
        *_opening_range(day),
        _candle(day, 10, 0, 101.0, 101.5, 100.8, 101.2),
        _candle(day, 10, 5, 101.2, 103.5, 101.1, 103.1),
    ]
    rows = simulate_day_trade_backtest(
        symbol="NVDA",
        risk_per_trade=100,
        candles=candles,
    )
    row = rows[0]
    levels = {
        "long_entry": row.long_trigger,
        "long_stop": row.vwap_at_entry,
        "long_target_1": row.target_1,
        "long_target_2": row.target_2,
        "open_range_high": row.opening_range_high,
        "open_range_low": row.opening_range_low,
        "vwap": row.vwap_at_entry,
    }
    payload = {"workflow": "day_trade", "levels": levels}
    plan = TradePlanRecord(
        plan_id="plan-validation",
        symbol="NVDA",
        workflow="day_trade",
        plan_date=day,
        generated_at=_dt(2026, 6, 5, 10, 0),
        source="delayed",
        source_freshness_label="Educational / delayed — not for live execution.",
        signature=plan_signature(levels, payload),
        levels=levels,
        payload=payload,
    )
    replay_bars = [
        ReplayBar(
            timestamp=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )
        for candle in candles
        if candle.timestamp >= _dt(2026, 6, 5, 10, 0)
    ]

    events = generate_replay_events(plan, replay_bars)

    assert row.entry_price == events[0].level_price
    assert row.entry_time == events[0].event_time
    assert row.stop_price == row.vwap_at_entry
    assert row.target_1 == events[1].level_price
    assert [event.event_type for event in events] == [
        "long_trigger_activated",
        "target_1_hit",
        "setup_extended",
    ]
    assert events[2].level_price == row.target_2


def test_day_trade_backtest_documents_live_rule_alignment() -> None:
    expected = {
        "opening_range_window",
        "trigger_rules",
        "vwap_calculation",
        "stop_logic",
        "target_logic",
        "invalidation_logic",
        "close_exit_rule",
        "timezone",
    }

    assert set(RULE_ALIGNMENT_NOTES) == expected
    assert "backtest-only" in RULE_ALIGNMENT_NOTES["close_exit_rule"]
