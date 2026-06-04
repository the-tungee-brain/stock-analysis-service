"""Walk-forward validation with expanding train and single-year test windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.metrics import performance_from_trades
from trade_planner.research.models import (
    PerformanceMetrics,
    WalkForwardFoldResult,
    WalkForwardReport,
)

DEFAULT_WALK_FORWARD_TEST_YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024)
DEFAULT_TRAIN_START = date(2000, 1, 1)


@dataclass(frozen=True, slots=True)
class WalkForwardFoldSpec:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    test_year: int


def default_walk_forward_folds(
    *,
    train_start: date = DEFAULT_TRAIN_START,
    test_years: Sequence[int] = DEFAULT_WALK_FORWARD_TEST_YEARS,
) -> tuple[WalkForwardFoldSpec, ...]:
    folds: list[WalkForwardFoldSpec] = []
    for year in test_years:
        train_end = date(year - 1, 12, 31)
        folds.append(
            WalkForwardFoldSpec(
                train_start=train_start,
                train_end=train_end,
                test_start=date(year, 1, 1),
                test_end=date(year, 12, 31),
                test_year=year,
            )
        )
    return tuple(folds)


def _trades_in_test_window(
    trades: Sequence[HistoricalTrade],
    *,
    test_start: date,
    test_end: date,
) -> tuple[HistoricalTrade, ...]:
    return tuple(
        trade
        for trade in trades
        if test_start <= trade.signal_date <= test_end
    )


class WalkForwardValidator:
    """Evaluate out-of-sample performance on rolling one-year test windows."""

    def __init__(
        self,
        folds: Sequence[WalkForwardFoldSpec] | None = None,
    ) -> None:
        self._folds = tuple(folds) if folds is not None else default_walk_forward_folds()

    @property
    def folds(self) -> tuple[WalkForwardFoldSpec, ...]:
        return self._folds

    def validate(
        self,
        trades: Sequence[HistoricalTrade],
        *,
        setup_name: str,
    ) -> WalkForwardReport:
        fold_results: list[WalkForwardFoldResult] = []
        oos_trades: list[HistoricalTrade] = []

        for spec in self._folds:
            window_trades = _trades_in_test_window(
                trades,
                test_start=spec.test_start,
                test_end=spec.test_end,
            )
            oos_trades.extend(window_trades)
            fold_results.append(
                WalkForwardFoldResult(
                    train_start=spec.train_start,
                    train_end=spec.train_end,
                    test_start=spec.test_start,
                    test_end=spec.test_end,
                    test_year=spec.test_year,
                    performance=performance_from_trades(
                        window_trades, setup_name=setup_name
                    ),
                )
            )

        aggregate = performance_from_trades(oos_trades, setup_name=setup_name)
        return WalkForwardReport(
            folds=tuple(fold_results),
            aggregate=aggregate,
        )
