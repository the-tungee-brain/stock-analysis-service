"""In-memory persistence for setup statistics (multi-setup per symbol)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from trade_planner.models import BacktestResult, SetupStatistics, SimulatedTrade
from trade_planner.persistence.historical_trade import HistoricalTrade


@dataclass(frozen=True, slots=True)
class SetupStatisticsRecord:
    """Statistics plus every historical trade used to compute them."""

    setup_name: str
    symbol: str
    statistics: SetupStatistics
    trades: tuple[HistoricalTrade, ...]
    recorded_at: datetime

    @property
    def total_trades(self) -> int:
        return self.statistics.total_trades


class SetupStatisticsStore(Protocol):
    def save_backtest(
        self,
        result: BacktestResult,
        *,
        trades: tuple[SimulatedTrade, ...] | None = None,
    ) -> SetupStatisticsRecord: ...

    def get(self, setup_name: str, symbol: str) -> SetupStatisticsRecord | None: ...

    def list_for_symbol(self, symbol: str) -> tuple[SetupStatisticsRecord, ...]: ...

    def list_for_setup(self, setup_name: str) -> tuple[SetupStatisticsRecord, ...]: ...

    def list_trades(
        self, setup_name: str, symbol: str
    ) -> tuple[HistoricalTrade, ...]: ...


class InMemorySetupStatisticsStore:
    """Thread-unsafe in-memory store keyed by (setup_name, symbol)."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], SetupStatisticsRecord] = {}

    def save_backtest(
        self,
        result: BacktestResult,
        *,
        trades: tuple[SimulatedTrade, ...] | None = None,
    ) -> SetupStatisticsRecord:
        source = trades if trades is not None else result.trades
        historical = tuple(HistoricalTrade.from_simulated(trade) for trade in source)
        symbol = result.symbol.upper()
        key = (result.setup_name, symbol)
        record = SetupStatisticsRecord(
            setup_name=result.setup_name,
            symbol=symbol,
            statistics=result.statistics,
            trades=historical,
            recorded_at=datetime.now(timezone.utc),
        )
        self._records[key] = record
        return record

    def get(self, setup_name: str, symbol: str) -> SetupStatisticsRecord | None:
        return self._records.get((setup_name, symbol.upper()))

    def list_for_symbol(self, symbol: str) -> tuple[SetupStatisticsRecord, ...]:
        sym = symbol.upper()
        return tuple(
            record
            for (_, s), record in sorted(self._records.items())
            if s == sym
        )

    def list_for_setup(self, setup_name: str) -> tuple[SetupStatisticsRecord, ...]:
        return tuple(
            record
            for (name, _), record in sorted(self._records.items())
            if name == setup_name
        )

    def list_trades(self, setup_name: str, symbol: str) -> tuple[HistoricalTrade, ...]:
        record = self.get(setup_name, symbol)
        if record is None:
            return ()
        return record.trades

    def clear(self) -> None:
        self._records.clear()
