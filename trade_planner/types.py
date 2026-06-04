"""Market data containers for deterministic setup evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence


@dataclass(frozen=True, slots=True)
class OHLCVBar:
    trading_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class StockData:
    """Point-in-time view of a symbol's OHLCV history."""

    symbol: str
    bars: tuple[OHLCVBar, ...]
    index: int
    benchmark_bars: tuple[OHLCVBar, ...] | None = None

    def __post_init__(self) -> None:
        if not self.bars:
            raise ValueError("bars must not be empty")
        if self.index < 0 or self.index >= len(self.bars):
            raise ValueError(f"index {self.index} out of range for {len(self.bars)} bars")
        if self.benchmark_bars is not None:
            if len(self.benchmark_bars) != len(self.bars):
                raise ValueError("benchmark_bars must align 1:1 with bars")
            if self.benchmark_bars[self.index].trading_date != self.bars[self.index].trading_date:
                raise ValueError("benchmark bar date must match stock bar date at index")

    @property
    def current(self) -> OHLCVBar:
        return self.bars[self.index]

    @property
    def prior(self) -> OHLCVBar | None:
        if self.index == 0:
            return None
        return self.bars[self.index - 1]

    def slice_end(self, lookback: int) -> Sequence[OHLCVBar]:
        """Inclusive window ending at ``index`` with at most ``lookback`` bars."""
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        start = max(0, self.index - lookback + 1)
        return self.bars[start : self.index + 1]

    def at(self, index: int) -> StockData:
        return StockData(symbol=self.symbol, bars=self.bars, index=index)

    @classmethod
    def from_bars(
        cls,
        symbol: str,
        bars: Sequence[OHLCVBar],
        *,
        index: int | None = None,
        benchmark_bars: Sequence[OHLCVBar] | None = None,
    ) -> StockData:
        frozen = tuple(bars)
        resolved_index = len(frozen) - 1 if index is None else index
        bench = tuple(benchmark_bars) if benchmark_bars is not None else None
        return cls(
            symbol=symbol,
            bars=frozen,
            index=resolved_index,
            benchmark_bars=bench,
        )
