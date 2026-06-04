"""Domain models for trade plans, backtests, statistics, and alerts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal

TradeDirection = Literal["LONG", "SHORT"]


class TradeOutcome(str, Enum):
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    EXPIRED = "EXPIRED"
    NOT_FILLED = "NOT_FILLED"


class AlertType(str, Enum):
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    STOP_HIT = "STOP_HIT"
    TARGET_HIT = "TARGET_HIT"


@dataclass(frozen=True, slots=True)
class TradePlan:
    symbol: str
    setup_name: str
    direction: TradeDirection
    entry_price: float
    stop_price: float
    target_price: float
    risk_reward: float
    confidence_score: float
    generated_at: datetime
    """When True, backtest waits for price to trade through entry (e.g. buy-stop above high)."""
    entry_is_stop: bool = False
    """Backtest-derived stats for this symbol + setup (optional, for UI)."""
    historical_statistics: SetupStatistics | None = None

    def with_historical_statistics(
        self, statistics: SetupStatistics | None
    ) -> TradePlan:
        return replace(self, historical_statistics=statistics)

    @property
    def historical_win_rate(self) -> float | None:
        if self.historical_statistics is None or self.historical_statistics.total_trades == 0:
            return None
        return self.historical_statistics.win_rate

    @property
    def historical_profit_factor(self) -> float | None:
        if self.historical_statistics is None or self.historical_statistics.total_trades == 0:
            return None
        return self.historical_statistics.profit_factor

    @property
    def historical_average_holding_days(self) -> float | None:
        if self.historical_statistics is None or self.historical_statistics.total_trades == 0:
            return None
        return self.historical_statistics.average_holding_days

    @property
    def historical_total_trades(self) -> int | None:
        if self.historical_statistics is None:
            return None
        return self.historical_statistics.total_trades

    @staticmethod
    def calculate_risk_reward(
        *,
        direction: TradeDirection,
        entry_price: float,
        stop_price: float,
        target_price: float,
    ) -> float:
        if entry_price <= 0:
            return 0.0
        if direction == "LONG":
            risk = entry_price - stop_price
            reward = target_price - entry_price
        else:
            risk = stop_price - entry_price
            reward = entry_price - target_price
        if risk <= 0:
            return 0.0
        return reward / risk


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    plan: TradePlan
    signal_date: date
    entry_date: date
    exit_date: date
    exit_price: float
    outcome: TradeOutcome
    return_pct: float
    holding_days: int


@dataclass(frozen=True, slots=True)
class SetupStatistics:
    setup_name: str
    symbol: str
    total_trades: int
    win_rate: float
    expectancy: float
    average_return: float
    average_win: float
    average_loss: float
    average_holding_days: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float

    @classmethod
    def empty(cls, setup_name: str, *, symbol: str = "") -> SetupStatistics:
        return cls(
            setup_name=setup_name,
            symbol=symbol.upper(),
            total_trades=0,
            win_rate=0.0,
            expectancy=0.0,
            average_return=0.0,
            average_win=0.0,
            average_loss=0.0,
            average_holding_days=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
        )

    @property
    def historical_win_rate_pct(self) -> float:
        """Win rate on 0–100 scale for UI."""
        return round(self.win_rate * 100.0, 1)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    setup_name: str
    symbol: str
    trades: tuple[SimulatedTrade, ...]
    statistics: SetupStatistics
    record_id: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class StockRank:
    symbol: str
    score: float
    trend_strength: float
    relative_strength: float
    volume_expansion: float
    setup_quality: float
    best_setup: str | None


@dataclass(frozen=True, slots=True)
class Alert:
    symbol: str
    setup_name: str
    alert_type: AlertType
    message: str
    triggered_at: datetime
    reference_price: float
    plan: TradePlan | None = None


@dataclass(frozen=True, slots=True)
class EnrichedTradePlan:
    """Trade plan with optional historical statistics from backtest."""

    plan: TradePlan
    statistics: SetupStatistics | None = None
    expected_hold_days: float | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
