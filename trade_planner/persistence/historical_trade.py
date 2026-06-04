"""Persisted historical trade records from backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from trade_planner.models import (
    SimulatedTrade,
    TradeDirection,
    TradeOutcome,
    TradePlan,
    utc_now,
)

if TYPE_CHECKING:
    from trade_planner.research.models import FeatureSnapshot


@dataclass(frozen=True, slots=True)
class HistoricalTrade:
    """Immutable record of one completed backtest trade."""

    trade_id: str
    setup_name: str
    symbol: str
    direction: TradeDirection
    signal_date: date
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    outcome: TradeOutcome
    return_pct: float
    holding_days: int
    feature_snapshot: FeatureSnapshot | None = None

    @staticmethod
    def make_trade_id(
        *,
        setup_name: str,
        symbol: str,
        signal_date: date,
        entry_date: date,
        exit_date: date,
    ) -> str:
        return (
            f"{symbol.upper()}:{setup_name}:{signal_date.isoformat()}:"
            f"{entry_date.isoformat()}:{exit_date.isoformat()}"
        )

    @classmethod
    def from_simulated(
        cls,
        trade: SimulatedTrade,
        *,
        feature_snapshot: FeatureSnapshot | None = None,
    ) -> HistoricalTrade:
        plan = trade.plan
        return cls(
            trade_id=cls.make_trade_id(
                setup_name=plan.setup_name,
                symbol=plan.symbol,
                signal_date=trade.signal_date,
                entry_date=trade.entry_date,
                exit_date=trade.exit_date,
            ),
            setup_name=plan.setup_name,
            symbol=plan.symbol.upper(),
            direction=plan.direction,
            signal_date=trade.signal_date,
            entry_date=trade.entry_date,
            exit_date=trade.exit_date,
            entry_price=round(plan.entry_price, 4),
            exit_price=round(trade.exit_price, 4),
            stop_price=round(plan.stop_price, 4),
            target_price=round(plan.target_price, 4),
            outcome=trade.outcome,
            return_pct=round(trade.return_pct, 6),
            holding_days=trade.holding_days,
            feature_snapshot=feature_snapshot,
        )

    def to_simulated(self) -> SimulatedTrade:
        plan = TradePlan(
            symbol=self.symbol,
            setup_name=self.setup_name,
            direction=self.direction,
            entry_price=self.entry_price,
            stop_price=self.stop_price,
            target_price=self.target_price,
            risk_reward=TradePlan.calculate_risk_reward(
                direction=self.direction,
                entry_price=self.entry_price,
                stop_price=self.stop_price,
                target_price=self.target_price,
            ),
            confidence_score=0.0,
            generated_at=utc_now(),
        )
        return SimulatedTrade(
            plan=plan,
            signal_date=self.signal_date,
            entry_date=self.entry_date,
            exit_date=self.exit_date,
            exit_price=self.exit_price,
            outcome=self.outcome,
            return_pct=self.return_pct,
            holding_days=self.holding_days,
        )
