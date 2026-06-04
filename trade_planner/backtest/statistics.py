"""Aggregate historical trades into setup statistics."""

from __future__ import annotations

import math
from typing import Sequence

from trade_planner.indicators import max_drawdown_pct
from trade_planner.models import SetupStatistics, SimulatedTrade, TradeOutcome
from trade_planner.persistence.historical_trade import HistoricalTrade

TradeLike = SimulatedTrade | HistoricalTrade


def _return_pct(trade: TradeLike) -> float:
    return trade.return_pct


def _holding_days(trade: TradeLike) -> int:
    return trade.holding_days


def sharpe_ratio(
    returns: Sequence[float],
    *,
    periods_per_year: float = 252.0,
    average_holding_days: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sharpe from per-trade returns.

    Scales by sqrt(trades_per_year) where trades_per_year = periods_per_year / avg_hold.
    """
    if len(returns) < 2 or average_holding_days <= 0:
        return 0.0

    excess = [r - risk_free_rate for r in returns]
    mean = sum(excess) / len(excess)
    variance = sum((x - mean) ** 2 for x in excess) / (len(excess) - 1)
    if variance <= 0:
        return 0.0

    trades_per_year = periods_per_year / average_holding_days
    return (mean / math.sqrt(variance)) * math.sqrt(trades_per_year)


def aggregate_setup_statistics(
    setup_name: str,
    trades: Sequence[TradeLike],
    *,
    symbol: str = "",
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> SetupStatistics:
    if not trades:
        return SetupStatistics.empty(setup_name, symbol=symbol)

    returns = [_return_pct(trade) for trade in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = len(wins) / len(returns)
    average_return = sum(returns) / len(returns)
    average_win = sum(wins) / len(wins) if wins else 0.0
    average_loss = sum(losses) / len(losses) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = win_rate * average_win + (1.0 - win_rate) * average_loss
    avg_hold = sum(_holding_days(trade) for trade in trades) / len(trades)

    equity = [1.0]
    for ret in returns:
        equity.append(equity[-1] * (1.0 + ret))

    sharpe = sharpe_ratio(
        returns,
        periods_per_year=periods_per_year,
        average_holding_days=avg_hold,
        risk_free_rate=risk_free_rate,
    )

    return SetupStatistics(
        setup_name=setup_name,
        symbol=symbol.upper(),
        total_trades=len(trades),
        win_rate=round(win_rate, 4),
        expectancy=round(expectancy, 6),
        average_return=round(average_return, 6),
        average_win=round(average_win, 6),
        average_loss=round(average_loss, 6),
        average_holding_days=round(avg_hold, 2),
        profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
        max_drawdown=round(max_drawdown_pct(equity), 4),
        sharpe_ratio=round(sharpe, 4),
    )


def win_rate_from_trades(trades: Sequence[TradeLike]) -> float:
    if not trades:
        return 0.0
    wins = sum(
        1
        for trade in trades
        if trade.outcome == TradeOutcome.TARGET_HIT or trade.return_pct > 0
    )
    return wins / len(trades)
