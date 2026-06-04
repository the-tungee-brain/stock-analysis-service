"""Forward candle simulation — stop-entry fill, then target vs stop resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trade_planner.config import BacktestConfig
from trade_planner.models import SimulatedTrade, TradeOutcome, TradePlan
from trade_planner.types import OHLCVBar


@dataclass(frozen=True, slots=True)
class _SimulationResult:
    outcome: TradeOutcome
    entry_date: date
    exit_date: date
    exit_price: float
    holding_days: int
    return_pct: float


def _apply_slippage(price: float, *, bps: float, adverse: bool, direction: str) -> float:
    slip = price * (bps / 10_000.0)
    if direction == "LONG":
        return price + slip if adverse else price - slip
    return price - slip if adverse else price + slip


def _return_pct(direction: str, entry: float, exit_price: float) -> float:
    if entry <= 0:
        return 0.0
    if direction == "LONG":
        return (exit_price - entry) / entry
    return (entry - exit_price) / entry


def simulate_trade_forward(
    *,
    plan: TradePlan,
    signal_index: int,
    bars: tuple[OHLCVBar, ...],
    config: BacktestConfig,
) -> _SimulationResult | None:
    if signal_index >= len(bars) - 1:
        return None

    trigger = plan.entry_price
    stop = plan.stop_price
    target = plan.target_price
    max_days = config.max_holding_days

    entry_price: float | None = None
    entry_date: date | None = None
    entry_day_offset = 0

    for offset in range(1, max_days + 1):
        idx = signal_index + offset
        if idx >= len(bars):
            break
        bar = bars[idx]

        if entry_price is None:
            if plan.entry_is_stop:
                if plan.direction == "LONG" and bar.high < trigger:
                    continue
                if plan.direction == "SHORT" and bar.low > trigger:
                    continue
                entry_price = _apply_slippage(
                    trigger,
                    bps=config.slippage_bps,
                    adverse=True,
                    direction=plan.direction,
                )
            else:
                entry_price = _apply_slippage(
                    trigger,
                    bps=config.slippage_bps,
                    adverse=True,
                    direction=plan.direction,
                )
            entry_date = bar.trading_date
            entry_day_offset = offset

        assert entry_price is not None and entry_date is not None

        if plan.direction == "LONG":
            stop_hit = bar.low <= stop
            target_hit = bar.high >= target
        else:
            stop_hit = bar.high >= stop
            target_hit = bar.low <= target

        if stop_hit and target_hit:
            outcome = TradeOutcome.STOP_HIT
            exit_price = _apply_slippage(
                stop, bps=config.slippage_bps, adverse=True, direction=plan.direction
            )
        elif stop_hit:
            outcome = TradeOutcome.STOP_HIT
            exit_price = _apply_slippage(
                stop, bps=config.slippage_bps, adverse=True, direction=plan.direction
            )
        elif target_hit:
            outcome = TradeOutcome.TARGET_HIT
            exit_price = _apply_slippage(
                target, bps=config.slippage_bps, adverse=False, direction=plan.direction
            )
        else:
            continue

        return _SimulationResult(
            outcome=outcome,
            entry_date=entry_date,
            exit_date=bar.trading_date,
            exit_price=exit_price,
            holding_days=offset - entry_day_offset + 1,
            return_pct=_return_pct(plan.direction, entry_price, exit_price),
        )

    if entry_price is None:
        return _SimulationResult(
            outcome=TradeOutcome.NOT_FILLED,
            entry_date=bars[signal_index].trading_date,
            exit_date=bars[min(signal_index + max_days, len(bars) - 1)].trading_date,
            exit_price=trigger,
            holding_days=0,
            return_pct=0.0,
        )

    last_idx = min(signal_index + max_days, len(bars) - 1)
    last_bar = bars[last_idx]
    return _SimulationResult(
        outcome=TradeOutcome.EXPIRED,
        entry_date=entry_date,
        exit_date=last_bar.trading_date,
        exit_price=last_bar.close,
        holding_days=last_idx - signal_index,
        return_pct=_return_pct(plan.direction, entry_price, last_bar.close),
    )


def build_simulated_trade(
    *,
    plan: TradePlan,
    signal_index: int,
    bars: tuple[OHLCVBar, ...],
    config: BacktestConfig,
) -> SimulatedTrade | None:
    result = simulate_trade_forward(
        plan=plan,
        signal_index=signal_index,
        bars=bars,
        config=config,
    )
    if result is None:
        return None
    if result.outcome == TradeOutcome.NOT_FILLED:
        return None

    signal_bar = bars[signal_index]
    return SimulatedTrade(
        plan=plan,
        signal_date=signal_bar.trading_date,
        entry_date=result.entry_date,
        exit_date=result.exit_date,
        exit_price=round(result.exit_price, 4),
        outcome=result.outcome,
        return_pct=round(result.return_pct, 6),
        holding_days=result.holding_days,
    )
