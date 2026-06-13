from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.intraday_trading_bias_engine import (
    EASTERN,
    MARKET_OPEN,
    OPENING_RANGE_END,
)
from app.models.day_trade_backtest_models import (
    DayTradeBacktestResponse,
    DayTradeBacktestRow,
    DayTradeBacktestSummary,
    DayTradeSetupDirection,
)

REGULAR_CLOSE = time(16, 0)
TRIGGER_BUFFER = 0.01
RULE_ALIGNMENT_NOTES = {
    "opening_range_window": "Aligned with live intraday bias: 9:30 inclusive to 10:00 exclusive ET.",
    "trigger_rules": "Aligned with live replay levels: OR high/low plus/minus the 0.01 trigger buffer; backtest limits execution to the first tradable side for one daily P/L row.",
    "vwap_calculation": "Aligned to live plan intent without lookahead: VWAP is calculated from regular-session candles available before 10:00 ET and then held fixed as the plan stop/control level.",
    "stop_logic": "Aligned with live day-trade plan: VWAP control level is the stop for both long and short setups.",
    "target_logic": "Level math is aligned with live replay: target 1 is one opening-range width from trigger and target 2 is two widths from trigger. Backtest exits at target 1 for a single conservative P/L row; live replay may continue the educational story toward target 2.",
    "invalidation_logic": "Aligned with live replay: long invalidates on close back below OR high; short invalidates on close back above OR low.",
    "close_exit_rule": "Intentional backtest-only addition: live replay records story milestones, while backtest marks any open trade to market on the final regular-session candle to produce daily P/L.",
    "timezone": "Aligned: all sessions and trading dates are interpreted in America/New_York.",
}


@dataclass(frozen=True)
class IntradayBacktestCandle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class _DayLevels:
    opening_range_high: float
    opening_range_low: float
    long_trigger: float
    short_trigger: float
    width: float


class DayTradeBacktestService:
    def __init__(self, yfinance_adapter: YFinanceAdapter) -> None:
        self._yfinance = yfinance_adapter

    def run_backtest(
        self,
        *,
        symbol: str,
        start: date,
        end: date,
        risk_per_trade: float,
    ) -> DayTradeBacktestResponse:
        if end < start:
            raise ValueError("end must be on or after start")
        if risk_per_trade <= 0:
            raise ValueError("risk_per_trade must be positive")

        symbol_upper = symbol.strip().upper()
        hist = self._yfinance.get_history(
            symbol_upper,
            interval="5m",
            prepost=True,
            start=start,
            end=end + timedelta(days=1),
        )
        candles = [
            candle
            for candle in normalize_intraday_candles(hist)
            if start <= market_date(candle.timestamp) <= end
            and _is_regular_session(candle.timestamp)
        ]
        rows = simulate_day_trade_backtest(
            symbol=symbol_upper,
            candles=candles,
            risk_per_trade=risk_per_trade,
        )
        return DayTradeBacktestResponse(
            symbol=symbol_upper,
            start=start,
            end=end,
            risk_per_trade=risk_per_trade,
            rows=rows,
            summary=summarize_day_trade_backtest(rows),
        )


def simulate_day_trade_backtest(
    *,
    symbol: str,
    candles: list[IntradayBacktestCandle],
    risk_per_trade: float,
) -> list[DayTradeBacktestRow]:
    rows: list[DayTradeBacktestRow] = []
    for trading_day, day_candles in _group_by_market_date(candles).items():
        rows.append(
            _simulate_day(
                symbol=symbol,
                trading_day=trading_day,
                candles=day_candles,
                risk_per_trade=risk_per_trade,
            )
        )
    return sorted(rows, key=lambda row: row.date)


def summarize_day_trade_backtest(
    rows: list[DayTradeBacktestRow],
) -> DayTradeBacktestSummary:
    trades = [row for row in rows if row.outcome != "no_trade"]
    wins = [row for row in trades if row.r_multiple > 0]
    losses = [row for row in trades if row.r_multiple < 0]
    total_r = round(sum(row.r_multiple for row in trades), 4)
    gross_win = sum(row.r_multiple for row in wins)
    gross_loss = abs(sum(row.r_multiple for row in losses))
    profit_factor = None if gross_loss == 0 else round(gross_win / gross_loss, 4)
    daily_r = [row.r_multiple for row in rows]

    return DayTradeBacktestSummary(
        total_trading_days_tested=len(rows),
        total_trades=len(trades),
        win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
        average_r=round(total_r / len(trades), 4) if trades else 0.0,
        total_r=total_r,
        profit_factor=profit_factor,
        max_drawdown=round(_max_drawdown([row.r_multiple for row in rows]), 4),
        average_win=round(gross_win / len(wins), 4) if wins else 0.0,
        average_loss=round(sum(row.r_multiple for row in losses) / len(losses), 4)
        if losses
        else 0.0,
        best_day=round(max(daily_r), 4) if daily_r else 0.0,
        worst_day=round(min(daily_r), 4) if daily_r else 0.0,
    )


def normalize_intraday_candles(hist: pd.DataFrame | None) -> list[IntradayBacktestCandle]:
    if hist is None or hist.empty or not isinstance(hist.index, pd.DatetimeIndex):
        return []
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(hist.columns):
        return []

    candles: list[IntradayBacktestCandle] = []
    for timestamp, row in hist.iterrows():
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        open_price = _float(row.get("Open"))
        high = _float(row.get("High"))
        low = _float(row.get("Low"))
        close = _float(row.get("Close"))
        if open_price is None or high is None or low is None or close is None:
            continue
        candles.append(
            IntradayBacktestCandle(
                timestamp=ts.to_pydatetime(),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=max(0, int(_float(row.get("Volume")) or 0)),
            )
        )
    return sorted(candles, key=lambda candle: candle.timestamp)


def market_date(value: datetime) -> date:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(EASTERN).date()


def _simulate_day(
    *,
    symbol: str,
    trading_day: date,
    candles: list[IntradayBacktestCandle],
    risk_per_trade: float,
) -> DayTradeBacktestRow:
    candles = sorted(candles, key=lambda candle: candle.timestamp)
    levels = _opening_range_levels(candles)
    if levels is None:
        return _no_trade_row(symbol, trading_day)

    vwap_by_timestamp = _vwap_by_timestamp(candles)
    plan_vwap = _plan_vwap(candles)
    entry: IntradayBacktestCandle | None = None
    direction: DayTradeSetupDirection = "none"
    entry_price: float | None = None

    for candle in candles:
        local_time = _local_time(candle.timestamp)
        if local_time < OPENING_RANGE_END:
            continue
        long_hit = candle.high >= levels.long_trigger
        short_hit = candle.low <= levels.short_trigger
        if long_hit and short_hit:
            direction = _direction_from_open(candle, levels)
        elif long_hit:
            direction = "long"
        elif short_hit:
            direction = "short"

        if direction == "long":
            entry = candle
            entry_price = levels.long_trigger
            break
        if direction == "short":
            entry = candle
            entry_price = levels.short_trigger
            break

    if entry is None or entry_price is None or direction == "none":
        return _no_trade_row(symbol, trading_day, levels)

    vwap_at_entry = vwap_by_timestamp.get(entry.timestamp)
    stop_price = _stop_price(direction, entry_price, levels, plan_vwap)
    target_1, target_2 = _targets(direction, entry_price, levels.width)
    exit_candle = entry
    exit_price = entry.close
    outcome = "breakeven"
    r_multiple = 0.0
    mfe = 0.0
    mae = 0.0

    for candle in candles[candles.index(entry) :]:
        favorable = (
            (candle.high - entry_price) / (entry_price - stop_price)
            if direction == "long"
            else (entry_price - candle.low) / (stop_price - entry_price)
        )
        adverse = (
            (entry_price - candle.low) / (entry_price - stop_price)
            if direction == "long"
            else (candle.high - entry_price) / (stop_price - entry_price)
        )
        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        stop_hit = candle.low <= stop_price if direction == "long" else candle.high >= stop_price
        target_2_hit = candle.high >= target_2 if direction == "long" else candle.low <= target_2
        target_1_hit = candle.high >= target_1 if direction == "long" else candle.low <= target_1
        invalidated = (
            candle.close < levels.opening_range_high
            if direction == "long"
            else candle.close > levels.opening_range_low
        )

        exit_candle = candle
        if stop_hit:
            exit_price = stop_price
            outcome = "loss"
            r_multiple = -1.0
            break
        if target_2_hit:
            exit_price = target_2
            outcome = "win"
            r_multiple = 2.0
            break
        if target_1_hit:
            exit_price = target_1
            outcome = "win"
            r_multiple = 1.0
            break
        if invalidated:
            exit_price = candle.close
            outcome = "invalidated"
            r_multiple = _r_multiple(direction, entry_price, stop_price, exit_price)
            break

    else:
        exit_candle = candles[-1]
        exit_price = exit_candle.close
        r_multiple = _r_multiple(direction, entry_price, stop_price, exit_price)
        outcome = _outcome_from_r(r_multiple)

    return DayTradeBacktestRow(
        date=trading_day,
        symbol=symbol,
        setup_direction=direction,
        opening_range_high=_round_price(levels.opening_range_high),
        opening_range_low=_round_price(levels.opening_range_low),
        long_trigger=_round_price(levels.long_trigger),
        short_trigger=_round_price(levels.short_trigger),
        vwap_at_entry=_round_price(plan_vwap),
        entry_time=entry.timestamp,
        entry_price=_round_price(entry_price),
        stop_price=_round_price(stop_price),
        target_1=_round_price(target_1),
        target_2=_round_price(target_2),
        exit_time=exit_candle.timestamp,
        exit_price=_round_price(exit_price),
        outcome=outcome,
        r_multiple=round(r_multiple, 4),
        dollar_pl=round(r_multiple * risk_per_trade, 2),
        max_favorable_excursion=round(max(0.0, mfe), 4),
        max_adverse_excursion=round(max(0.0, mae), 4),
    )


def _opening_range_levels(
    candles: list[IntradayBacktestCandle],
) -> _DayLevels | None:
    opening = [
        candle
        for candle in candles
        if MARKET_OPEN <= _local_time(candle.timestamp) < OPENING_RANGE_END
    ]
    if not opening:
        return None
    high = max(candle.high for candle in opening)
    low = min(candle.low for candle in opening)
    width = high - low
    if width <= 0:
        return None
    return _DayLevels(
        opening_range_high=high,
        opening_range_low=low,
        long_trigger=high + TRIGGER_BUFFER,
        short_trigger=low - TRIGGER_BUFFER,
        width=width,
    )


def _no_trade_row(
    symbol: str,
    trading_day: date,
    levels: _DayLevels | None = None,
) -> DayTradeBacktestRow:
    return DayTradeBacktestRow(
        date=trading_day,
        symbol=symbol,
        setup_direction="none",
        opening_range_high=_round_price(levels.opening_range_high) if levels else None,
        opening_range_low=_round_price(levels.opening_range_low) if levels else None,
        long_trigger=_round_price(levels.long_trigger) if levels else None,
        short_trigger=_round_price(levels.short_trigger) if levels else None,
        outcome="no_trade",
        r_multiple=0.0,
        dollar_pl=0.0,
        max_favorable_excursion=0.0,
        max_adverse_excursion=0.0,
    )


def _vwap_by_timestamp(
    candles: list[IntradayBacktestCandle],
) -> dict[datetime, float]:
    cumulative_price_volume = 0.0
    cumulative_volume = 0
    result: dict[datetime, float] = {}
    for candle in candles:
        volume = max(candle.volume, 1)
        typical = (candle.high + candle.low + candle.close) / 3
        cumulative_price_volume += typical * volume
        cumulative_volume += volume
        result[candle.timestamp] = cumulative_price_volume / cumulative_volume
    return result


def _plan_vwap(candles: list[IntradayBacktestCandle]) -> float | None:
    plan_candles = [
        candle
        for candle in candles
        if MARKET_OPEN <= _local_time(candle.timestamp) < OPENING_RANGE_END
    ]
    if not plan_candles:
        return None
    return _vwap_by_timestamp(plan_candles).get(plan_candles[-1].timestamp)


def _stop_price(
    direction: DayTradeSetupDirection,
    entry_price: float,
    levels: _DayLevels,
    vwap_at_entry: float | None,
) -> float:
    del entry_price
    if vwap_at_entry is not None:
        return vwap_at_entry
    if direction == "long":
        return levels.opening_range_low
    return levels.opening_range_high


def _targets(
    direction: DayTradeSetupDirection,
    entry_price: float,
    width: float,
) -> tuple[float, float]:
    if direction == "long":
        return entry_price + width, entry_price + 2 * width
    return entry_price - width, entry_price - 2 * width


def _direction_from_open(
    candle: IntradayBacktestCandle,
    levels: _DayLevels,
) -> DayTradeSetupDirection:
    long_distance = abs(levels.long_trigger - candle.open)
    short_distance = abs(candle.open - levels.short_trigger)
    return "long" if long_distance <= short_distance else "short"


def _r_multiple(
    direction: DayTradeSetupDirection,
    entry_price: float,
    stop_price: float,
    exit_price: float,
) -> float:
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return 0.0
    if direction == "long":
        return (exit_price - entry_price) / risk
    return (entry_price - exit_price) / risk


def _outcome_from_r(value: float) -> str:
    if value > 0.05:
        return "win"
    if value < -0.05:
        return "loss"
    return "breakeven"


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    return drawdown


def _group_by_market_date(
    candles: list[IntradayBacktestCandle],
) -> dict[date, list[IntradayBacktestCandle]]:
    grouped: dict[date, list[IntradayBacktestCandle]] = {}
    for candle in sorted(candles, key=lambda item: item.timestamp):
        grouped.setdefault(market_date(candle.timestamp), []).append(candle)
    return grouped


def _is_regular_session(value: datetime) -> bool:
    local = _local_time(value)
    return MARKET_OPEN <= local < REGULAR_CLOSE


def _local_time(value: datetime) -> time:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(EASTERN).time()


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_price(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None
