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
    DayTradeBacktestComparisonRow,
    DayTradeBacktestDirectionComparisonRow,
    DayTradeBacktestFailedSymbol,
    DayTradeBacktestEntryFilters,
    DayTradeBacktestMultiSymbolReport,
    DayTradeBacktestPortfolioSummary,
    DayTradeBacktestResponse,
    DayTradeBacktestRow,
    DayTradeBacktestSummary,
    DayTradeBacktestSymbolAggregateRow,
    DayTradeDirectionMode,
    DayTradeSetupDirection,
)

REGULAR_CLOSE = time(16, 0)
NOON_ET = time(12, 0)
TRIGGER_BUFFER = 0.01
DEFAULT_MIN_OR_WIDTH_PCT = 0.01
DEFAULT_MAX_OR_WIDTH_PCT = 0.035
PRIMARY_CANDIDATE_SCENARIO = "close-confirmed breakout"
DEFAULT_DIRECTION_MODE: DayTradeDirectionMode = "long_only"
SHORT_SAMPLE_WARNING_LABEL = (
    "This is a short sample. Validate across multiple symbols and longer history "
    "before trusting."
)
YAHOO_INTRADAY_5M_HISTORY_DAYS = 60
YAHOO_INTRADAY_5M_LIMIT_REASON = (
    "Yahoo Finance 5-minute intraday candles are only available for "
    "approximately 60 calendar days."
)
RULE_ALIGNMENT_NOTES = {
    "opening_range_window": "Aligned with live intraday bias: 9:30 inclusive to 10:00 exclusive ET.",
    "trigger_rules": "Aligned with live replay levels: OR high/low plus/minus the 0.01 trigger buffer; backtest limits execution to the first tradable side for one daily P/L row.",
    "vwap_calculation": "Aligned to live plan intent without lookahead: VWAP is calculated from regular-session candles available before 10:00 ET and then held fixed as the plan stop/control level.",
    "stop_logic": "Aligned with live day-trade plan: VWAP control level is the stop for both long and short setups.",
    "target_logic": "Level math is aligned with live replay: target 1 is one opening-range width from trigger and target 2 is two widths from trigger. Backtest exits at target 1 for a single conservative P/L row; live replay may continue the educational story toward target 2.",
    "invalidation_logic": "Intentional backtest confirmation layer: hard stops and targets remain immediate, but opening-range invalidation starts after the entry candle and requires consecutive closes back inside the range. Default is 2 closes below OR high for longs or above OR low for shorts; live replay may still record single-candle story milestones.",
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


@dataclass(frozen=True)
class IntradayProviderAvailability:
    available_start_date: date
    available_end_date: date
    provider_limit_reason: str


@dataclass(frozen=True)
class DayTradeEntryFilterConfig:
    direction_mode: DayTradeDirectionMode = DEFAULT_DIRECTION_MODE
    require_close_confirmed_breakout: bool = False
    require_vwap_alignment: bool = False
    min_or_width_pct: float | None = None
    max_or_width_pct: float | None = None
    no_trade_after_noon: bool = False

    def to_model(self) -> DayTradeBacktestEntryFilters:
        return DayTradeBacktestEntryFilters(
            direction_mode=self.direction_mode,
            require_close_confirmed_breakout=self.require_close_confirmed_breakout,
            require_vwap_alignment=self.require_vwap_alignment,
            min_or_width_pct=self.min_or_width_pct,
            max_or_width_pct=self.max_or_width_pct,
            no_trade_after_noon=self.no_trade_after_noon,
        )


CLOSE_CONFIRMED_ENTRY_FILTERS = DayTradeEntryFilterConfig(
    direction_mode=DEFAULT_DIRECTION_MODE,
    require_close_confirmed_breakout=True,
    require_vwap_alignment=False,
    min_or_width_pct=None,
    max_or_width_pct=None,
    no_trade_after_noon=False,
)

DIRECTION_MODE_LABELS: dict[DayTradeDirectionMode, str] = {
    "long_only": "Long only",
    "short_only": "Short only",
    "long_and_short": "Long + Short",
}
DIRECTION_MODE_ORDER: tuple[DayTradeDirectionMode, ...] = (
    "long_only",
    "short_only",
    "long_and_short",
)


class DayTradeBacktestDataError(ValueError):
    code = "day_trade_backtest_data_error"

    def __init__(
        self,
        message: str,
        *,
        availability: IntradayProviderAvailability,
    ) -> None:
        super().__init__(message)
        self.availability = availability

    def to_detail(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": str(self),
            "available_start_date": self.availability.available_start_date.isoformat(),
            "available_end_date": self.availability.available_end_date.isoformat(),
            "provider_limit_reason": self.availability.provider_limit_reason,
        }


class ProviderHistoryLimitExceededError(DayTradeBacktestDataError):
    code = "provider_history_limit_exceeded"


class NoMarketDataAvailableError(DayTradeBacktestDataError):
    code = "no_market_data_available"


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
        invalidation_confirmation_closes: int = 2,
        require_close_confirmed_breakout: bool = False,
        require_vwap_alignment: bool = False,
        min_or_width_pct: float | None = None,
        max_or_width_pct: float | None = None,
        no_trade_after_noon: bool = False,
        direction_mode: DayTradeDirectionMode = DEFAULT_DIRECTION_MODE,
    ) -> DayTradeBacktestResponse:
        if end < start:
            raise ValueError("end must be on or after start")
        if risk_per_trade <= 0:
            raise ValueError("risk_per_trade must be positive")
        if invalidation_confirmation_closes <= 0:
            raise ValueError("invalidation_confirmation_closes must be positive")
        entry_filters = DayTradeEntryFilterConfig(
            direction_mode=direction_mode,
            require_close_confirmed_breakout=require_close_confirmed_breakout,
            require_vwap_alignment=require_vwap_alignment,
            min_or_width_pct=min_or_width_pct,
            max_or_width_pct=max_or_width_pct,
            no_trade_after_noon=no_trade_after_noon,
        )
        _validate_entry_filters(entry_filters)

        symbol_upper = symbol.strip().upper()
        availability = intraday_provider_availability()
        _validate_requested_intraday_window(
            start=start,
            end=end,
            availability=availability,
        )
        hist = self._yfinance.get_history(
            symbol_upper,
            interval="5m",
            prepost=True,
            start=start,
            end=end + timedelta(days=1),
        )
        normalized = normalize_intraday_candles(hist)
        if not normalized:
            raise NoMarketDataAvailableError(
                (
                    f"No 5-minute market data is available for {symbol_upper} "
                    f"from {start.isoformat()} to {end.isoformat()}."
                ),
                availability=availability,
            )
        candles = [
            candle
            for candle in normalized
            if start <= market_date(candle.timestamp) <= end
            and _is_regular_session(candle.timestamp)
        ]
        if not candles:
            raise NoMarketDataAvailableError(
                (
                    f"No regular-session 5-minute candles are available for "
                    f"{symbol_upper} from {start.isoformat()} to {end.isoformat()}."
                ),
                availability=availability,
            )
        rows = simulate_day_trade_backtest(
            symbol=symbol_upper,
            candles=candles,
            risk_per_trade=risk_per_trade,
            invalidation_confirmation_closes=invalidation_confirmation_closes,
            entry_filters=entry_filters,
        )
        comparison_table = build_day_trade_comparison_table(
            symbol=symbol_upper,
            candles=candles,
            risk_per_trade=risk_per_trade,
            invalidation_confirmation_closes=invalidation_confirmation_closes,
        )
        return DayTradeBacktestResponse(
            symbol=symbol_upper,
            start=start,
            end=end,
            available_start_date=availability.available_start_date,
            available_end_date=availability.available_end_date,
            provider_limit_reason=availability.provider_limit_reason,
            risk_per_trade=risk_per_trade,
            invalidation_confirmation_closes=invalidation_confirmation_closes,
            entry_filters=entry_filters.to_model(),
            rows=rows,
            summary=summarize_day_trade_backtest(rows),
            top_winners=top_day_trade_winners(rows),
            top_losers=top_day_trade_losers(rows),
            comparison_table=comparison_table,
        )


def simulate_day_trade_backtest(
    *,
    symbol: str,
    candles: list[IntradayBacktestCandle],
    risk_per_trade: float,
    invalidation_confirmation_closes: int = 2,
    entry_filters: DayTradeEntryFilterConfig | None = None,
) -> list[DayTradeBacktestRow]:
    filters = entry_filters or DayTradeEntryFilterConfig()
    rows: list[DayTradeBacktestRow] = []
    for trading_day, day_candles in _group_by_market_date(candles).items():
        rows.append(
            _simulate_day(
                symbol=symbol,
                trading_day=trading_day,
                candles=day_candles,
                risk_per_trade=risk_per_trade,
                invalidation_confirmation_closes=invalidation_confirmation_closes,
                entry_filters=filters,
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
    stop_distances = [row.stop_distance for row in trades if row.stop_distance is not None]
    or_widths = [row.or_width for row in rows if row.or_width is not None]
    hold_minutes = [row.hold_minutes for row in trades if row.hold_minutes is not None]

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
        stop_hit_pct=_exit_rate(trades, "stop_hit"),
        target_1_hit_pct=_target_1_hit_rate(trades),
        target_2_hit_pct=_exit_rate(trades, "target_2_hit"),
        close_exit_pct=_exit_rate(trades, "close_exit"),
        invalidation_pct=_exit_rate(trades, "invalidated"),
        same_candle_invalidation_count=sum(
            1 for row in trades if row.entry_candle_closed_inside_or
        ),
        average_stop_distance=_average(stop_distances),
        average_or_width=_average(or_widths),
        average_hold_minutes=_average(hold_minutes),
    )


def top_day_trade_winners(
    rows: list[DayTradeBacktestRow],
) -> list[DayTradeBacktestRow]:
    trades = [
        row for row in rows if row.outcome != "no_trade" and row.r_multiple > 0
    ]
    return sorted(
        trades,
        key=lambda row: (row.r_multiple, row.dollar_pl, row.date),
        reverse=True,
    )[:10]


def top_day_trade_losers(
    rows: list[DayTradeBacktestRow],
) -> list[DayTradeBacktestRow]:
    trades = [
        row for row in rows if row.outcome != "no_trade" and row.r_multiple < 0
    ]
    return sorted(trades, key=lambda row: (row.r_multiple, row.dollar_pl, row.date))[:10]


def build_symbol_aggregate_row(
    symbol: str,
    rows: list[DayTradeBacktestRow],
) -> DayTradeBacktestSymbolAggregateRow:
    summary = summarize_day_trade_backtest(rows)
    return DayTradeBacktestSymbolAggregateRow(
        symbol=symbol.strip().upper(),
        total_trades=summary.total_trades,
        win_rate=summary.win_rate,
        average_r=summary.average_r,
        total_r=summary.total_r,
        profit_factor=summary.profit_factor,
        max_drawdown=summary.max_drawdown,
        target_1_hit_pct=summary.target_1_hit_pct,
        invalidation_pct=summary.invalidation_pct,
        stop_hit_pct=summary.stop_hit_pct,
    )


def build_multi_symbol_backtest_report(
    *,
    candidate_rows_by_symbol: dict[str, list[DayTradeBacktestRow]],
    baseline_rows_by_symbol: dict[str, list[DayTradeBacktestRow]] | None = None,
    direction_rows_by_mode: (
        dict[DayTradeDirectionMode, dict[str, list[DayTradeBacktestRow]]] | None
    ) = None,
    failed_symbols: list[DayTradeBacktestFailedSymbol] | None = None,
    candidate_filters: DayTradeEntryFilterConfig = CLOSE_CONFIRMED_ENTRY_FILTERS,
) -> DayTradeBacktestMultiSymbolReport:
    symbols = [symbol.strip().upper() for symbol in candidate_rows_by_symbol]
    aggregate = [
        build_symbol_aggregate_row(symbol, candidate_rows_by_symbol[symbol])
        for symbol in candidate_rows_by_symbol
    ]
    baseline_source = baseline_rows_by_symbol or {}
    baseline = [
        build_symbol_aggregate_row(symbol, baseline_source[symbol])
        for symbol in baseline_source
    ]
    direction_source = direction_rows_by_mode or {}
    direction_comparison = _highlight_direction_comparison(
        [
            _direction_comparison_row(mode, rows_by_symbol)
            for mode, rows_by_symbol in direction_source.items()
        ]
    )
    return DayTradeBacktestMultiSymbolReport(
        warning_label=SHORT_SAMPLE_WARNING_LABEL,
        candidate_scenario=PRIMARY_CANDIDATE_SCENARIO,  # type: ignore[arg-type]
        entry_filters=candidate_filters.to_model(),
        symbols=symbols,
        aggregate_comparison=aggregate,
        portfolio_summary=_portfolio_summary(candidate_rows_by_symbol, aggregate),
        baseline_comparison=baseline,
        direction_comparison=direction_comparison,
        failed_symbols=failed_symbols or [],
    )


def _direction_comparison_row(
    mode: DayTradeDirectionMode,
    rows_by_symbol: dict[str, list[DayTradeBacktestRow]],
) -> DayTradeBacktestDirectionComparisonRow:
    rows = sorted(
        [row for symbol_rows in rows_by_symbol.values() for row in symbol_rows],
        key=lambda row: (row.date, row.symbol, row.entry_time or datetime.min),
    )
    summary = summarize_day_trade_backtest(rows)
    return DayTradeBacktestDirectionComparisonRow(
        direction_mode=mode,
        label=DIRECTION_MODE_LABELS[mode],
        total_trades=summary.total_trades,
        win_rate=summary.win_rate,
        average_r=summary.average_r,
        total_r=summary.total_r,
        profit_factor=summary.profit_factor,
        max_drawdown=summary.max_drawdown,
        target_1_hit_pct=summary.target_1_hit_pct,
        invalidation_pct=summary.invalidation_pct,
    )


def _highlight_direction_comparison(
    rows: list[DayTradeBacktestDirectionComparisonRow],
) -> list[DayTradeBacktestDirectionComparisonRow]:
    if not rows:
        return rows
    best_total_r = max(row.total_r for row in rows)
    best_profit_factor = max(_profit_factor_score(row) for row in rows)
    best_max_drawdown = max(row.max_drawdown for row in rows)
    return [
        row.model_copy(
            update={
                "best_total_r": row.total_r == best_total_r,
                "best_profit_factor": _profit_factor_score(row)
                == best_profit_factor,
                "best_max_drawdown": row.max_drawdown == best_max_drawdown,
            }
        )
        for row in rows
    ]


def _profit_factor_score(row: DayTradeBacktestDirectionComparisonRow) -> float:
    if row.profit_factor is not None:
        return row.profit_factor
    if row.total_trades > 0 and row.total_r > 0:
        return float("inf")
    return float("-inf")


def _portfolio_summary(
    rows_by_symbol: dict[str, list[DayTradeBacktestRow]],
    aggregate: list[DayTradeBacktestSymbolAggregateRow],
) -> DayTradeBacktestPortfolioSummary:
    trades = [
        row
        for rows in rows_by_symbol.values()
        for row in rows
        if row.outcome != "no_trade"
    ]
    ordered = sorted(trades, key=lambda row: (row.date, row.symbol, row.entry_time or datetime.min))
    total_r = round(sum(row.r_multiple for row in ordered), 4)
    wins = [row for row in ordered if row.r_multiple > 0]
    losses = [row for row in ordered if row.r_multiple < 0]
    gross_win = sum(row.r_multiple for row in wins)
    gross_loss = abs(sum(row.r_multiple for row in losses))
    best = max(aggregate, key=lambda row: row.total_r, default=None)
    worst = min(aggregate, key=lambda row: row.total_r, default=None)
    return DayTradeBacktestPortfolioSummary(
        total_trades=len(ordered),
        total_r=total_r,
        average_r=round(total_r / len(ordered), 4) if ordered else 0.0,
        profit_factor=None if gross_loss == 0 else round(gross_win / gross_loss, 4),
        max_drawdown=round(_max_drawdown([row.r_multiple for row in ordered]), 4),
        best_symbol=best.symbol if best else None,
        worst_symbol=worst.symbol if worst else None,
    )


def build_day_trade_comparison_table(
    *,
    symbol: str,
    candles: list[IntradayBacktestCandle],
    risk_per_trade: float,
    invalidation_confirmation_closes: int = 2,
) -> list[DayTradeBacktestComparisonRow]:
    scenarios: list[tuple[str, DayTradeEntryFilterConfig]] = [
        ("baseline", DayTradeEntryFilterConfig()),
        (
            "close-confirmed breakout",
            DayTradeEntryFilterConfig(require_close_confirmed_breakout=True),
        ),
        ("VWAP-aligned", DayTradeEntryFilterConfig(require_vwap_alignment=True)),
        (
            "OR-width-filtered",
            DayTradeEntryFilterConfig(
                min_or_width_pct=DEFAULT_MIN_OR_WIDTH_PCT,
                max_or_width_pct=DEFAULT_MAX_OR_WIDTH_PCT,
            ),
        ),
        (
            "all filters combined",
            DayTradeEntryFilterConfig(
                require_close_confirmed_breakout=True,
                require_vwap_alignment=True,
                min_or_width_pct=DEFAULT_MIN_OR_WIDTH_PCT,
                max_or_width_pct=DEFAULT_MAX_OR_WIDTH_PCT,
                no_trade_after_noon=True,
            ),
        ),
    ]
    comparison: list[DayTradeBacktestComparisonRow] = []
    for scenario, filters in scenarios:
        rows = simulate_day_trade_backtest(
            symbol=symbol,
            candles=candles,
            risk_per_trade=risk_per_trade,
            invalidation_confirmation_closes=invalidation_confirmation_closes,
            entry_filters=filters,
        )
        summary = summarize_day_trade_backtest(rows)
        comparison.append(
            DayTradeBacktestComparisonRow(
                scenario=scenario,  # type: ignore[arg-type]
                total_trades=summary.total_trades,
                win_rate=summary.win_rate,
                average_r=summary.average_r,
                total_r=summary.total_r,
                profit_factor=summary.profit_factor,
                max_drawdown=summary.max_drawdown,
                target_1_hit_pct=summary.target_1_hit_pct,
                invalidation_pct=summary.invalidation_pct,
            )
        )
    return comparison


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


def intraday_provider_availability(
    as_of: date | None = None,
) -> IntradayProviderAvailability:
    available_end = as_of or datetime.now(EASTERN).date()
    return IntradayProviderAvailability(
        available_start_date=available_end - timedelta(days=YAHOO_INTRADAY_5M_HISTORY_DAYS),
        available_end_date=available_end,
        provider_limit_reason=YAHOO_INTRADAY_5M_LIMIT_REASON,
    )


def _validate_requested_intraday_window(
    *,
    start: date,
    end: date,
    availability: IntradayProviderAvailability,
) -> None:
    if start < availability.available_start_date or end > availability.available_end_date:
        raise ProviderHistoryLimitExceededError(
            (
                "Requested 5-minute intraday history is outside the available "
                f"Yahoo Finance window ({availability.available_start_date.isoformat()} "
                f"to {availability.available_end_date.isoformat()})."
            ),
            availability=availability,
        )


def _validate_entry_filters(filters: DayTradeEntryFilterConfig) -> None:
    if filters.min_or_width_pct is not None and filters.min_or_width_pct < 0:
        raise ValueError("min_or_width_pct must be non-negative")
    if filters.max_or_width_pct is not None and filters.max_or_width_pct < 0:
        raise ValueError("max_or_width_pct must be non-negative")
    if (
        filters.min_or_width_pct is not None
        and filters.max_or_width_pct is not None
        and filters.max_or_width_pct < filters.min_or_width_pct
    ):
        raise ValueError(
            "max_or_width_pct must be greater than or equal to min_or_width_pct"
        )


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
    invalidation_confirmation_closes: int,
    entry_filters: DayTradeEntryFilterConfig,
) -> DayTradeBacktestRow:
    candles = sorted(candles, key=lambda candle: candle.timestamp)
    levels = _opening_range_levels(candles)
    if levels is None:
        return _no_trade_row(symbol, trading_day)

    vwap_by_timestamp = _vwap_by_timestamp(candles)
    plan_vwap = _plan_vwap(candles)
    if not _or_width_filter_passes(candles, levels, entry_filters):
        return _no_trade_row(symbol, trading_day, levels)
    entry: IntradayBacktestCandle | None = None
    direction: DayTradeSetupDirection = "none"
    entry_price: float | None = None

    for candle in candles:
        local_time = _local_time(candle.timestamp)
        if local_time < OPENING_RANGE_END:
            continue
        if entry_filters.no_trade_after_noon and local_time >= NOON_ET:
            break
        long_hit = (
            _direction_allowed("long", entry_filters)
            and candle.high >= levels.long_trigger
            and _entry_filters_pass(
                direction="long",
                candle=candle,
                levels=levels,
                plan_vwap=plan_vwap,
                filters=entry_filters,
            )
        )
        short_hit = (
            _direction_allowed("short", entry_filters)
            and candle.low <= levels.short_trigger
            and _entry_filters_pass(
                direction="short",
                candle=candle,
                levels=levels,
                plan_vwap=plan_vwap,
                filters=entry_filters,
            )
        )
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
    exit_reason = "close_exit"
    stop_reason: str | None = None
    target_reason: str | None = None
    r_multiple = 0.0
    mfe = 0.0
    mae = 0.0
    consecutive_inside_closes = 0
    entry_index = candles.index(entry)
    entry_candle_closed_inside_or = _closed_back_inside_range(
        direction=direction,
        candle=entry,
        levels=levels,
    )

    for index, candle in enumerate(candles[entry_index:], start=entry_index):
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
        closed_back_inside = _closed_back_inside_range(
            direction=direction,
            candle=candle,
            levels=levels,
        )

        exit_candle = candle
        if stop_hit:
            exit_price = stop_price
            outcome = "loss"
            exit_reason = "stop_hit"
            stop_reason = _stop_reason(direction, plan_vwap)
            r_multiple = -1.0
            break
        if target_2_hit:
            exit_price = target_2
            outcome = "win"
            exit_reason = "target_2_hit"
            target_reason = "target_2_hit_two_opening_range_widths"
            r_multiple = 2.0
            break
        if target_1_hit:
            exit_price = target_1
            outcome = "win"
            exit_reason = "target_1_hit"
            target_reason = "target_1_hit_one_opening_range_width"
            r_multiple = 1.0
            break
        if index > entry_index:
            if closed_back_inside:
                consecutive_inside_closes += 1
            else:
                consecutive_inside_closes = 0
            if consecutive_inside_closes >= invalidation_confirmation_closes:
                exit_price = candle.close
                outcome = "invalidated"
                exit_reason = "invalidated"
                r_multiple = _r_multiple(direction, entry_price, stop_price, exit_price)
                break

    else:
        exit_candle = candles[-1]
        exit_price = exit_candle.close
        r_multiple = _r_multiple(direction, entry_price, stop_price, exit_price)
        outcome = _outcome_from_r(r_multiple)

    stop_distance = abs(entry_price - stop_price)
    target_distance = abs(target_1 - entry_price)
    hold_minutes = (exit_candle.timestamp - entry.timestamp).total_seconds() / 60
    rounded_r = round(r_multiple, 4)
    rounded_mfe = round(max(0.0, mfe), 4)
    rounded_mae = round(max(0.0, mae), 4)

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
        or_width=_round_price(levels.width),
        stop_distance=_round_price(stop_distance),
        target_distance=_round_price(target_distance),
        exit_time=exit_candle.timestamp,
        exit_price=_round_price(exit_price),
        exit_reason=exit_reason,
        stop_reason=stop_reason,
        target_reason=target_reason,
        hold_minutes=round(hold_minutes, 2),
        entry_candle_closed_inside_or=entry_candle_closed_inside_or,
        outcome=outcome,
        r_achieved=rounded_r,
        r_multiple=rounded_r,
        dollar_pl=round(rounded_r * risk_per_trade, 2),
        mfe=rounded_mfe,
        mae=rounded_mae,
        max_favorable_excursion=rounded_mfe,
        max_adverse_excursion=rounded_mae,
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
        or_width=_round_price(levels.width) if levels else None,
        exit_reason="no_trade",
        outcome="no_trade",
        r_achieved=0.0,
        r_multiple=0.0,
        dollar_pl=0.0,
        mfe=0.0,
        mae=0.0,
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


def _entry_filters_pass(
    *,
    direction: DayTradeSetupDirection,
    candle: IntradayBacktestCandle,
    levels: _DayLevels,
    plan_vwap: float | None,
    filters: DayTradeEntryFilterConfig,
) -> bool:
    if filters.require_close_confirmed_breakout:
        if direction == "long" and candle.close <= levels.opening_range_high:
            return False
        if direction == "short" and candle.close >= levels.opening_range_low:
            return False
    if filters.require_vwap_alignment:
        if plan_vwap is None:
            return False
        if direction == "long" and candle.close <= plan_vwap:
            return False
        if direction == "short" and candle.close >= plan_vwap:
            return False
    return True


def _direction_allowed(
    direction: DayTradeSetupDirection,
    filters: DayTradeEntryFilterConfig,
) -> bool:
    if filters.direction_mode == "long_only":
        return direction == "long"
    if filters.direction_mode == "short_only":
        return direction == "short"
    return True


def _or_width_filter_passes(
    candles: list[IntradayBacktestCandle],
    levels: _DayLevels,
    filters: DayTradeEntryFilterConfig,
) -> bool:
    if filters.min_or_width_pct is None and filters.max_or_width_pct is None:
        return True
    reference_price = _opening_range_reference_price(candles, levels)
    if reference_price <= 0:
        return False
    width_pct = levels.width / reference_price
    if filters.min_or_width_pct is not None and width_pct < filters.min_or_width_pct:
        return False
    if filters.max_or_width_pct is not None and width_pct > filters.max_or_width_pct:
        return False
    return True


def _opening_range_reference_price(
    candles: list[IntradayBacktestCandle],
    levels: _DayLevels,
) -> float:
    opening = [
        candle
        for candle in candles
        if MARKET_OPEN <= _local_time(candle.timestamp) < OPENING_RANGE_END
    ]
    if opening:
        return opening[-1].close
    return (levels.opening_range_high + levels.opening_range_low) / 2


def _closed_back_inside_range(
    *,
    direction: DayTradeSetupDirection,
    candle: IntradayBacktestCandle,
    levels: _DayLevels,
) -> bool:
    if direction == "long":
        return candle.close < levels.opening_range_high
    if direction == "short":
        return candle.close > levels.opening_range_low
    return False


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


def _stop_reason(
    direction: DayTradeSetupDirection,
    vwap_at_entry: float | None,
) -> str:
    if vwap_at_entry is not None:
        return f"{direction}_vwap_stop_hit"
    return f"{direction}_opening_range_stop_hit"


def _exit_rate(rows: list[DayTradeBacktestRow], exit_reason: str) -> float:
    if not rows:
        return 0.0
    hits = [row for row in rows if row.exit_reason == exit_reason]
    return round(len(hits) / len(rows), 4)


def _target_1_hit_rate(rows: list[DayTradeBacktestRow]) -> float:
    if not rows:
        return 0.0
    hits = [
        row
        for row in rows
        if row.exit_reason in {"target_1_hit", "target_2_hit"}
    ]
    return round(len(hits) / len(rows), 4)


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


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
