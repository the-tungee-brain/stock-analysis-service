#!/usr/bin/env python3
"""Run the Day Trade / Missed Moves backtest without starting the API server."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.market.yfinance_adapter import YFinanceAdapter  # noqa: E402
from app.models.day_trade_backtest_models import DayTradeBacktestFailedSymbol  # noqa: E402
from app.services.day_trade_backtest_service import (  # noqa: E402
    CLOSE_CONFIRMED_ENTRY_FILTERS,
    DIRECTION_MODE_ORDER,
    DayTradeBacktestDataError,
    DayTradeBacktestService,
    build_multi_symbol_backtest_report,
    intraday_provider_availability,
)

DEFAULT_SYMBOLS = ["NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ"]
DEFAULT_OUTPUT = Path("artifacts/day_trade_backtest_multi_symbol.json")


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-MM-DD date, got {value!r}"
        ) from exc


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4g}"


def _parse_symbols(values: Sequence[str] | None) -> list[str]:
    if not values:
        return DEFAULT_SYMBOLS
    symbols: list[str] = []
    for value in values:
        symbols.extend(
            part.strip().upper()
            for part in value.split(",")
            if part.strip()
        )
    return symbols or DEFAULT_SYMBOLS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Ticker symbols, either space-separated or comma-separated",
    )
    parser.add_argument(
        "--symbol",
        dest="single_symbol",
        default=None,
        help="Run one ticker symbol (legacy alias)",
    )
    availability = intraday_provider_availability()
    default_start = availability.available_start_date + timedelta(days=2)
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=default_start,
        help="YYYY-MM-DD; defaults to the earliest available 5m Yahoo date",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=availability.available_end_date,
        help="YYYY-MM-DD; defaults to the latest available 5m Yahoo date",
    )
    parser.add_argument(
        "--risk-per-trade",
        "--risk_per_trade",
        dest="risk_per_trade",
        type=float,
        default=100.0,
        help="Dollar risk per trade",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path",
    )
    parser.add_argument(
        "--direction-mode",
        choices=DIRECTION_MODE_ORDER,
        default="long_only",
        help="Highlighted candidate direction mode",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbols = (
        _parse_symbols([args.single_symbol])
        if args.single_symbol
        else _parse_symbols(args.symbols)
    )
    service = DayTradeBacktestService(YFinanceAdapter())
    candidate_results = {}
    baseline_results = {}
    direction_results = {mode: {} for mode in DIRECTION_MODE_ORDER}
    failed_symbols: list[DayTradeBacktestFailedSymbol] = []
    candidate_filters = replace(
        CLOSE_CONFIRMED_ENTRY_FILTERS,
        direction_mode=args.direction_mode,
    )
    for symbol in symbols:
        try:
            symbol_direction_results = {}
            for mode in DIRECTION_MODE_ORDER:
                symbol_direction_results[mode] = service.run_backtest(
                    symbol=symbol,
                    start=args.start,
                    end=args.end,
                    risk_per_trade=args.risk_per_trade,
                    require_close_confirmed_breakout=True,
                    require_vwap_alignment=False,
                    min_or_width_pct=None,
                    max_or_width_pct=None,
                    no_trade_after_noon=False,
                    invalidation_confirmation_closes=2,
                    direction_mode=mode,
                )
            baseline_result = service.run_backtest(
                symbol=symbol,
                start=args.start,
                end=args.end,
                risk_per_trade=args.risk_per_trade,
                require_close_confirmed_breakout=False,
                require_vwap_alignment=False,
                min_or_width_pct=None,
                max_or_width_pct=None,
                no_trade_after_noon=False,
                invalidation_confirmation_closes=2,
            )
        except DayTradeBacktestDataError as exc:
            failed_symbols.append(
                DayTradeBacktestFailedSymbol(symbol=symbol, reason=str(exc))
            )
            print(f"Skipping {symbol}: {exc}", file=sys.stderr)
            continue

        for mode, result in symbol_direction_results.items():
            direction_results[mode][symbol] = result
        candidate_results[symbol] = symbol_direction_results[args.direction_mode]
        baseline_results[symbol] = baseline_result

    report = build_multi_symbol_backtest_report(
        candidate_rows_by_symbol={
            symbol: result.rows for symbol, result in candidate_results.items()
        },
        baseline_rows_by_symbol={
            symbol: result.rows for symbol, result in baseline_results.items()
        },
        direction_rows_by_mode={
            mode: {
                symbol: result.rows for symbol, result in mode_results.items()
            }
            for mode, mode_results in direction_results.items()
        },
        failed_symbols=failed_symbols,
        candidate_filters=candidate_filters,
    )
    payload = {
        **report.model_dump(mode="json", by_alias=True),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "risk_per_trade": args.risk_per_trade,
        "direction_mode": args.direction_mode,
        "results": {
            symbol: result.model_dump(mode="json", by_alias=True)
            for symbol, result in candidate_results.items()
        },
        "direction_results": {
            direction_mode: {
                symbol: result.model_dump(mode="json", by_alias=True)
                for symbol, result in mode_results.items()
            }
            for direction_mode, mode_results in direction_results.items()
        },
        "failed_symbols": [
            failed.model_dump(mode="json", by_alias=True)
            for failed in failed_symbols
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    summary = report.portfolio_summary
    print("Day Trade multi-symbol backtest summary")
    print("Candidate: close-confirmed breakout")
    print(f"Direction mode: {args.direction_mode}")
    print(f"Symbols: {', '.join(report.symbols)}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Risk per trade: ${args.risk_per_trade:.2f}")
    if failed_symbols:
        failed_label = ", ".join(failed.symbol for failed in failed_symbols)
        print(f"Failed symbols: {failed_label}")
    print(f"Total trades: {summary.total_trades}")
    print(f"Avg R: {_format_number(summary.average_r)}")
    print(f"Total R: {_format_number(summary.total_r)}")
    print(f"Profit factor: {_format_number(summary.profit_factor)}")
    print(f"Max drawdown: {_format_number(summary.max_drawdown)}")
    print(f"Best symbol: {summary.best_symbol or 'n/a'}")
    print(f"Worst symbol: {summary.worst_symbol or 'n/a'}")
    print("")
    print("Symbol comparison")
    for row in report.aggregate_comparison:
        print(
            f"{row.symbol}: trades={row.total_trades} win={_format_rate(row.win_rate)} "
            f"avgR={_format_number(row.average_r)} totalR={_format_number(row.total_r)} "
            f"PF={_format_number(row.profit_factor)} DD={_format_number(row.max_drawdown)}"
        )
    print("")
    print("Direction Mode")
    print("--------------")
    for row in report.direction_comparison:
        print(row.label)
        print(
            f"  trades={row.total_trades} win={_format_rate(row.win_rate)} "
            f"avgR={_format_number(row.average_r)} totalR={_format_number(row.total_r)} "
            f"PF={_format_number(row.profit_factor)} DD={_format_number(row.max_drawdown)} "
            f"target1={_format_rate(row.target_1_hit_pct)} "
            f"invalidation={_format_rate(row.invalidation_pct)}"
        )
    print(f"JSON written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
