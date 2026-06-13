#!/usr/bin/env python3
"""Run the Day Trade / Missed Moves backtest without starting the API server."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.market.yfinance_adapter import YFinanceAdapter  # noqa: E402
from app.services.day_trade_backtest_service import (  # noqa: E402
    DayTradeBacktestService,
)

DEFAULT_OUTPUT = Path("artifacts/day_trade_backtest.json")


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NVDA", help="Ticker symbol")
    parser.add_argument("--start", type=_parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, required=True, help="YYYY-MM-DD")
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    service = DayTradeBacktestService(YFinanceAdapter())
    result = service.run_backtest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        risk_per_trade=args.risk_per_trade,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.model_dump(mode="json", by_alias=True), indent=2),
        encoding="utf-8",
    )

    summary = result.summary
    print("Day Trade backtest summary")
    print(f"Symbol: {result.symbol}")
    print(f"Period: {result.start} to {result.end}")
    print(f"Risk per trade: ${result.risk_per_trade:.2f}")
    print(f"Total trades: {summary.total_trades}")
    print(f"Win rate: {_format_rate(summary.win_rate)}")
    print(f"Avg R: {_format_number(summary.average_r)}")
    print(f"Total R: {_format_number(summary.total_r)}")
    print(f"Profit factor: {_format_number(summary.profit_factor)}")
    print(f"Max drawdown: {_format_number(summary.max_drawdown)}")
    print(f"Best day: {_format_number(summary.best_day)}")
    print(f"Worst day: {_format_number(summary.worst_day)}")
    print(f"JSON written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
