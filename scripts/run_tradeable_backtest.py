#!/usr/bin/env python3
"""Run walk-forward backtest on UNIVERSE_TRADEABLE_V1 with the production config."""

from __future__ import annotations

import argparse
from typing import Sequence

from backtest.run_backtest import (
    print_backtest_report,
    print_compact_backtest_report,
    run_backtest,
)
from data.symbols import UNIVERSE_TRADEABLE_V1
from models.pattern_production import (
    format_tradeable_universe,
    production_strategy_config,
    production_walk_forward_config,
    resolve_tradeable_symbols,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run walk-forward backtest on the tradeable symbol universe "
            f"({format_tradeable_universe()}) with the default production config."
        ),
    )
    parser.add_argument(
        "--extra-symbols",
        nargs="+",
        default=None,
        help="Additional symbols to include beyond UNIVERSE_TRADEABLE_V1",
    )
    parser.add_argument(
        "--universe",
        default="tradeable_v1",
        help="Named universe to use as the base (default: tradeable_v1)",
    )
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Print the full backtest report instead of the compact summary",
    )
    parser.add_argument(
        "--random-baseline-runs",
        type=int,
        default=30,
        help="Monte Carlo runs for random-trade baseline (full report only)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random-trade baseline (full report only)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        tickers = resolve_tradeable_symbols(
            extra_symbols=args.extra_symbols,
            universe=args.universe,
        )
    except ValueError as exc:
        parser.error(str(exc))

    config = production_walk_forward_config(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    strategy = production_strategy_config()

    result, labeled = run_backtest(tickers, config=config, return_labeled=True)
    if args.full_report:
        print_backtest_report(
            result,
            labeled,
            label_scheme=config.label_scheme,
            strategy=strategy,
            n_random_runs=args.random_baseline_runs,
            random_state=args.random_seed,
        )
    else:
        print_compact_backtest_report(
            result,
            labeled,
            label_scheme=config.label_scheme,
            strategy=strategy,
            n_random_runs=args.random_baseline_runs,
            random_state=args.random_seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
