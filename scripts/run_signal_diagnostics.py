#!/usr/bin/env python3
"""Run out-of-sample signal diagnostics for the pattern trend model."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from analysis.signal_diagnostics import run_full_diagnostics
from data.symbols import UNIVERSE_TOP20, UNIVERSE_TRADEABLE_V1, get_training_universe, get_universe
from models.pattern_production import production_walk_forward_config, resolve_tradeable_symbols


def _fmt(value: float | int | str | bool | None, *, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "nan"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _print_section(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def _print_feature_table(
    df: pd.DataFrame,
    *,
    columns: list[str],
    head: int | None = 15,
) -> None:
    if df.empty:
        print("  (no data)")
        return
    view = df if head is None else df.head(head)
    col_widths = {
        col: max(len(col), *(len(_fmt(row[col])) for _, row in view.iterrows())) for col in columns
    }
    header = "  ".join(f"{col:>{col_widths[col]}}" for col in columns)
    print(f"  {header}")
    for _, row in view.iterrows():
        print("  ".join(f"{_fmt(row[col]):>{col_widths[col]}}" for col in columns))


def _resolve_symbols(universe: str, extra_symbols: Sequence[str] | None) -> list[str]:
    if universe == "tradeable_v1":
        return resolve_tradeable_symbols(extra_symbols=extra_symbols, universe=universe)
    symbols = get_training_universe(universe)
    if extra_symbols:
        seen = {symbol.upper() for symbol in symbols}
        for raw in extra_symbols:
            symbol = raw.strip().upper()
            if symbol and symbol not in seen and symbol not in {"SPY"}:
                symbols.append(symbol)
                seen.add(symbol)
    return symbols


def _universe_label(universe: str) -> str:
    if universe == "tradeable_v1":
        return ", ".join(UNIVERSE_TRADEABLE_V1)
    if universe == "top20":
        return ", ".join(get_training_universe("top20"))
    return ", ".join(get_universe(universe))


def print_diagnostics_report(diagnostics: dict, *, universe: str) -> None:
    print("Pattern model signal diagnostics")
    print(f"Universe key: {universe}")
    print(f"Training symbols ({len(diagnostics['symbols'])}): {', '.join(diagnostics['symbols'])}")
    print(f"OOS predictions: {diagnostics['n_predictions']}")
    print(f"Overall IC: {_fmt(diagnostics['overall_ic'])}")
    print(f"Overall Rank IC: {_fmt(diagnostics['overall_rank_ic'])}")

    dist = diagnostics["ic_distribution"]
    _print_section("IC distribution (daily cross-sectional periods)")
    print(f"  IC mean: {_fmt(dist['ic_mean'])}")
    print(f"  IC median: {_fmt(dist['ic_median'])}")
    print(f"  IC std dev: {_fmt(dist['ic_std'])}")
    print(f"  IC % positive periods: {_fmt(dist['ic_pct_positive'], digits=2)}")
    print(f"  Rank IC mean: {_fmt(dist['rank_ic_mean'])}")
    print(f"  Rank IC median: {_fmt(dist['rank_ic_median'])}")
    print(f"  Rank IC std dev: {_fmt(dist['rank_ic_std'])}")
    print(f"  Rank IC % positive periods: {_fmt(dist['rank_ic_pct_positive'], digits=2)}")
    print(f"  Number of IC periods: {dist['n_ic_periods']}")

    breadth = diagnostics["cross_sectional_breadth"]
    _print_section("Cross-sectional breadth")
    print(f"  Avg symbols per rebalance date: {_fmt(breadth['avg_symbols_per_date'], digits=2)}")
    print(f"  Median symbols per rebalance date: {_fmt(breadth['median_symbols_per_date'], digits=2)}")
    print(f"  Min / max symbols per date: {breadth['min_symbols_per_date']} / {breadth['max_symbols_per_date']}")
    print(f"  Rebalance dates: {breadth['n_rebalance_dates']}")
    print("  Observations per symbol:")
    for symbol, count in breadth["observations_per_symbol"].items():
        print(f"    {symbol}: {count}")

    strat = diagnostics["strategy_metrics"]
    bnh = diagnostics["buy_and_hold_metrics"]
    _print_section("Sharpe / PF comparison (threshold strategy vs buy-and-hold)")
    print(f"  Strategy Sharpe: {_fmt(strat['sharpe_ratio'])}")
    print(f"  Strategy PF: {_fmt(strat['profit_factor'])}")
    print(f"  Strategy max drawdown: {_fmt(strat['max_drawdown'])}")
    print(f"  Strategy trades: {strat['n_trades']}")
    print(f"  Directional accuracy: {_fmt(strat['directional_accuracy'])}")
    print(f"  Buy-and-hold Sharpe: {_fmt(bnh['sharpe_ratio'])}")
    print(f"  Buy-and-hold max drawdown: {_fmt(bnh['max_drawdown'])}")

    _print_section("A. IC by year")
    _print_feature_table(
        diagnostics["ic_by_year"],
        columns=["year", "n_predictions", "ic", "rank_ic"],
    )

    _print_section("A. IC by symbol (time-series correlation)")
    _print_feature_table(
        diagnostics["ic_by_symbol"],
        columns=["symbol", "n_predictions", "ic", "rank_ic"],
        head=None,
    )

    _print_section("B. Score bucket analysis")
    _print_feature_table(
        diagnostics["score_buckets"],
        columns=["bucket", "count", "avg_excess_return", "win_rate", "sharpe"],
        head=None,
    )

    quintile = diagnostics["quintile_portfolio"]
    _print_section("C. Top vs bottom quintile spread (overall)")
    print(f"  Top quintile avg excess return: {_fmt(quintile['top_bucket_avg_excess_return'])}")
    print(f"  Bottom quintile avg excess return: {_fmt(quintile['bottom_bucket_avg_excess_return'])}")
    print(f"  Top-bottom spread (avg): {_fmt(quintile['spread_avg'])}")
    print(f"  Spread Sharpe: {_fmt(quintile['spread_sharpe'])}")
    _print_feature_table(
        quintile["bucket_summary"],
        columns=["bucket", "bucket_label", "n_observations", "avg_excess_return"],
        head=None,
    )

    q_by_year = diagnostics["quintile_spread_by_year"]
    _print_section("C. Quintile spread by year (top vs bottom)")
    _print_feature_table(
        q_by_year,
        columns=[
            "year",
            "avg_symbols_per_date",
            "top_quintile_avg_excess_return",
            "bottom_quintile_avg_excess_return",
            "spread_avg",
            "spread_positive",
        ],
        head=None,
    )
    if not q_by_year.empty:
        positive_years = int(q_by_year["spread_positive"].sum())
        print(
            f"  Years with positive top-bottom spread: {positive_years}/{len(q_by_year)} "
            f"({_fmt(positive_years / len(q_by_year), digits=2)} rate)"
        )

    _print_section("A. Feature importance (top 15)")
    _print_feature_table(
        diagnostics["feature_importance"],
        columns=["feature", "importance", "importance_pct"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OOS signal diagnostics for the pattern model.")
    parser.add_argument(
        "--extra-symbols",
        nargs="+",
        default=None,
        help="Additional symbols beyond the selected universe",
    )
    parser.add_argument(
        "--universe",
        default="tradeable_v1",
        help="Named universe: tradeable_v1, top20, etc.",
    )
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    args = parser.parse_args(list(argv) if argv is not None else None)

    tickers = _resolve_symbols(args.universe, args.extra_symbols)
    config = production_walk_forward_config(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Running diagnostics")
    print(f"Named universe: {_universe_label(args.universe)}")
    if args.universe == "top20":
        print(f"(SPY excluded from training; benchmark-only. Full TOP20 list includes SPY.)")

    diagnostics = run_full_diagnostics(tickers, config)
    print_diagnostics_report(diagnostics, universe=args.universe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
