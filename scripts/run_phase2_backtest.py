#!/usr/bin/env python3
"""Run Phase 2 ranking portfolio backtests."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from analysis.phase2_backtest import (
    PHASE2_UNIVERSES,
    run_cost_sensitivity,
    run_phase2_universe_from_oos,
    run_rebalance_sensitivity,
    run_universe_walk_forward,
)
from data.benchmarks import ensure_benchmark_ohlcv
from data.download import download_and_store_symbol
from data.paths import features_parquet_path, raw_parquet_path
from data.symbols import get_training_universe
from features.build_features import build_and_save_features


def _fmt(value: float | int | str | None, *, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "nan"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def ensure_universe_data(universe: str, *, years: int = 15) -> None:
    ensure_benchmark_ohlcv(years=years)
    for symbol in get_training_universe(universe):
        if not raw_parquet_path(symbol).exists():
            download_and_store_symbol(symbol, years=years)
        if not features_parquet_path(symbol).exists():
            build_and_save_features(symbol)


def _print_section(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def _print_summary_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("  (no data)")
        return
    widths = {
        col: max(len(col), *(len(_fmt(row.get(col))) for row in rows))
        for col in columns
    }
    print("  ".join(f"{col:>{widths[col]}}" for col in columns))
    for row in rows:
        print("  ".join(f"{_fmt(row.get(col)):>{widths[col]}}" for col in columns))


def print_universe_report(result: dict) -> None:
    stats = result["signal_stats"]
    _print_section(f"Universe: {result['universe']} ({len(result['symbols'])} symbols)")
    print(f"  OOS predictions: {stats['n_predictions']}")
    print(f"  Overall IC: {_fmt(stats['overall_ic'])}")
    print(f"  Overall Rank IC: {_fmt(stats['overall_rank_ic'])}")
    print(
        f"  Settings: rebalance={result['rebalance_days']}d, "
        f"cost={_fmt(result['trade_cost_bps'], digits=1)} bps"
    )

    rows = []
    for item in result["strategy_results"]:
        summary = item["summary"]
        rows.append(
            {
                "strategy": item["strategy_key"],
                "cagr": summary["cagr"],
                "ann_return": summary["annualized_return"],
                "sharpe": summary["sharpe_ratio"],
                "sortino": summary["sortino_ratio"],
                "pf": summary["profit_factor"],
                "max_dd": summary["max_drawdown"],
                "turnover": summary["avg_turnover"],
                "periods": summary["n_periods"],
            }
        )
    _print_section("Portfolio strategies")
    _print_summary_table(
        rows,
        ["strategy", "cagr", "ann_return", "sharpe", "sortino", "pf", "max_dd", "turnover", "periods"],
    )

    primary = next(
        (item for item in result["strategy_results"] if item["strategy_key"] == "long_top_quintile"),
        result["strategy_results"][0],
    )
    attr = primary.get("attribution") or {}
    by_symbol = attr.get("by_symbol")
    _print_section("Alpha attribution (long top quintile)")
    if by_symbol is None or by_symbol.empty:
        print("  (no data)")
        return
    attr_rows = by_symbol.head(15).to_dict(orient="records")
    _print_summary_table(
        attr_rows,
        [
            "symbol",
            "gross_return_contribution",
            "contribution_share",
            "time_series_ic",
            "loo_ic_delta",
            "loss_contribution",
        ],
    )


def print_cost_sensitivity(title: str, frame: pd.DataFrame) -> None:
    _print_section(title)
    if frame.empty:
        print("  (no data)")
        return
    rows = frame.to_dict(orient="records")
    for row in rows:
        row["sharpe"] = row.pop("sharpe_ratio", row.get("sharpe"))
        row["sortino"] = row.pop("sortino_ratio", row.get("sortino"))
        row["profit_factor"] = row.get("profit_factor")
    _print_summary_table(
        rows,
        ["trade_cost_bps", "cagr", "sharpe", "sortino", "profit_factor", "max_drawdown", "avg_turnover"],
    )


def print_rebalance_sensitivity(title: str, frame: pd.DataFrame) -> None:
    _print_section(title)
    if frame.empty:
        print("  (no data)")
        return
    rows = frame.to_dict(orient="records")
    for row in rows:
        row["sharpe"] = row.pop("sharpe_ratio", row.get("sharpe"))
        row["sortino"] = row.pop("sortino_ratio", row.get("sortino"))
        row["profit_factor"] = row.get("profit_factor")
    _print_summary_table(
        rows,
        ["rebalance_days", "cagr", "sharpe", "sortino", "profit_factor", "max_drawdown", "avg_turnover"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 ranking portfolio backtests.")
    parser.add_argument(
        "--universe",
        nargs="+",
        default=list(PHASE2_UNIVERSES),
        help=f"Universes to run (default: {' '.join(PHASE2_UNIVERSES)})",
    )
    parser.add_argument("--trade-cost-bps", type=float, default=10.0)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--skip-data-prep", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    universes = [universe.strip().lower() for universe in args.universe]
    if not args.skip_data_prep:
        for universe in universes:
            print(f"Preparing data for {universe}...")
            ensure_universe_data(universe)

    from analysis.phase2_backtest import (
        run_cost_sensitivity,
        run_phase2_universe_from_oos,
        run_rebalance_sensitivity,
        run_universe_walk_forward,
    )

    print("Phase 2 ranking portfolio backtest", flush=True)
    print(f"Universes: {', '.join(universes)}", flush=True)

    for universe in universes:
        print(f"\n>>> Walk-forward OOS for {universe}...", flush=True)
        oos = run_universe_walk_forward(universe)
        universe_result = run_phase2_universe_from_oos(
            oos,
            trade_cost_bps=args.trade_cost_bps,
            rebalance_days=args.rebalance_days,
        )
        print_universe_report(universe_result)
        print_cost_sensitivity(
            f"{universe} cost sensitivity (long top quintile)",
            run_cost_sensitivity(
                oos,
                "long_top_quintile",
                rebalance_days=args.rebalance_days,
            ),
        )
        print_cost_sensitivity(
            f"{universe} cost sensitivity (long/short quintile)",
            run_cost_sensitivity(
                oos,
                "long_short_quintile",
                rebalance_days=args.rebalance_days,
            ),
        )
        print_rebalance_sensitivity(
            f"{universe} rebalance sensitivity (long top quintile)",
            run_rebalance_sensitivity(
                oos,
                "long_top_quintile",
                trade_cost_bps=args.trade_cost_bps,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
