#!/usr/bin/env python3
"""Generate daily signal monitoring report for production portfolio."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from analysis.phase2_backtest import run_universe_walk_forward
from analysis.signal_monitoring import generate_monitoring_report
from backtest.production_portfolio import ProductionPortfolioConfig, run_production_portfolio_backtest
from data.benchmarks import ensure_benchmark_ohlcv
from data.download import download_and_store_symbol
from data.paths import features_parquet_path, raw_parquet_path
from data.symbols import get_training_universe
from features.build_features import build_and_save_features
from models.pattern_production import production_portfolio_config, production_walk_forward_config


def _fmt(value: float | int | str | None, *, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "nan"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def ensure_universe_data(universe: str, *, years: int = 15) -> None:
    ensure_benchmark_ohlcv(years=years)
    for symbol in get_training_universe(universe):
        if not raw_parquet_path(symbol).exists():
            download_and_store_symbol(symbol, years=years)
        if not features_parquet_path(symbol).exists():
            build_and_save_features(symbol)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run production signal monitoring report.")
    parser.add_argument("--universe", default="top20")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--max-position-weight", type=float, default=0.15)
    parser.add_argument("--recent-days", type=int, default=63)
    parser.add_argument("--baseline-days", type=int, default=252)
    parser.add_argument("--skip-data-prep", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = production_portfolio_config(
        universe=args.universe.strip().lower(),
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        hold_days=args.hold_days,
        max_position_weight=args.max_position_weight,
    )

    if not args.skip_data_prep:
        ensure_universe_data(config.universe)

    oos = run_universe_walk_forward(config.universe, walk_forward_config=production_walk_forward_config())
    portfolio = run_production_portfolio_backtest(
        oos["predictions"],
        oos["labeled_by_symbol"],
        config,
    )
    report = generate_monitoring_report(
        oos["predictions"],
        portfolio["periods"],
        recent_days=args.recent_days,
        baseline_days=args.baseline_days,
    )

    print("Production signal monitoring report")
    print(f"Universe: {config.universe} | Top N: {config.top_n}")
    print(f"Overall IC: {_fmt(report['overall_ic'])} | Rank IC: {_fmt(report['overall_rank_ic'])}")

    recent = report["recent_summary"]
    print(
        f"Recent window ({recent['window_days']}d): "
        f"IC={_fmt(recent['ic'])}, Rank IC={_fmt(recent['rank_ic'])}, "
        f"Hit rate={_fmt(recent['hit_rate'])}, Turnover={_fmt(recent['avg_turnover'])}, "
        f"Avg score={_fmt(recent['avg_rank_score'])}, "
        f"Realized excess={_fmt(recent['realized_excess_return'])}"
    )
    print(f"Positive IC days: {_fmt(recent['positive_ic_days'])}")

    print("\nSignal decay check")
    print("metric              recent    baseline  delta     pct_change")
    for row in report["baseline_comparison"].itertuples(index=False):
        print(
            f"{row.metric:<18}  {_fmt(row.recent):>8}  {_fmt(row.baseline):>8}  "
            f"{_fmt(row.delta):>8}  {_fmt(row.pct_change):>8}"
        )

    daily = report["daily_metrics"]
    if not daily.empty:
        print("\nLatest 10 daily monitoring rows")
        print(
            daily.tail(10).to_string(
                index=False,
                columns=[
                    "date",
                    "n_symbols",
                    "ic",
                    "rank_ic",
                    "hit_rate",
                    "turnover",
                    "avg_rank_score",
                    "realized_excess_return",
                ],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
