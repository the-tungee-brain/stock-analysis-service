#!/usr/bin/env python3
"""Run Phase 3 production portfolio, monitoring, and robustness analysis."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from analysis.phase3_backtest import run_phase3_analysis
from backtest.production_portfolio import format_production_portfolio_summary
from data.benchmarks import ensure_benchmark_ohlcv
from data.download import download_and_store_symbol
from data.paths import features_parquet_path, raw_parquet_path
from data.symbols import get_training_universe
from features.build_features import build_and_save_features
from models.pattern_production import production_portfolio_config


def _fmt(value: float | int | str | None, *, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "nan"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
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


def _print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("  (no data)")
        return
    widths = {col: max(len(col), *(len(_fmt(row.get(col))) for row in rows)) for col in columns}
    print("  ".join(f"{col:>{widths[col]}}" for col in columns))
    for row in rows:
        print("  ".join(f"{_fmt(row.get(col)):>{widths[col]}}" for col in columns))


def print_concentration_report(result: dict) -> None:
    concentration = result["portfolio"]["concentration"]
    _print_section("Symbol exposure (top 15)")
    symbol_rows = concentration["symbol_exposure"].head(15).to_dict(orient="records")
    _print_table(
        symbol_rows,
        ["symbol", "periods_held", "selection_rate", "avg_weight", "max_weight"],
    )

    _print_section("Sector exposure")
    sector_rows = concentration["sector_exposure"].to_dict(orient="records")
    _print_table(sector_rows, ["sector", "avg_weight", "max_weight", "periods"])

    _print_section("Contribution to risk (top 10)")
    risk_rows = concentration["contribution_to_risk"].head(10).to_dict(orient="records")
    _print_table(risk_rows, ["symbol", "risk_contribution", "risk_share", "avg_weight"])

    overlap = concentration["position_overlap"]
    if not overlap.empty:
        _print_section("Position overlap (recent 5 rebalances)")
        overlap_rows = overlap.tail(5).to_dict(orient="records")
        _print_table(
            overlap_rows,
            ["entry_date", "overlap_count", "overlap_ratio", "entered", "exited"],
        )


def print_monitoring_report(result: dict) -> None:
    monitoring = result["monitoring"]
    _print_section("Signal monitoring")
    recent = monitoring["recent_summary"]
    print(f"  Overall IC: {_fmt(monitoring['overall_ic'])}")
    print(f"  Overall Rank IC: {_fmt(monitoring['overall_rank_ic'])}")
    print(
        f"  Recent ({recent['window_days']}d): IC={_fmt(recent['ic'])}, "
        f"Rank IC={_fmt(recent['rank_ic'])}, Hit rate={_fmt(recent['hit_rate'])}, "
        f"Turnover={_fmt(recent['avg_turnover'])}, Avg score={_fmt(recent['avg_rank_score'])}"
    )
    _print_section("Signal decay check (recent vs baseline)")
    comparison_rows = monitoring["baseline_comparison"].to_dict(orient="records")
    _print_table(comparison_rows, ["metric", "recent", "baseline", "delta", "pct_change"])


def print_rolling_diagnostics(result: dict) -> None:
    rolling = result["rolling"]
    _print_section("Rolling 3-year diagnostics")
    print(f"  Signal trend: {rolling['signal_trend']}")
    print(f"  Latest rolling IC: {_fmt(rolling['latest_rolling_ic'])}")
    print(f"  Latest rolling quintile spread: {_fmt(rolling['latest_rolling_quintile_spread'])}")
    print(f"  Latest rolling Sharpe: {_fmt(rolling['latest_rolling_sharpe'])}")
    print(f"  Full-sample Sharpe: {_fmt(rolling['full_sample_sharpe'])}")


def print_regime_analysis(result: dict) -> None:
    regimes = result["regimes"]
    for title, key in (
        ("SPY 200-DMA regime", "by_spy_200dma"),
        ("SPY bull/bear regime", "by_spy_trend"),
        ("VIX regime", "by_vix_regime"),
    ):
        _print_section(title)
        rows = regimes[key].to_dict(orient="records")
        _print_table(rows, ["regime", "n_days", "ic", "rank_ic", "sharpe", "quintile_spread"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 production portfolio analysis.")
    parser.add_argument("--universe", default="top20")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--max-position-weight", type=float, default=0.15)
    parser.add_argument("--trade-cost-bps", type=float, default=10.0)
    parser.add_argument("--skip-data-prep", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = production_portfolio_config(
        universe=args.universe.strip().lower(),
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        hold_days=args.hold_days,
        max_position_weight=args.max_position_weight,
        trade_cost_bps=args.trade_cost_bps,
    )

    if not args.skip_data_prep:
        print(f"Preparing data for {config.universe}...", flush=True)
        ensure_universe_data(config.universe)

    print("Phase 3 production portfolio analysis", flush=True)
    print(f">>> Walk-forward OOS for {config.universe}...", flush=True)
    result = run_phase3_analysis(config)

    print(format_production_portfolio_summary(result["portfolio"]))
    print_concentration_report(result)
    print_monitoring_report(result)
    print_rolling_diagnostics(result)
    print_regime_analysis(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
