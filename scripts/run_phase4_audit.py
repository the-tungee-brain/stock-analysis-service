#!/usr/bin/env python3
"""Run Phase 4 signal durability research audit."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from analysis.phase4_audit import run_phase4_audit
from data.benchmarks import ensure_benchmark_ohlcv
from data.download import download_and_store_symbol
from data.paths import features_parquet_path, raw_parquet_path
from data.symbols import get_training_universe
from features.build_features import build_and_save_features


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


def _print_df(frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    if frame.empty:
        print("  (no data)")
        return
    cols = columns or list(frame.columns)
    rows = frame.to_dict(orient="records")
    widths = {col: max(len(col), *(len(_fmt(row.get(col))) for row in rows)) for col in cols}
    print("  ".join(f"{col:>{widths[col]}}" for col in cols))
    for row in rows:
        print("  ".join(f"{_fmt(row.get(col)):>{widths[col]}}" for col in cols))


def print_ablation_report(ablation: dict) -> None:
    baseline = ablation["baseline"]
    simple = ablation["simplicity_benchmark"]
    _print_section("1. Feature decomposition — full model baseline")
    print(
        f"  Features: {baseline['feature_count']} | IC: {_fmt(baseline['ic'])} | "
        f"Rank IC: {_fmt(baseline['rank_ic'])} | Sharpe: {_fmt(baseline['sharpe_ratio'])} | "
        f"Quintile spread: {_fmt(baseline['quintile_spread'])}"
    )

    _print_section("Leave-one-group-out (IC contribution = full IC − IC without group)")
    _print_df(
        ablation["leave_one_out"],
        ["group", "ic", "ic_delta_vs_full", "rank_ic_delta_vs_full", "n_features"],
    )

    _print_section("Only-group models (isolated edge)")
    _print_df(ablation["only_group"], ["group", "ic", "rank_ic", "ic_share_of_full", "n_features"])

    _print_section("5. Simplicity benchmark (RS + momentum + market context only)")
    print(
        f"  Features: {simple['feature_count']} | IC: {_fmt(simple['ic'])} | "
        f"Rank IC: {_fmt(simple['rank_ic'])} | Sharpe: {_fmt(simple['sharpe_ratio'])} | "
        f"Quintile spread: {_fmt(simple['quintile_spread'])}"
    )
    ic_retention = simple["ic"] / baseline["ic"] if baseline["ic"] else float("nan")
    sharpe_retention = (
        simple["sharpe_ratio"] / baseline["sharpe_ratio"]
        if baseline["sharpe_ratio"] not in (0, float("nan"))
        else float("nan")
    )
    print(
        f"  IC retention vs full: {_fmt(ic_retention)} | "
        f"Sharpe retention vs full: {_fmt(sharpe_retention)}"
    )


def print_regime_report(regimes: dict) -> None:
    _print_section("2. Regime attribution — SPY 200-DMA")
    _print_df(regimes["by_spy_200dma"], ["regime", "n_days", "ic", "rank_ic", "sharpe", "quintile_spread"])
    _print_section("Regime attribution — bull / bear")
    _print_df(regimes["by_spy_trend"], ["regime", "n_days", "ic", "rank_ic", "sharpe", "quintile_spread"])
    _print_section("Regime attribution — VIX")
    _print_df(regimes["by_vix_regime"], ["regime", "n_days", "ic", "rank_ic", "sharpe", "quintile_spread"])


def print_symbol_report(symbol_frame: pd.DataFrame, classes: dict) -> None:
    _print_section("3. Symbol-level persistence (top 10 by return contribution)")
    _print_df(
        symbol_frame.head(10),
        ["symbol", "ic", "rank_ic", "selection_rate", "gross_return_contribution", "symbol_sharpe"],
    )
    _print_section("Symbol classifications")
    print(f"  Persistent winners: {', '.join(classes['persistent_winners']) or '(none)'}")
    print(f"  Persistent losers: {', '.join(classes['persistent_losers']) or '(none)'}")
    print(f"  Signal diluters: {', '.join(classes['signal_diluters']) or '(none)'}")


def print_temporal_report(temporal: dict) -> None:
    _print_section("4. Temporal decay — IC by year")
    _print_df(temporal["ic_by_year"], ["year", "n_predictions", "ic", "rank_ic"])
    _print_section("Rolling diagnostics")
    print(f"  Signal trend: {temporal['signal_trend']}")
    print(f"  Latest rolling IC: {_fmt(temporal['latest_rolling_ic'])}")
    print(f"  Latest rolling quintile spread: {_fmt(temporal['latest_rolling_quintile_spread'])}")
    onset = temporal["decay_onset_date"]
    print(f"  Estimated decay onset (rolling IC < 0 after positive): {_fmt(onset) if onset else 'not detected'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 signal durability audit.")
    parser.add_argument("--universe", default="top20")
    parser.add_argument("--skip-data-prep", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    universe = args.universe.strip().lower()
    if not args.skip_data_prep:
        print(f"Preparing data for {universe}...", flush=True)
        ensure_universe_data(universe)

    print("Phase 4 signal durability audit", flush=True)
    print(f"Universe: {universe}", flush=True)
    print("Running walk-forward baseline + feature ablations (this may take several minutes)...", flush=True)

    result = run_phase4_audit(universe)
    print_ablation_report(result["ablation"])
    print_regime_report(result["regime_attribution"])
    print_symbol_report(result["symbol_persistence"], result["symbol_classes"])
    print_temporal_report(result["temporal_decay"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
