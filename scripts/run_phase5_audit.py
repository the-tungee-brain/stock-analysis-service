#!/usr/bin/env python3
"""Run Phase 5 minimum viable alpha model comparison."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from analysis.phase5_minimal_models import run_phase5_comparison
from data.benchmarks import ensure_benchmark_ohlcv
from data.download import download_and_store_symbol
from data.paths import features_parquet_path, raw_parquet_path
from data.symbols import get_training_universe
from features.build_features import build_and_save_features


def _fmt(value: float | int | str | None, *, digits: int = 4, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "nan"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if pct:
        return f"{float(value):.{digits}f}"
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


def _print_df(frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.empty:
        print("  (no data)")
        return
    rows = frame.to_dict(orient="records")
    widths = {col: max(len(col), *(len(_fmt(row.get(col), pct=(col in {"max_drawdown"}))) for row in rows)) for col in columns}
    print("  ".join(f"{col:>{widths[col]}}" for col in columns))
    for row in rows:
        print("  ".join(f"{_fmt(row.get(col), pct=(col in {'max_drawdown'})):>{widths[col]}}" for col in columns))


def _recommendation(result: dict) -> None:
    summary = result["summary"]
    full_sharpe = result["full_model_sharpe"]
    full_ic = result["full_model_ic"]
    _print_section("Recommendation")
    candidates = summary[summary["model"].isin(["C", "D", "E"])].copy()
    best = candidates.sort_values(["sharpe", "n_features"], ascending=[False, True]).iloc[0]
    print(
        f"  Full model (F): IC={_fmt(full_ic)}, Sharpe={_fmt(full_sharpe)}, "
        f"{int(summary.loc[summary['model']=='F', 'n_features'].iloc[0])} features"
    )
    print(
        f"  Best simple candidate: Model {best['model']} ({best['label']}) — "
        f"IC={_fmt(best['ic'])} ({_fmt(best['ic_vs_full'])} of full), "
        f"Sharpe={_fmt(best['sharpe'])} ({_fmt(best['sharpe_vs_full'])} of full), "
        f"{int(best['n_features'])} features"
    )
    model_d = summary[summary["model"] == "D"].iloc[0]
    if model_d["sharpe_vs_full"] >= 0.8 and model_d["n_features"] <= 20:
        print(
            f"  Model D captures the essential thesis (RS + trend + regime) in "
            f"{int(model_d['n_features'])} features with {_fmt(model_d['sharpe_vs_full'])} of full Sharpe."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5 minimum viable alpha model comparison.")
    parser.add_argument("--universe", default="top20")
    parser.add_argument("--skip-data-prep", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    universe = args.universe.strip().lower()
    if not args.skip_data_prep:
        print(f"Preparing data for {universe}...", flush=True)
        ensure_universe_data(universe)

    print("Phase 5 minimum viable alpha model comparison", flush=True)
    print(f"Universe: {universe}", flush=True)
    print("Running models A–F (six walk-forward passes)...", flush=True)

    result = run_phase5_comparison(universe)

    _print_section("Complexity vs performance frontier")
    _print_df(
        result["frontier"],
        [
            "model",
            "n_features",
            "ic",
            "rank_ic",
            "sharpe",
            "sortino",
            "quintile_spread",
            "max_drawdown",
            "turnover",
            "sharpe_vs_full",
            "ic_vs_full",
        ],
    )

    _print_section("Stability score")
    _print_df(result["stability"], ["model", "ic_std", "positive_ic_pct", "positive_years_pct"])

    _print_section("Regime sensitivity (portfolio Sharpe)")
    _print_df(
        result["regime_sharpe"],
        ["model", "bull_sharpe", "bear_sharpe", "high_vix_sharpe", "medium_vix_sharpe", "low_vix_sharpe"],
    )

    _print_section("Simplicity score (Sharpe per feature)")
    _print_df(
        result["simplicity"],
        ["model", "n_features", "performance_sharpe", "performance_per_feature", "ic", "quintile_spread"],
    )

    _recommendation(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
