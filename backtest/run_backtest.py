"""CLI runner for walk-forward backtests on stored feature Parquets."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from backtest.metrics import summarize_predictions
from data.symbols import get_symbols
from data.store import load_features
from data.loader import load_symbol
from models.labels import add_labels
from models.walk_forward import WalkForwardConfig, WalkForwardResult, run_walk_forward
from models.xgb_model import XGBModelConfig


def load_labeled_universe(symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Load feature Parquets and attach labels using raw close prices."""
    labeled: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        features = load_features(symbol)
        raw = load_symbol(symbol)
        labeled[symbol.strip().upper()] = add_labels(features, raw["close"])
    return labeled


def run_backtest(
    symbols: Sequence[str] | None = None,
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Load labeled data and run walk-forward validation."""
    tickers = list(symbols) if symbols else get_symbols()
    labeled = load_labeled_universe(tickers)
    return run_walk_forward(labeled, config=config)


def format_backtest_report(summary: dict) -> str:
    lines = [
        "Walk-forward backtest summary",
        f"  Windows: {summary['n_windows']}",
        f"  Predictions: {summary['n_predictions']}",
        f"  Directional accuracy: {summary['directional_accuracy']:.4f}",
        f"  Sharpe ratio: {summary['sharpe_ratio']:.4f}",
        f"  Max drawdown: {summary['max_drawdown']:.4f}",
        f"  Profit factor: {summary['profit_factor']:.4f}",
    ]
    per_class = summary.get("per_class_accuracy") or {}
    for label in (-1, 0, 1):
        value = per_class.get(label, float("nan"))
        lines.append(f"  Class {label} accuracy: {value:.4f}")
    return "\n".join(lines)


def print_backtest_report(result: WalkForwardResult) -> None:
    summary = summarize_predictions(result.predictions)
    print(format_backtest_report(summary))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest on feature Parquets.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to include (default: data.symbols.DEFAULT_SYMBOLS)",
    )
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--min-train-samples", type=int, default=500)
    parser.add_argument("--min-test-samples", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = WalkForwardConfig(
        train_years=args.train_years,
        test_years=args.test_years,
        start_date=pd.Timestamp(args.start_date) if args.start_date else None,
        end_date=pd.Timestamp(args.end_date) if args.end_date else None,
        min_train_samples=args.min_train_samples,
        min_test_samples=args.min_test_samples,
        model_config=XGBModelConfig(),
    )

    result = run_backtest(args.symbols, config=config)
    print_backtest_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
