"""End-to-end pipeline: download OHLCV, build features, train and save model."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from data.benchmarks import BENCHMARK_SYMBOLS
from data.download import download_and_store_all
from data.symbols import get_symbols, get_training_symbols, get_universe, list_universe_names
from features.build_features import build_and_save_all
from models.train_and_save import TrainAndSaveConfig, train_and_save
from models.xgb_model import XGBModelConfig


def default_train_end() -> str:
    return (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def symbols_with_benchmarks(symbols: Sequence[str]) -> list[str]:
    """Return training symbols plus required benchmark OHLCV dependencies."""
    seen: set[str] = set()
    resolved: list[str] = []
    for symbol in [*symbols, *BENCHMARK_SYMBOLS]:
        symbol_upper = symbol.strip().upper()
        if not symbol_upper or symbol_upper in seen:
            continue
        seen.add(symbol_upper)
        resolved.append(symbol_upper)
    return resolved


def run_pipeline(
    symbols: Sequence[str] | None = None,
    *,
    years: int = 15,
    train_end: str | None = None,
    train_start: str | None = None,
    model_config: XGBModelConfig | None = None,
    train_metadata: dict | None = None,
    universe: str | None = None,
) -> dict[str, str | int]:
    tickers = list(symbols) if symbols else get_training_symbols()
    resolved_train_end = train_end or default_train_end()
    meta_kwargs = dict(train_metadata or {})
    download_symbols = symbols_with_benchmarks(tickers)

    print(f"Downloading {len(download_symbols)} symbols ({years}y, including benchmarks)...")
    download_and_store_all(download_symbols, years=years)

    print("Building feature matrices...")
    build_and_save_all(tickers)

    print(f"Training model through {resolved_train_end}...")
    config = TrainAndSaveConfig(
        symbols=tuple(ticker.strip().upper() for ticker in tickers),
        train_end_date=pd.Timestamp(resolved_train_end),
        train_start_date=pd.Timestamp(train_start) if train_start else None,
        model_config=model_config,
        min_up_prob=meta_kwargs.pop("min_up_prob", None),
        universe=meta_kwargs.pop("universe", universe),
        feature_columns=meta_kwargs.pop("feature_columns", None),
        extra_metadata=meta_kwargs or None,
    )
    return train_and_save(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download data, build features, and train the 5D trend model.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to include (default: stocks + ETFs from data.symbols)",
    )
    parser.add_argument(
        "--universe",
        default=None,
        help=f"Named symbol universe instead of --symbols (choices: {', '.join(list_universe_names())})",
    )
    parser.add_argument(
        "--label-scheme",
        choices=[
            "original_3class",
            "binary_updown",
            "wideband_3class",
            "binary_outperform_spy",
        ],
        default="original_3class",
        help="Target label scheme for training (default: original_3class)",
    )
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply inverse-frequency or scale_pos_weight during training",
    )
    parser.add_argument(
        "--min-up-prob",
        type=float,
        default=None,
        help="Store min P(up) threshold in artifact metadata for trade signals",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=15,
        help="Years of OHLCV history to download (default: 15)",
    )
    parser.add_argument(
        "--train-end",
        default=None,
        help="Last training date YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument(
        "--train-start",
        default=None,
        help="Optional first training date YYYY-MM-DD",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.symbols and args.universe:
        parser.error("Use either --symbols or --universe, not both")
    if args.universe:
        try:
            tickers = get_universe(args.universe)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        tickers = args.symbols

    model_config = XGBModelConfig(
        label_scheme=args.label_scheme,
        use_class_weights=args.use_class_weights,
    )
    train_metadata = {}
    if args.min_up_prob is not None:
        train_metadata["min_up_prob"] = args.min_up_prob
    if args.universe:
        train_metadata["universe"] = args.universe

    result = run_pipeline(
        tickers,
        years=args.years,
        train_end=args.train_end,
        train_start=args.train_start,
        model_config=model_config,
        train_metadata=train_metadata or None,
        universe=args.universe,
    )
    print(
        "Pipeline complete:",
        f"rows={result['n_rows']}",
        f"features={result['n_features']}",
        f"train={result['train_start_date']}..{result['train_end_date']}",
        f"model={result['model_path']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
