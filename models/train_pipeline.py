"""End-to-end pipeline: download OHLCV, build features, train and save model."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from data.download import download_and_store_all
from data.symbols import get_training_symbols
from features.build_features import build_and_save_all
from models.train_and_save import TrainAndSaveConfig, train_and_save


def default_train_end() -> str:
    return (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def run_pipeline(
    symbols: Sequence[str] | None = None,
    *,
    years: int = 15,
    train_end: str | None = None,
    train_start: str | None = None,
) -> dict[str, str | int]:
    tickers = list(symbols) if symbols else get_training_symbols()
    resolved_train_end = train_end or default_train_end()

    print(f"Downloading {len(tickers)} symbols ({years}y)...")
    download_and_store_all(tickers, years=years)

    print("Building feature matrices...")
    build_and_save_all(tickers)

    print(f"Training model through {resolved_train_end}...")
    config = TrainAndSaveConfig(
        symbols=tuple(ticker.strip().upper() for ticker in tickers),
        train_end_date=pd.Timestamp(resolved_train_end),
        train_start_date=pd.Timestamp(train_start) if train_start else None,
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

    result = run_pipeline(
        args.symbols,
        years=args.years,
        train_end=args.train_end,
        train_start=args.train_start,
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
