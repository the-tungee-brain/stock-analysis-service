#!/usr/bin/env python3
"""Download, featurize, and train the production tradeable pattern model."""

from __future__ import annotations

import argparse
from typing import Sequence

from models.pattern_production import (
    format_tradeable_universe,
    production_model_config,
    production_train_metadata_kwargs,
    resolve_tradeable_symbols,
)
from models.train_pipeline import run_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the deployed 5-day trend model on the tradeable universe "
            f"({format_tradeable_universe()}) using binary labels and class weights."
        ),
    )
    parser.add_argument(
        "--extra-symbols",
        nargs="+",
        default=None,
        help="Additional symbols beyond UNIVERSE_TRADEABLE_V1",
    )
    parser.add_argument(
        "--universe",
        default="tradeable_v1",
        help="Named base universe (default: tradeable_v1)",
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

    try:
        tickers = resolve_tradeable_symbols(
            extra_symbols=args.extra_symbols,
            universe=args.universe,
        )
    except ValueError as exc:
        parser.error(str(exc))

    result = run_pipeline(
        tickers,
        years=args.years,
        train_end=args.train_end,
        train_start=args.train_start,
        model_config=production_model_config(),
        train_metadata=production_train_metadata_kwargs(universe=args.universe),
    )
    print(
        "Tradeable model training complete:",
        f"symbols={','.join(tickers)}",
        f"rows={result['n_rows']}",
        f"features={result['n_features']}",
        f"train={result['train_start_date']}..{result['train_end_date']}",
        f"label_scheme={result.get('label_scheme')}",
        f"model={result['model_path']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
