#!/usr/bin/env python3
"""Download, featurize, and train the production Model C pattern model."""

from __future__ import annotations

import argparse
from typing import Sequence

from data.symbols import get_training_universe
from models.pattern_production import (
    PRODUCTION_MODEL_LABEL,
    PRODUCTION_TRAINING_UNIVERSE,
    format_production_universe,
    production_model_config,
    production_train_metadata_kwargs,
    resolve_production_symbols,
)
from models.train_pipeline import run_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the deployed 5-day ranking model (Model C: relative strength + trend) "
            f"on {PRODUCTION_TRAINING_UNIVERSE.upper()} "
            f"({format_production_universe()})."
        ),
    )
    parser.add_argument(
        "--extra-symbols",
        nargs="+",
        default=None,
        help=f"Additional symbols beyond {PRODUCTION_TRAINING_UNIVERSE}",
    )
    parser.add_argument(
        "--universe",
        default=PRODUCTION_TRAINING_UNIVERSE,
        help=f"Named training universe (default: {PRODUCTION_TRAINING_UNIVERSE})",
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
        tickers = resolve_production_symbols(
            extra_symbols=args.extra_symbols,
            universe=args.universe,
        )
    except ValueError as exc:
        parser.error(str(exc))

    training_symbols = get_training_universe(args.universe)
    result = run_pipeline(
        training_symbols,
        years=args.years,
        train_end=args.train_end,
        train_start=args.train_start,
        model_config=production_model_config(),
        train_metadata=production_train_metadata_kwargs(universe=args.universe),
    )
    print(
        "Production model training complete:",
        f"model={PRODUCTION_MODEL_LABEL}",
        f"symbols={','.join(training_symbols)}",
        f"rows={result['n_rows']}",
        f"features={result['n_features']}",
        f"train={result['train_start_date']}..{result['train_end_date']}",
        f"label_scheme={result.get('label_scheme')}",
        f"artifact={result['model_path']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
