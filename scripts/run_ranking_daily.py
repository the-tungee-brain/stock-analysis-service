#!/usr/bin/env python3
"""Nightly ranking pipeline: OHLCV → features → rank → SQLite."""

from __future__ import annotations

import argparse
import logging
import sys

from ranking_pipeline.pipeline.daily import run_daily_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run nightly stock ranking pipeline.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional symbol override (default: active universe)",
    )
    args = parser.parse_args(argv)
    result = run_daily_pipeline(symbols=args.symbols)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
