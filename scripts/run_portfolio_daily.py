#!/usr/bin/env python3
"""Construct portfolio from latest ranking run (downstream only)."""

from __future__ import annotations

import argparse
import logging
import sys

from ranking_pipeline.backtest.portfolio_sim import evaluate_portfolio_backtest
from ranking_pipeline.config import default_config
from ranking_pipeline.portfolio.config import default_portfolio_config
from ranking_pipeline.portfolio.constructor import construct_portfolio_from_run
from ranking_pipeline.portfolio.persistence import open_portfolio_store
from ranking_pipeline.storage.sqlite import open_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build portfolio from latest precomputed ranking."
    )
    parser.add_argument("--run-id", default=None, help="Ranking run id override")
    args = parser.parse_args(argv)

    pcfg = default_portfolio_config()
    result = construct_portfolio_from_run(args.run_id, portfolio_config=pcfg)

    rstore = open_store(default_config())
    pstore = open_portfolio_store(pcfg)
    prev = pstore.load_previous_weights(result["as_of_date"])
    bt_id = evaluate_portfolio_backtest(
        result["portfolio_id"],
        result["weights"],
        result["ranking_run_id"],
        result["as_of_date"],
        previous_weights=prev,
        ranking_store=rstore,
        portfolio_store=pstore,
        config=pcfg,
    )
    result["portfolio_backtest_id"] = bt_id
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
