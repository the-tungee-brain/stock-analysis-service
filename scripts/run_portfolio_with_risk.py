#!/usr/bin/env python3
"""Construct portfolio and apply risk layer (does not modify constructor)."""

from __future__ import annotations

import argparse
import logging
import sys

from ranking_pipeline.backtest.portfolio_risk_sim import evaluate_risk_backtest
from ranking_pipeline.config import default_config
from ranking_pipeline.portfolio.config import default_portfolio_config
from ranking_pipeline.portfolio.envelope import construct_portfolio_with_risk
from ranking_pipeline.portfolio.persistence import open_portfolio_store
from ranking_pipeline.risk.config import default_risk_config
from ranking_pipeline.storage.sqlite import open_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build portfolio + apply risk layer from latest ranking."
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    pcfg = default_portfolio_config()
    rcfg = default_risk_config()
    result = construct_portfolio_with_risk(args.run_id, portfolio_config=pcfg, risk_config=rcfg)

    rstore = open_store(default_config())
    pstore = open_portfolio_store(pcfg)
    prev = pstore.load_previous_weights(result["as_of_date"])
    bt_id = evaluate_risk_backtest(
        result["portfolio_id"],
        result.get("weights_before_risk", result["weights"]),
        result["ranking_run_id"],
        result["as_of_date"],
        previous_weights=prev,
        ranking_store=rstore,
        portfolio_store=pstore,
        portfolio_config=pcfg,
        risk_config=rcfg,
    )
    result["portfolio_backtest_id"] = bt_id
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
