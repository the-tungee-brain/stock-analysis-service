#!/usr/bin/env python3
"""Backfill paper-trading performance rows for existing Momentum Breakout alerts."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.strategy.momentum_breakout_alert_store_factory import (  # noqa: E402
    build_momentum_breakout_alert_store,
)
from app.adapters.strategy.paper_trade_performance_store_factory import (  # noqa: E402
    build_paper_trade_performance_store,
)
from app.services.strategy.momentum_breakout_paper_trade_backfill_service import (  # noqa: E402
    MomentumBreakoutPaperTradeBackfillService,
)
from app.services.strategy.paper_trade_performance_service import (  # noqa: E402
    PaperTradePerformanceService,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_VALID_ENVIRONMENTS = frozenset({"production", "staging"})


def _apply_environment(environment: str) -> None:
    if environment not in _VALID_ENVIRONMENTS:
        raise ValueError(
            f"environment must be one of {sorted(_VALID_ENVIRONMENTS)}, got {environment!r}"
        )
    os.environ["MB_ALERT_STORE"] = "oracle"
    os.environ["MB_PAPER_TRADE_STORE"] = "oracle"
    if environment == "production":
        os.environ["MB_PRODUCTION"] = "true"
        os.environ["ENV"] = "production"
    else:
        os.environ.pop("MB_PRODUCTION", None)
        os.environ["ENV"] = "staging"


def _build_pool():
    try:
        import oracledb
    except ImportError as exc:
        raise RuntimeError(
            "oracledb is required for Oracle backfill. Install requirements.txt."
        ) from exc
    user = os.getenv("POWERPOCKETDB_USER")
    password = os.getenv("POWERPOCKETDB_PASSWORD")
    dsn = os.getenv("POWERPOCKETDB_TP_TNS")
    if not (user and password and dsn):
        raise RuntimeError(
            "POWERPOCKETDB_USER, POWERPOCKETDB_PASSWORD, and POWERPOCKETDB_TP_TNS "
            "must be set for Oracle backfill (same as the API service)."
        )
    return oracledb.create_pool(user=user, password=password, dsn=dsn, min=1, max=2)


def _print_summary(result, *, dry_run: bool) -> None:
    mode = "DRY RUN — " if dry_run else ""
    print(f"{mode}alerts scanned: {result.alerts_scanned}")
    label = "rows to create" if dry_run else "rows created"
    print(f"{mode}{label}:   {result.rows_created}")
    print(f"{mode}rows skipped:   {result.rows_skipped}")
    print(f"{mode}rows failed:    {result.rows_failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Maximum alerts to scan",
    )
    parser.add_argument(
        "--environment",
        choices=sorted(_VALID_ENVIRONMENTS),
        default=None,
        help="Target deployment (sets Oracle store mode and ENV labels)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan alerts and report missing paper rows without writing",
    )
    args = parser.parse_args()

    if args.environment:
        _apply_environment(args.environment)
        logger.info("Environment: %s", args.environment)
    if args.dry_run:
        logger.info("Dry run enabled — no paper-trade rows will be written")

    pool = _build_pool()
    alert_store = build_momentum_breakout_alert_store(pool)
    paper_store = build_paper_trade_performance_store(pool)
    paper_service = PaperTradePerformanceService(paper_store)
    backfill = MomentumBreakoutPaperTradeBackfillService(
        alert_store=alert_store,
        paper_trade_service=paper_service,
    )

    result = backfill.run(limit=args.limit, dry_run=args.dry_run)
    _print_summary(result, dry_run=args.dry_run)
    for failure in result.failures:
        logger.error("%s", failure)
    if result.rows_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
