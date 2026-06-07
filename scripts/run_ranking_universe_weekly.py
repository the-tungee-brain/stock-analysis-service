#!/usr/bin/env python3
"""Weekly US equity universe refresh (liquidity filters)."""

from __future__ import annotations

import argparse
import logging
import sys

from ranking_pipeline.pipeline.weekly_universe import refresh_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh weekly ranking universe.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Limit symbols screened (useful for dev/smoke tests)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Symbols to process per batch (default: config/env)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Concurrent screening workers (default: config/env)",
    )
    parser.add_argument(
        "--memory-log-interval",
        type=int,
        default=None,
        help="Log RSS memory every N processed symbols (default: config/env)",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=None,
        help="Commit Oracle screening results every N symbols (default: config/env)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing universe screening checkpoint and restart the snapshot.",
    )
    args = parser.parse_args(argv)
    snapshot_id = refresh_universe(
        max_candidates=args.max_candidates,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        memory_log_interval=args.memory_log_interval,
        commit_interval=args.commit_interval,
        resume=not args.no_resume,
    )
    print({"snapshot_id": snapshot_id})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
