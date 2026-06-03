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
    args = parser.parse_args(argv)
    snapshot_id = refresh_universe(max_candidates=args.max_candidates)
    print({"snapshot_id": snapshot_id})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
