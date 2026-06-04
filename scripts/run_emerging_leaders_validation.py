#!/usr/bin/env python3
"""Capture Emerging Leaders daily snapshot and backfill forward returns."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.emerging_leaders_validation_service import (  # noqa: E402
    run_emerging_leaders_validation_job,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-date",
        help="ISO date for snapshot (default: latest ranking as-of or today)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace snapshot if date already exists",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Only backfill forward returns",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Only capture snapshot",
    )
    args = parser.parse_args()

    result = run_emerging_leaders_validation_job(
        snapshot_date=args.snapshot_date,
        force=args.force,
        skip_snapshot=args.skip_snapshot,
        skip_backfill=args.skip_backfill,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
