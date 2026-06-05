#!/usr/bin/env python3
"""Precompute Momentum Breakout scan snapshot rows for fast future serving."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.strategy.momentum_breakout_precompute_service import (  # noqa: E402
    precompute_momentum_breakout_scan_snapshot,
)

logging.basicConfig(level=logging.INFO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-universe",
        type=int,
        default=None,
        help="Emergency cap for scanned symbols. Default scans the full qualified universe.",
    )
    args = parser.parse_args()

    result = precompute_momentum_breakout_scan_snapshot(
        max_universe=args.max_universe,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
