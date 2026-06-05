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


def _print_blocked_report(result: dict) -> None:
    diagnostics = result.get("blocked_diagnostics")
    if not isinstance(diagnostics, dict):
        print("\nBlocked diagnostics: unavailable")
        return

    print("\nBlocked diagnostics")
    print(
        "- historical profit factor: "
        f"{diagnostics.get('blocked_by_historical_profit_factor', 0)}"
    )
    print(
        "- historical trade count: "
        f"{diagnostics.get('blocked_by_historical_trade_count', 0)}"
    )
    print(f"- stop distance: {diagnostics.get('blocked_by_stop_distance', 0)}")
    print(f"- risk gate: {diagnostics.get('blocked_by_risk_gate', 0)}")

    top_blocked = diagnostics.get("top_blocked_symbols") or []
    if not top_blocked:
        print("- top blocked symbols: none")
        return

    print("- top blocked symbols:")
    for item in top_blocked:
        print(
            "  "
            f"{item.get('symbol')}: {item.get('primary_block_reason')} "
            f"(all: {', '.join(item.get('block_reasons') or [])})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-universe",
        type=int,
        default=None,
        help="Emergency cap for scanned symbols. Default scans the full qualified universe.",
    )
    parser.add_argument(
        "--explain-blocked",
        action="store_true",
        help="Print a short human-readable explanation of blocked candidates.",
    )
    args = parser.parse_args()

    result = precompute_momentum_breakout_scan_snapshot(
        max_universe=args.max_universe,
    )
    print(json.dumps(result, indent=2))
    if args.explain_blocked:
        _print_blocked_report(result)
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
