#!/usr/bin/env python3
"""Backfill provider symbol metadata into Oracle for ranking universe screening."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yfinance as yf  # noqa: E402

from app.adapters.market.provider_symbol_profile_adapter import (  # noqa: E402
    ProviderSymbolProfileAdapter,
)
from app.adapters.market.yfinance_bootstrap import (  # noqa: E402
    configure_yfinance,
    yfinance_fetch_lock,
)
from ranking_pipeline.providers.symbol_metadata import (  # noqa: E402
    OracleProviderSymbolMetadataStore,
)
from ranking_pipeline.storage.oracle_screening import build_oracle_pool  # noqa: E402
from ranking_pipeline.universe.us_listings import fetch_all_us_equity_symbols  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _chunks(items: list[str], size: int):
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _fetch_info(symbol: str) -> dict:
    configure_yfinance()
    with yfinance_fetch_lock():
        return yf.Ticker(symbol).info or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    symbols = args.symbols or fetch_all_us_equity_symbols()
    if args.max_candidates:
        symbols = symbols[: args.max_candidates]
    symbol_keys = list(dict.fromkeys(sym.strip().upper() for sym in symbols if sym.strip()))

    pool = build_oracle_pool()
    provider_store = OracleProviderSymbolMetadataStore(pool)
    writer = ProviderSymbolProfileAdapter(pool)
    checked = fetched = written = skipped = failed = 0

    for batch in _chunks(symbol_keys, max(1, args.batch_size)):
        existing = provider_store.get_many(batch)
        missing = [
            sym
            for sym in batch
            if args.force or sym not in existing or existing[sym].market_cap is None
        ]
        skipped += len(batch) - len(missing)
        pending_writes: list[tuple[str, dict]] = []
        for symbol in missing:
            checked += 1
            try:
                info = _fetch_info(symbol)
                fetched += 1
                if not info:
                    failed += 1
                    logger.warning("No provider metadata returned for %s", symbol)
                    continue
                if args.dry_run:
                    logger.info("Would upsert provider metadata for %s", symbol)
                else:
                    pending_writes.append((symbol, info))
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
            except Exception as exc:
                failed += 1
                logger.warning("Provider metadata backfill failed for %s: %s", symbol, exc)
        if pending_writes:
            written += writer.upsert_success_many(
                "yahoo",
                pending_writes,
                fetched_at=datetime.now(timezone.utc),
            )
        logger.info(
            "Provider metadata backfill progress: checked=%d fetched=%d written=%d skipped=%d failed=%d",
            checked,
            fetched,
            written,
            skipped,
            failed,
        )

    print(
        {
            "symbols": len(symbol_keys),
            "checked": checked,
            "fetched": fetched,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "dry_run": args.dry_run,
        }
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
