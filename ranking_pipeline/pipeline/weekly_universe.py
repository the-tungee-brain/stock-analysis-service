"""Weekly US equity universe refresh with liquidity filters."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from app.adapters.market.yfinance_bootstrap import configure_yfinance, yfinance_fetch_lock
from data.download import download_symbol
from data.store import load_raw, raw_exists, save_raw
from ranking_pipeline.config import RankingPipelineConfig, default_config
from ranking_pipeline.pipeline.progress_log import log_batch_progress
from ranking_pipeline.universe.filters import screen_symbol_ohlcv
from ranking_pipeline.universe.us_listings import fetch_all_us_equity_symbols
from ranking_pipeline.storage.sqlite import open_store

logger = logging.getLogger(__name__)


def _fetch_market_cap(symbol: str) -> float | None:
    configure_yfinance()
    try:
        with yfinance_fetch_lock():
            info = yf.Ticker(symbol).info
        cap = info.get("marketCap")
        return float(cap) if cap else None
    except Exception:
        return None


def _screen_one(
    symbol: str,
    filters,
    lookback_days: int,
) -> dict:
    try:
        sym = symbol.strip().upper()
        if raw_exists(sym):
            ohlcv = load_raw(sym)
        else:
            ohlcv = download_symbol(sym, years=1)
            save_raw(ohlcv, sym)
        ohlcv = ohlcv.tail(lookback_days)
        cap = _fetch_market_cap(symbol)
        metrics = screen_symbol_ohlcv(symbol, ohlcv, market_cap=cap, filters=filters)
        return {
            "symbol": metrics.symbol,
            "last_close": metrics.last_close,
            "market_cap": metrics.market_cap,
            "avg_dollar_volume_20d": metrics.avg_dollar_volume_20d,
            "passed_filters": metrics.passed,
        }
    except Exception:
        return {
            "symbol": symbol.strip().upper(),
            "passed_filters": False,
        }


def refresh_universe(
    config: RankingPipelineConfig | None = None,
    *,
    max_candidates: int | None = None,
) -> str:
    """Download listings, screen liquidity, persist snapshot. Returns snapshot_id."""
    cfg = config or default_config()
    store = open_store(cfg)
    snapshot_id = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    symbols = fetch_all_us_equity_symbols()
    if max_candidates:
        symbols = symbols[:max_candidates]

    total = len(symbols)
    logger.info("Screening %d US listing candidates", total)
    members: list[dict] = []
    workers = max(1, cfg.max_workers)
    done = 0
    passed_so_far = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_screen_one, sym, cfg.liquidity, cfg.liquidity.screening_lookback_days): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            member = fut.result()
            members.append(member)
            done += 1
            if member.get("passed_filters"):
                passed_so_far += 1
            log_batch_progress(
                "Universe screen",
                done,
                total,
                detail=f"{passed_so_far} passed",
                step=25,
            )

    store.save_universe_snapshot(snapshot_id, members)
    passed = sum(1 for m in members if m.get("passed_filters"))
    logger.info("Universe snapshot %s: %d / %d passed", snapshot_id, passed, len(members))
    return snapshot_id
