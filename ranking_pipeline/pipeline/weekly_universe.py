"""Weekly US equity universe refresh with liquidity filters."""

from __future__ import annotations

import gc
import logging
import os
import resource
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from data.download import download_symbol
from data.store import load_raw, raw_exists, save_raw
from ranking_pipeline.config import RankingPipelineConfig, default_config
from ranking_pipeline.pipeline.progress_log import log_batch_progress
from ranking_pipeline.providers.symbol_metadata import (
    ProviderFirstSymbolMetadataResolver,
    build_provider_first_symbol_metadata_resolver,
)
from ranking_pipeline.storage.oracle_screening import (
    OracleScreeningStore,
    open_oracle_screening_store,
)
from ranking_pipeline.universe.filters import screen_symbol_ohlcv
from ranking_pipeline.universe.us_listings import fetch_all_us_equity_symbols
from ranking_pipeline.storage.sqlite import open_store

logger = logging.getLogger(__name__)

_STOP_REQUESTED = False


def _rss_mb() -> float:
    """Return resident memory in MB for the current process."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        except (OSError, IndexError, ValueError):
            pass
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _request_stop(signum, _frame) -> None:  # noqa: ANN001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    logger.warning("Universe screen received signal %s; stopping after current batch", signum)


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _screen_one(
    symbol: str,
    filters,
    lookback_days: int,
    market_cap: float | None = None,
) -> dict:
    sym = symbol.strip().upper()
    try:
        if raw_exists(sym):
            ohlcv = load_raw(sym)
        else:
            ohlcv = download_symbol(sym, years=1)
            save_raw(ohlcv, sym)
        screen_ohlcv = ohlcv.tail(lookback_days).copy()
        del ohlcv
        metrics = screen_symbol_ohlcv(
            sym,
            screen_ohlcv,
            market_cap=market_cap,
            filters=filters,
        )
        del screen_ohlcv
        reasons = {
            "min_price": bool(metrics.last_close > filters.min_price),
            "min_avg_dollar_volume_20d": bool(
                metrics.avg_dollar_volume_20d >= filters.min_avg_dollar_volume_20d
            ),
            "min_market_cap": bool(
                metrics.market_cap is not None and metrics.market_cap >= filters.min_market_cap
            ),
        }
        return {
            "symbol": metrics.symbol,
            "last_close": metrics.last_close,
            "market_cap": metrics.market_cap,
            "avg_dollar_volume_20d": metrics.avg_dollar_volume_20d,
            "passed_filters": metrics.passed,
            "reasons": reasons,
        }
    except Exception as exc:
        message = str(exc)
        logger.warning("Universe screen skipped %s: %s", sym, message)
        return {
            "symbol": sym,
            "passed_filters": False,
            "error": message[:1000],
            "reasons": {"error": message[:500]},
        }


def refresh_universe(
    config: RankingPipelineConfig | None = None,
    *,
    max_candidates: int | None = None,
    batch_size: int | None = None,
    max_workers: int | None = None,
    memory_log_interval: int | None = None,
    commit_interval: int | None = None,
    resume: bool = True,
    screen_store: OracleScreeningStore | None = None,
    metadata_resolver: ProviderFirstSymbolMetadataResolver | None = None,
) -> str:
    """Download listings, screen liquidity, persist snapshot. Returns snapshot_id."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    previous_sigterm = None
    try:
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _request_stop)
    except ValueError:
        logger.warning("SIGTERM checkpoint handler unavailable outside the main thread")
    cfg = config or default_config()
    store = open_store(cfg)
    snapshot_id = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    symbols = fetch_all_us_equity_symbols()
    if max_candidates:
        symbols = symbols[:max_candidates]
    symbols = [sym.strip().upper() for sym in symbols if sym.strip()]

    total = len(symbols)
    workers = max(1, int(max_workers or cfg.universe_max_workers))
    batch = max(1, int(batch_size or cfg.universe_batch_size))
    log_every = max(1, int(memory_log_interval or cfg.universe_memory_log_interval))
    commit_every = max(1, int(commit_interval or cfg.universe_commit_interval))
    oracle_store = screen_store or open_oracle_screening_store()
    resolver = metadata_resolver or build_provider_first_symbol_metadata_resolver()
    screen_run = oracle_store.start_or_resume_run(
        snapshot_id=snapshot_id,
        total_count=total,
        batch_size=batch,
        max_workers=workers,
        resume=resume,
    )
    snapshot_id = screen_run.snapshot_id
    done = screen_run.processed_count
    passed_so_far = screen_run.passed_count
    if done:
        logger.info(
            "Resuming universe screen run %s snapshot %s: %d/%d symbols already persisted",
            screen_run.run_id,
            screen_run.snapshot_id,
            done,
            total,
        )
    pending_since_commit = 0

    logger.info(
        "Screening %d US listing candidates (%d remaining, run_id=%s, batch_size=%d, workers=%d, commit_interval=%d)",
        total,
        max(0, total - done),
        screen_run.run_id,
        batch,
        workers,
        commit_every,
    )
    logger.info(
        "Universe screen memory: rss_mb=%.1f batch_size=%d processed=%d/%d",
        _rss_mb(),
        batch,
        done,
        total,
    )

    try:
        for candidate_batch in _chunks(symbols, batch):
            completed_in_batch = oracle_store.completed_symbols_for(
                screen_run.run_id,
                candidate_batch,
            )
            batch_symbols = [sym for sym in candidate_batch if sym not in completed_in_batch]
            if not batch_symbols:
                continue
            metadata_by_symbol = resolver.resolve_many(
                batch_symbols,
                required_fields=("market_cap",),
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _screen_one,
                        sym,
                        cfg.liquidity,
                        cfg.liquidity.screening_lookback_days,
                        (
                            metadata_by_symbol[sym].market_cap
                            if sym in metadata_by_symbol
                            else None
                        ),
                    ): sym
                    for sym in batch_symbols
                }
                for fut in as_completed(futures):
                    symbol = futures[fut]
                    try:
                        member = fut.result()
                    except Exception as exc:
                        message = str(exc)
                        logger.warning("Universe screen skipped %s: %s", symbol, message)
                        member = {
                            "symbol": symbol,
                            "passed_filters": False,
                            "error": message[:1000],
                            "reasons": {"error": message[:500]},
                        }
                    oracle_store.upsert_result(screen_run.run_id, member)
                    if member.get("error"):
                        oracle_store.upsert_error(screen_run.run_id, symbol, str(member["error"]))
                    done += 1
                    pending_since_commit += 1
                    if member.get("passed_filters"):
                        passed_so_far += 1
                    should_commit = pending_since_commit >= commit_every or done == total
                    oracle_store.update_progress(
                        screen_run.run_id,
                        processed_count=done,
                        passed_count=passed_so_far,
                        rss_mb=_rss_mb(),
                        commit=should_commit,
                    )
                    if should_commit:
                        pending_since_commit = 0
                    log_batch_progress(
                        "Universe screen",
                        done,
                        total,
                        detail=f"{passed_so_far} passed",
                        step=25,
                    )
                    if done == total or done % log_every == 0:
                        logger.info(
                            "Universe screen memory: rss_mb=%.1f batch_size=%d processed=%d/%d",
                            _rss_mb(),
                            len(batch_symbols),
                            done,
                            total,
                        )
            gc.collect()
            if _STOP_REQUESTED:
                oracle_store.mark_interrupted(
                    screen_run.run_id,
                    processed_count=done,
                    passed_count=passed_so_far,
                )
                logger.warning(
                    "Universe screen run %s interrupted at %d/%d; rerun to resume",
                    screen_run.run_id,
                    done,
                    total,
                )
                raise SystemExit(130)

        result_counts = oracle_store.result_counts(screen_run.run_id)
        if result_counts.total_count < total:
            oracle_store.mark_failed(
                screen_run.run_id,
                processed_count=result_counts.total_count,
                passed_count=result_counts.passed_count,
            )
            raise RuntimeError(
                "Universe screen did not persist results for every candidate "
                f"(run_id={screen_run.run_id}, results={result_counts.total_count}, "
                f"expected={total})"
            )
        if result_counts.passed_count <= 0:
            oracle_store.mark_failed(
                screen_run.run_id,
                processed_count=result_counts.total_count,
                passed_count=result_counts.passed_count,
            )
            raise RuntimeError(
                "Universe screen produced zero passed symbols in SCREEN_RESULTS "
                f"(run_id={screen_run.run_id}, total_results={result_counts.total_count}). "
                "Check market-cap metadata/yfinance fallback before rerunning."
            )
        if result_counts.passed_count != passed_so_far:
            logger.warning(
                "Universe screen passed-count mismatch for run %s: in-memory=%d oracle_results=%d; using Oracle result count",
                screen_run.run_id,
                passed_so_far,
                result_counts.passed_count,
            )
        oracle_store.finalize_run(
            screen_run.run_id,
            processed_count=result_counts.total_count,
            passed_count=result_counts.passed_count,
        )
        store.start_universe_snapshot(snapshot_id)
        passed = 0
        for page in oracle_store.iter_results(screen_run.run_id, page_size=batch):
            store.append_universe_members(snapshot_id, page)
            passed += sum(1 for member in page if member.get("passed_filters"))
            del page
        store.finalize_universe_snapshot(snapshot_id)
        if passed <= 0:
            raise RuntimeError(
                f"Universe snapshot {snapshot_id} has zero passed SQLite members after Oracle projection"
            )
        logger.info("Universe snapshot %s: %d / %d passed", snapshot_id, passed, total)
        return snapshot_id
    except KeyboardInterrupt:
        oracle_store.mark_interrupted(
            screen_run.run_id,
            processed_count=done,
            passed_count=passed_so_far,
        )
        logger.warning(
            "Universe screen run %s interrupted at %d/%d; rerun to resume",
            screen_run.run_id,
            done,
            total,
        )
        raise
    finally:
        oracle_store.close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
