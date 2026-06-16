from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable

from app.models.day_trade_backtest_models import DayTradeDirectionMode
from app.services.emerging_leaders_service import build_emerging_leaders
from app.services.strategy.momentum_breakout_scan_universe import (
    load_production_scan_symbol_list,
)
from app.services.trade_replay_service import TradeReplayService, _is_trading_day

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolSourceResult:
    source: str
    symbols: list[str]
    skipped_reason: str | None = None


@dataclass
class DayTradeReplayPersistenceResult:
    trading_date: date
    symbols_processed: int = 0
    refreshes_attempted: int = 0
    events_created: int = 0
    missed_move_rows_after_run: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)


class TrackedDayTradeSymbolResolver:
    def __init__(
        self,
        *,
        oracle_pool: Any | None = None,
        watchlist_table: str = "WATCHLIST_ITEM",
        portfolio_table: str = "PORTFOLIO_SNAPSHOT",
    ) -> None:
        self.oracle_pool = oracle_pool
        self.watchlist_table = watchlist_table
        self.portfolio_table = portfolio_table

    def resolve(self) -> tuple[list[str], list[SymbolSourceResult]]:
        results = [
            self._configured_symbols(),
            self._watchlist_symbols(),
            self._portfolio_symbols(),
            self._top_movers_symbols(),
            self._emerging_leaders_symbols(),
        ]
        symbols = _dedupe_symbols(
            symbol for result in results for symbol in result.symbols
        )
        logger.info(
            "Day trade replay symbol resolver: total=%s sources=%s skipped=%s",
            len(symbols),
            {result.source: len(result.symbols) for result in results},
            {
                result.source: result.skipped_reason
                for result in results
                if result.skipped_reason
            },
        )
        return symbols, results

    def _configured_symbols(self) -> SymbolSourceResult:
        raw = os.getenv("DAY_TRADE_REPLAY_SYMBOLS", "")
        symbols = _dedupe_symbols(raw.replace(";", ",").split(","))
        return SymbolSourceResult(source="configured", symbols=symbols)

    def _watchlist_symbols(self) -> SymbolSourceResult:
        if self.oracle_pool is None:
            return SymbolSourceResult(
                source="watchlist",
                symbols=[],
                skipped_reason="oracle_pool_unavailable",
            )
        sql = f"SELECT DISTINCT symbol FROM {self.watchlist_table}"
        try:
            return SymbolSourceResult(
                source="watchlist",
                symbols=self._fetch_symbols(sql, {}),
            )
        except Exception as exc:
            logger.warning("Watchlist symbol source skipped", exc_info=True)
            return SymbolSourceResult(
                source="watchlist",
                symbols=[],
                skipped_reason=str(exc),
            )

    def _portfolio_symbols(self) -> SymbolSourceResult:
        if self.oracle_pool is None:
            return SymbolSourceResult(
                source="portfolio",
                symbols=[],
                skipped_reason="oracle_pool_unavailable",
            )
        sql = f"""
            SELECT positions_json
            FROM (
                SELECT positions_json,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id
                           ORDER BY snapshot_date DESC, created_at DESC
                       ) AS rn
                FROM {self.portfolio_table}
            )
            WHERE rn = 1
        """
        try:
            rows = self._fetchall(sql, {})
            symbols: list[str] = []
            for (positions_json,) in rows:
                text = _lob_text(positions_json)
                if not text:
                    continue
                for item in json.loads(text):
                    symbol = item.get("symbol") if isinstance(item, dict) else None
                    if isinstance(symbol, str):
                        symbols.append(symbol)
            return SymbolSourceResult(
                source="portfolio",
                symbols=_dedupe_symbols(symbols),
            )
        except Exception as exc:
            logger.warning("Portfolio symbol source skipped", exc_info=True)
            return SymbolSourceResult(
                source="portfolio",
                symbols=[],
                skipped_reason=str(exc),
            )

    def _top_movers_symbols(self) -> SymbolSourceResult:
        limit = int(os.getenv("DAY_TRADE_REPLAY_TOP_MOVERS_LIMIT", "50"))
        if limit <= 0:
            return SymbolSourceResult(
                source="top_movers",
                symbols=[],
                skipped_reason="disabled",
            )
        try:
            return SymbolSourceResult(
                source="top_movers",
                symbols=load_production_scan_symbol_list(max_symbols=limit),
            )
        except Exception as exc:
            logger.warning("Top Movers symbol source skipped", exc_info=True)
            return SymbolSourceResult(
                source="top_movers",
                symbols=[],
                skipped_reason=str(exc),
            )

    def _emerging_leaders_symbols(self) -> SymbolSourceResult:
        limit = int(os.getenv("DAY_TRADE_REPLAY_EMERGING_LEADERS_LIMIT", "50"))
        if limit <= 0:
            return SymbolSourceResult(
                source="emerging_leaders",
                symbols=[],
                skipped_reason="disabled",
            )
        try:
            response = build_emerging_leaders(limit=limit)
            return SymbolSourceResult(
                source="emerging_leaders",
                symbols=[item.symbol for item in response.items],
            )
        except Exception as exc:
            logger.warning("Emerging Leaders symbol source skipped", exc_info=True)
            return SymbolSourceResult(
                source="emerging_leaders",
                symbols=[],
                skipped_reason=str(exc),
            )

    def _fetch_symbols(self, sql: str, params: dict[str, Any]) -> list[str]:
        return _dedupe_symbols(row[0] for row in self._fetchall(sql, params))

    def _fetchall(self, sql: str, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        con = self.oracle_pool.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, params)
            return list(cur.fetchall())
        finally:
            con.close()


class DayTradeReplayPersistenceService:
    def __init__(
        self,
        *,
        trade_replay_service: TradeReplayService,
        symbol_resolver: TrackedDayTradeSymbolResolver,
    ) -> None:
        self.trade_replay_service = trade_replay_service
        self.symbol_resolver = symbol_resolver

    def persist_for_trading_date(
        self,
        trading_date: date,
        *,
        symbols: Iterable[str] | None = None,
        direction_modes: Iterable[DayTradeDirectionMode] = ("long_and_short",),
    ) -> DayTradeReplayPersistenceResult:
        result = DayTradeReplayPersistenceResult(trading_date=trading_date)
        if not _is_trading_day(trading_date):
            reason = f"{trading_date.isoformat()} is not a trading day"
            result.skipped.append(reason)
            logger.info("Day trade replay persistence skipped: %s", reason)
            return result

        if symbols is None:
            resolved_symbols, source_results = self.symbol_resolver.resolve()
            symbols_to_process = resolved_symbols
            result.source_counts = {
                source.source: len(source.symbols) for source in source_results
            }
            result.skipped.extend(
                f"{source.source}: {source.skipped_reason}"
                for source in source_results
                if source.skipped_reason
            )
        else:
            symbols_to_process = _dedupe_symbols(symbols)
            result.source_counts = {"explicit": len(symbols_to_process)}

        modes = list(direction_modes)
        logger.info(
            (
                "Day trade replay persistence started: trading_date=%s "
                "symbols=%s direction_modes=%s source_counts=%s"
            ),
            trading_date.isoformat(),
            len(symbols_to_process),
            modes,
            result.source_counts,
        )

        for symbol in symbols_to_process:
            symbol_rows_after = 0
            try:
                for direction_mode in modes:
                    refresh = self.trade_replay_service.refresh(
                        symbol=symbol,
                        workflow="day_trade",
                        event_date=trading_date,
                        direction_mode=direction_mode,
                    )
                    result.refreshes_attempted += 1
                    result.events_created += refresh.events_created
                rows = self.trade_replay_service.store.list_missed_moves(
                    symbol=symbol,
                    workflow="day_trade",
                    start_date=trading_date,
                    end_date=trading_date,
                )
                symbol_rows_after = len(rows)
                result.missed_move_rows_after_run += symbol_rows_after
                result.symbols_processed += 1
                logger.info(
                    (
                        "Day trade replay persistence symbol complete: "
                        "symbol=%s trading_date=%s modes=%s rows_after=%s "
                        "events_created_total=%s"
                    ),
                    symbol,
                    trading_date.isoformat(),
                    modes,
                    symbol_rows_after,
                    result.events_created,
                )
            except Exception as exc:
                message = f"{symbol}: {exc}"
                result.errors.append(message)
                logger.exception(
                    "Day trade replay persistence symbol failed: symbol=%s "
                    "trading_date=%s rows_after=%s",
                    symbol,
                    trading_date.isoformat(),
                    symbol_rows_after,
                )

        logger.info("Day trade replay persistence finished: %s", result)
        return result

    def backfill_symbol_date(
        self,
        symbol: str,
        trading_date: date,
        *,
        direction_modes: Iterable[DayTradeDirectionMode] = ("long_and_short",),
    ) -> DayTradeReplayPersistenceResult:
        return self.persist_for_trading_date(
            trading_date,
            symbols=[symbol],
            direction_modes=direction_modes,
        )

    def backfill_symbol_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        direction_modes: Iterable[DayTradeDirectionMode] = ("long_and_short",),
    ) -> list[DayTradeReplayPersistenceResult]:
        return [
            self.backfill_symbol_date(
                symbol,
                trading_date,
                direction_modes=direction_modes,
            )
            for trading_date in _trading_dates_between(start_date, end_date)
        ]

    def backfill_tracked_symbols_range(
        self,
        start_date: date,
        end_date: date,
        *,
        direction_modes: Iterable[DayTradeDirectionMode] = ("long_and_short",),
    ) -> list[DayTradeReplayPersistenceResult]:
        return [
            self.persist_for_trading_date(
                trading_date,
                direction_modes=direction_modes,
            )
            for trading_date in _trading_dates_between(start_date, end_date)
        ]


def _trading_dates_between(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    cursor = start_date
    dates: list[date] = []
    while cursor <= end_date:
        if _is_trading_day(cursor):
            dates.append(cursor)
        cursor += timedelta(days=1)
    return dates


def _dedupe_symbols(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def _lob_text(value: Any) -> str | None:
    if value is None:
        return None
    read = getattr(value, "read", None)
    if callable(read):
        return str(read())
    return str(value)
