from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.constants.watchlist_swatches import (
    DEFAULT_WATCHLIST_SWATCH_ID,
    normalize_watchlist_swatch_id,
)
from app.adapters.user.watchlist_adapter import WatchlistAdapter
from app.builders.finnhub_builder import FinnhubBuilder
from app.models.watchlist_models import (
    WatchlistFolderRecord,
    WatchlistWorkspaceResponse,
    WatchlistWorkspaceSyncRequest,
)
from app.services.ticker_service import TickerService

class WatchlistService:
    def __init__(
        self,
        *,
        watchlist_adapter: WatchlistAdapter,
        ticker_service: TickerService,
        finnhub_builder: FinnhubBuilder,
    ):
        self.watchlist_adapter = watchlist_adapter
        self.ticker_service = ticker_service
        self.finnhub_builder = finnhub_builder
        self._quote_workers = max(
            1,
            int(os.getenv("WATCHLIST_QUOTE_WORKERS", "8")),
        )

    @staticmethod
    def _normalize_folders(
        folders: list[WatchlistFolderRecord],
    ) -> list[WatchlistFolderRecord]:
        normalized: list[WatchlistFolderRecord] = []
        for folder in folders:
            swatch_id = normalize_watchlist_swatch_id(folder.swatch_id)
            normalized.append(
                folder.model_copy(
                    update={
                        "name": folder.name.strip(),
                        "icon_name": folder.icon_name or "folder.fill",
                        "swatch_id": swatch_id,
                        "symbols": [
                            item.model_copy(
                                update={
                                    "ticker": item.ticker.strip().upper(),
                                }
                            )
                            for item in folder.symbols
                        ],
                    }
                )
            )
        return normalized

    @staticmethod
    def _collect_symbols(folders: list[WatchlistFolderRecord]) -> list[str]:
        symbols: list[str] = []
        seen: set[str] = set()
        for folder in folders:
            for item in folder.symbols:
                ticker = item.ticker.strip().upper()
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    symbols.append(ticker)
        return symbols

    def _titles_for_symbols(self, symbols: list[str]) -> dict[str, str]:
        catalog = self.ticker_service.get_by_symbols(symbols)
        titles: dict[str, str] = {}
        for symbol in symbols:
            item = catalog.get(symbol)
            titles[symbol] = item.title if item and item.title else symbol
        return titles

    def _quotes_for_symbols(
        self, symbols: list[str]
    ) -> dict[str, tuple[float, float, float]]:
        if not symbols:
            return {}

        quotes: dict[str, tuple[float, float, float]] = {}
        workers = min(self._quote_workers, len(symbols))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.finnhub_builder.get_quote, symbol): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                quote = future.result()
                if quote is None or quote.pc <= 0:
                    continue
                day_change = quote.c - quote.pc
                day_change_percent = (day_change / quote.pc) * 100.0
                quotes[symbol] = (quote.c, day_change, day_change_percent)

        return quotes

    def get_workspace(
        self, *, user_id: str, include_quotes: bool = True
    ) -> WatchlistWorkspaceResponse:
        folders, workspace_version = self.watchlist_adapter.get_workspace_snapshot(
            user_id
        )
        symbols = self._collect_symbols(folders)
        titles = self._titles_for_symbols(symbols)
        quote_map = self._quotes_for_symbols(symbols) if include_quotes else None
        return self.watchlist_adapter.build_response(
            folders,
            titles_by_symbol=titles,
            quotes_by_symbol=quote_map,
            workspace_version=workspace_version,
        )

    def sync_workspace(
        self, *, user_id: str, payload: WatchlistWorkspaceSyncRequest
    ) -> WatchlistWorkspaceResponse:
        folders = self._normalize_folders(payload.folders)
        saved, workspace_version = self.watchlist_adapter.sync_workspace(
            user_id,
            folders,
            base_version=payload.base_version,
        )
        symbols = self._collect_symbols(saved)
        titles = self._titles_for_symbols(symbols)
        quote_map = self._quotes_for_symbols(symbols)
        return self.watchlist_adapter.build_response(
            saved,
            titles_by_symbol=titles,
            quotes_by_symbol=quote_map,
            workspace_version=workspace_version,
        )
