from __future__ import annotations

import re
from typing import Iterable

import oracledb

from app.models.watchlist_models import (
    WatchlistFolderRecord,
    WatchlistFolderResponse,
    WatchlistSymbolRecord,
    WatchlistSymbolResponse,
    WatchlistWorkspaceResponse,
)

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,16}$")
_MAX_FOLDERS = 50
_MAX_SYMBOLS_PER_FOLDER = 100
_MAX_TOTAL_SYMBOLS = 500


class WatchlistAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.folder_table = "WATCHLIST_FOLDER"
        self.item_table = "WATCHLIST_ITEM"

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.strip().upper()

    @staticmethod
    def _validate_workspace(folders: list[WatchlistFolderRecord]) -> None:
        if len(folders) > _MAX_FOLDERS:
            raise ValueError(f"At most {_MAX_FOLDERS} folders allowed")

        total_symbols = 0
        folder_ids: set[str] = set()
        for folder in folders:
            if not folder.id or folder.id in folder_ids:
                raise ValueError("Each folder must have a unique id")
            folder_ids.add(folder.id)
            if not folder.name.strip():
                raise ValueError("Folder name is required")
            if len(folder.symbols) > _MAX_SYMBOLS_PER_FOLDER:
                raise ValueError(
                    f"At most {_MAX_SYMBOLS_PER_FOLDER} symbols per folder"
                )
            symbol_ids: set[str] = set()
            tickers: set[str] = set()
            for item in folder.symbols:
                if not item.id or item.id in symbol_ids:
                    raise ValueError("Each symbol row must have a unique id")
                symbol_ids.add(item.id)
                ticker = WatchlistAdapter._normalize_symbol(item.ticker)
                if not _SYMBOL_PATTERN.match(ticker):
                    raise ValueError(f"Invalid symbol: {item.ticker}")
                if ticker in tickers:
                    raise ValueError(
                        f"Duplicate symbol {ticker} in folder {folder.name}"
                    )
                tickers.add(ticker)
                total_symbols += 1

        if total_symbols > _MAX_TOTAL_SYMBOLS:
            raise ValueError(f"At most {_MAX_TOTAL_SYMBOLS} total symbols allowed")

    def list_workspace(self, user_id: str) -> list[WatchlistFolderRecord]:
        folder_sql = f"""
            SELECT id, name, icon_name, swatch_id, accent_hex,
                   is_pinned, is_collapsed, sort_order
            FROM {self.folder_table}
            WHERE user_id = :user_id
            ORDER BY is_pinned DESC, sort_order ASC, name ASC
        """
        item_sql = f"""
            SELECT id, folder_id, symbol, sort_order
            FROM {self.item_table}
            WHERE user_id = :user_id
            ORDER BY sort_order ASC, symbol ASC
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(folder_sql, {"user_id": user_id})
            folder_rows = cur.fetchall()

            cur.execute(item_sql, {"user_id": user_id})
            item_rows = cur.fetchall()
        finally:
            self.client.release(con)

        items_by_folder: dict[str, list[WatchlistSymbolRecord]] = {}
        for item_id, folder_id, symbol, sort_order in item_rows:
            items_by_folder.setdefault(folder_id, []).append(
                WatchlistSymbolRecord(
                    id=item_id,
                    ticker=self._normalize_symbol(symbol),
                    sortOrder=int(sort_order or 0),
                )
            )

        folders: list[WatchlistFolderRecord] = []
        for (
            folder_id,
            name,
            icon_name,
            swatch_id,
            accent_hex,
            is_pinned,
            is_collapsed,
            sort_order,
        ) in folder_rows:
            folders.append(
                WatchlistFolderRecord(
                    id=folder_id,
                    name=name,
                    iconName=icon_name or "folder.fill",
                    swatchID=swatch_id or "slate",
                    accentHex=int(accent_hex) if accent_hex is not None else None,
                    isPinned=bool(is_pinned),
                    isCollapsed=bool(is_collapsed),
                    sortOrder=int(sort_order or 0),
                    symbols=items_by_folder.get(folder_id, []),
                )
            )
        return folders

    def sync_workspace(
        self, user_id: str, folders: list[WatchlistFolderRecord]
    ) -> list[WatchlistFolderRecord]:
        self._validate_workspace(folders)

        incoming_folder_ids = {folder.id for folder in folders}
        incoming_item_ids = {
            item.id for folder in folders for item in folder.symbols
        }

        con = self.client.acquire()
        try:
            cur = con.cursor()

            if incoming_folder_ids:
                folder_placeholders = ", ".join(
                    f":folder_id_{index}"
                    for index, _ in enumerate(incoming_folder_ids)
                )
                folder_params = {
                    f"folder_id_{index}": folder_id
                    for index, folder_id in enumerate(incoming_folder_ids)
                }
                cur.execute(
                    f"""
                    DELETE FROM {self.folder_table}
                    WHERE user_id = :user_id
                      AND id NOT IN ({folder_placeholders})
                    """,
                    {"user_id": user_id, **folder_params},
                )
            else:
                cur.execute(
                    f"DELETE FROM {self.folder_table} WHERE user_id = :user_id",
                    {"user_id": user_id},
                )
                con.commit()
                return []

            merge_folder_sql = f"""
                MERGE INTO {self.folder_table} t
                USING (
                    SELECT :id AS id FROM dual
                ) s
                ON (t.id = s.id AND t.user_id = :user_id)
                WHEN MATCHED THEN UPDATE SET
                    name = :name,
                    icon_name = :icon_name,
                    swatch_id = :swatch_id,
                    accent_hex = :accent_hex,
                    is_pinned = :is_pinned,
                    is_collapsed = :is_collapsed,
                    sort_order = :sort_order,
                    updated_at = systimestamp
                WHEN NOT MATCHED THEN INSERT (
                    id, user_id, name, icon_name, swatch_id, accent_hex,
                    is_pinned, is_collapsed, sort_order
                ) VALUES (
                    :id, :user_id, :name, :icon_name, :swatch_id, :accent_hex,
                    :is_pinned, :is_collapsed, :sort_order
                )
            """

            for folder in folders:
                cur.execute(
                    merge_folder_sql,
                    {
                        "id": folder.id,
                        "user_id": user_id,
                        "name": folder.name.strip(),
                        "icon_name": folder.icon_name or "folder.fill",
                        "swatch_id": folder.swatch_id or "slate",
                        "accent_hex": folder.accent_hex,
                        "is_pinned": 1 if folder.is_pinned else 0,
                        "is_collapsed": 1 if folder.is_collapsed else 0,
                        "sort_order": folder.sort_order,
                    },
                )

                folder_item_ids = {item.id for item in folder.symbols}
                if folder_item_ids:
                    item_placeholders = ", ".join(
                        f":item_id_{index}"
                        for index, _ in enumerate(folder_item_ids)
                    )
                    item_params = {
                        f"item_id_{index}": item_id
                        for index, item_id in enumerate(folder_item_ids)
                    }
                    cur.execute(
                        f"""
                        DELETE FROM {self.item_table}
                        WHERE user_id = :user_id
                          AND folder_id = :folder_id
                          AND id NOT IN ({item_placeholders})
                        """,
                        {
                            "user_id": user_id,
                            "folder_id": folder.id,
                            **item_params,
                        },
                    )
                else:
                    cur.execute(
                        f"""
                        DELETE FROM {self.item_table}
                        WHERE user_id = :user_id AND folder_id = :folder_id
                        """,
                        {"user_id": user_id, "folder_id": folder.id},
                    )

                merge_item_sql = f"""
                    MERGE INTO {self.item_table} t
                    USING (
                        SELECT :id AS id FROM dual
                    ) s
                    ON (t.id = s.id AND t.user_id = :user_id)
                    WHEN MATCHED THEN UPDATE SET
                        folder_id = :folder_id,
                        symbol = :symbol,
                        sort_order = :sort_order,
                        updated_at = systimestamp
                    WHEN NOT MATCHED THEN INSERT (
                        id, user_id, folder_id, symbol, sort_order
                    ) VALUES (
                        :id, :user_id, :folder_id, :symbol, :sort_order
                    )
                """

                for item in folder.symbols:
                    cur.execute(
                        merge_item_sql,
                        {
                            "id": item.id,
                            "user_id": user_id,
                            "folder_id": folder.id,
                            "symbol": self._normalize_symbol(item.ticker),
                            "sort_order": item.sort_order,
                        },
                    )

            if incoming_item_ids:
                global_item_placeholders = ", ".join(
                    f":global_item_id_{index}"
                    for index, _ in enumerate(incoming_item_ids)
                )
                global_item_params = {
                    f"global_item_id_{index}": item_id
                    for index, item_id in enumerate(incoming_item_ids)
                }
                cur.execute(
                    f"""
                    DELETE FROM {self.item_table}
                    WHERE user_id = :user_id
                      AND id NOT IN ({global_item_placeholders})
                    """,
                    {"user_id": user_id, **global_item_params},
                )

            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            self.client.release(con)

        return self.list_workspace(user_id)

    def delete_by_user_id(self, user_id: str) -> None:
        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                f"DELETE FROM {self.item_table} WHERE user_id = :user_id",
                {"user_id": user_id},
            )
            cur.execute(
                f"DELETE FROM {self.folder_table} WHERE user_id = :user_id",
                {"user_id": user_id},
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            self.client.release(con)

    @staticmethod
    def build_response(
        folders: Iterable[WatchlistFolderRecord],
        *,
        titles_by_symbol: dict[str, str],
        quotes_by_symbol: dict[str, tuple[float, float, float]] | None = None,
    ) -> WatchlistWorkspaceResponse:
        from datetime import datetime, timezone

        quote_map = quotes_by_symbol or {}
        response_folders: list[WatchlistFolderResponse] = []

        for folder in folders:
            symbols: list[WatchlistSymbolResponse] = []
            for item in folder.symbols:
                ticker = WatchlistAdapter._normalize_symbol(item.ticker)
                title = titles_by_symbol.get(ticker, ticker)
                quote = quote_map.get(ticker)
                symbols.append(
                    WatchlistSymbolResponse(
                        id=item.id,
                        ticker=ticker,
                        sortOrder=item.sort_order,
                        companyName=title,
                        price=quote[0] if quote else None,
                        dayChange=quote[1] if quote else None,
                        dayChangePercent=quote[2] if quote else None,
                    )
                )
            response_folders.append(
                WatchlistFolderResponse(
                    id=folder.id,
                    name=folder.name,
                    iconName=folder.icon_name,
                    swatchID=folder.swatch_id,
                    accentHex=folder.accent_hex,
                    isPinned=folder.is_pinned,
                    isCollapsed=folder.is_collapsed,
                    sortOrder=folder.sort_order,
                    symbols=symbols,
                )
            )

        return WatchlistWorkspaceResponse(
            folders=response_folders,
            asOf=datetime.now(timezone.utc),
        )
