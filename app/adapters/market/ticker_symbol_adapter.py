import oracledb
from typing import List

from app.models.ticker_symbol_models import TickerSymbolItem
from app.core.latency_observability import observe_dependency


class TickerSymbolAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "TICKER_SYMBOLS"

    @staticmethod
    def _coerce_db_str(value) -> str | None:
        if value is None:
            return None
        if hasattr(value, "read"):
            value = value.read()
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    def dict_to_item(self, row: dict) -> TickerSymbolItem:
        title = self._coerce_db_str(row.get("TITLE"))
        asset_type = self._coerce_db_str(row.get("ASSET_TYPE"))
        logo_url = self._coerce_db_str(row.get("LOGO_URL"))
        return TickerSymbolItem(
            symbol=row["SYMBOL"],
            title=title,
            asset_type=asset_type if asset_type else None,
            logo_url=logo_url,
        )

    def get_by_keyword(self, keyword: str, limit: int = 10) -> List[TickerSymbolItem]:
        pattern = keyword.strip().upper()
        if not pattern:
            return []

        resolved_limit = max(1, min(limit, 100))

        with observe_dependency("oracle"):
            con = self.client.acquire()
            try:
                cur = con.cursor()

                sql = f"""
                SELECT SYMBOL, TITLE, ASSET_TYPE, LOGO_URL
                FROM {self.table_name}
                WHERE UPPER(SYMBOL) LIKE :pattern || '%'
                   OR UPPER(TITLE) LIKE '%' || :pattern || '%'
                ORDER BY
                    CASE
                        WHEN UPPER(SYMBOL) = :exact THEN 0
                        WHEN UPPER(SYMBOL) LIKE :pattern || '%' THEN 1
                        ELSE 2
                    END,
                    SYMBOL
                FETCH FIRST :limit ROWS ONLY
            """

                cur.execute(
                    sql,
                    {
                        "pattern": pattern,
                        "exact": pattern,
                        "limit": resolved_limit,
                    },
                )

                cols = [col[0] for col in cur.description]
                rows = cur.fetchall()
                return [self.dict_to_item(dict(zip(cols, row))) for row in rows]
            finally:
                self.client.release(con)

    def get_by_symbols(self, symbols: list[str]) -> dict[str, TickerSymbolItem]:
        normalized = [
            symbol.strip().upper()
            for symbol in symbols
            if symbol and symbol.strip()
        ]
        unique_symbols = list(dict.fromkeys(normalized))
        if not unique_symbols:
            return {}

        placeholders = ", ".join(
            f":symbol_{index}" for index, _ in enumerate(unique_symbols)
        )
        params = {
            f"symbol_{index}": symbol
            for index, symbol in enumerate(unique_symbols)
        }

        with observe_dependency("oracle"):
            con = self.client.acquire()
            try:
                cur = con.cursor()
                sql = f"""
                SELECT SYMBOL, TITLE, ASSET_TYPE, LOGO_URL
                FROM {self.table_name}
                WHERE UPPER(SYMBOL) IN ({placeholders})
            """
                cur.execute(sql, params)
                cols = [col[0] for col in cur.description]
                rows = cur.fetchall()
                items = [self.dict_to_item(dict(zip(cols, row))) for row in rows]
                return {item.symbol.upper(): item for item in items}
            finally:
                self.client.release(con)

    def get_by_symbol(self, symbol: str) -> TickerSymbolItem | None:
        symbol_upper = symbol.strip().upper()
        if not symbol_upper:
            return None

        with observe_dependency("oracle"):
            con = self.client.acquire()
            try:
                cur = con.cursor()
                sql = f"""
                SELECT SYMBOL, TITLE, ASSET_TYPE, LOGO_URL
                FROM {self.table_name}
                WHERE UPPER(SYMBOL) = :symbol
                FETCH FIRST 1 ROW ONLY
            """
                cur.execute(sql, {"symbol": symbol_upper})
                cols = [col[0] for col in cur.description]
                row = cur.fetchone()
                if row is None:
                    return None
                return self.dict_to_item(dict(zip(cols, row)))
            finally:
                self.client.release(con)
