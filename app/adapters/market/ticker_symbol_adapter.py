import oracledb
from typing import List

from app.models.ticker_symbol_models import TickerSymbolItem


class TickerSymbolAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "TICKER_SYMBOLS"

    def dict_to_item(self, row: dict) -> TickerSymbolItem:
        title = row.get("TITLE")
        return TickerSymbolItem(
            symbol=row["SYMBOL"],
            title=title.strip() if isinstance(title, str) and title.strip() else None,
        )

    def get_by_keyword(self, keyword: str, limit: int = 10) -> List[TickerSymbolItem]:
        pattern = keyword.strip().upper()
        if not pattern:
            return []

        resolved_limit = max(1, min(limit, 100))

        con = self.client.acquire()
        try:
            cur = con.cursor()

            sql = f"""
                SELECT SYMBOL, TITLE
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
            con.close()
