import oracledb
from typing import List

from app.models.ticker_symbol_models import TickerSymbolItem


class TickerSymbolAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "TICKER_SYMBOLS"

    def dict_to_item(self, row: dict) -> TickerSymbolItem:
        return TickerSymbolItem(
            symbol=row["SYMBOL"],
            created_at=row["CREATED_AT"],
        )

    def get_by_keyword(self, keyword: str, limit: int = 10) -> List[TickerSymbolItem]:
        keyword = keyword.strip().upper()
        if not keyword:
            return []

        con = self.client.acquire()
        try:
            cur = con.cursor()

            sql = f"""
                SELECT SYMBOL, CREATED_AT
                FROM {self.table_name}
                WHERE UPPER(SYMBOL) LIKE :pattern || '%'
                ORDER BY SYMBOL
                FETCH FIRST :limit ROWS ONLY
            """

            cur.execute(
                sql,
                {
                    "pattern": keyword,
                    "limit": limit,
                },
            )

            cols = [col[0] for col in cur.description]
            rows = cur.fetchall()
            results: List[TickerSymbolItem] = []

            for row in rows:
                row_dict = dict(zip(cols, row))
                results.append(self.dict_to_item(row_dict))

            return results
        finally:
            con.close()
