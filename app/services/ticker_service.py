from app.builders.ticker_symbol_builder import TickerSymbolBuilder
from app.models.ticker_symbol_models import TickerSymbolItem
from typing import List


class TickerService:
    def __init__(self, ticker_symbol_builder: TickerSymbolBuilder):
        self.ticker_symbol_builder = ticker_symbol_builder

    def get_symbols_by_keyword(
        self, keyword: str, limit: int = 10
    ) -> List[TickerSymbolItem]:
        return self.ticker_symbol_builder.get_symbols_by_keyword(
            keyword=keyword, limit=limit
        )
