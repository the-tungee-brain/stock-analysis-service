from app.builders.symbol_search_builder import SymbolSearchBuilder
from app.models.ticker_symbol_models import TickerSymbolItem
from typing import List


class TickerSymbolBuilder:
    def __init__(self, symbol_search_builder: SymbolSearchBuilder):
        self.symbol_search_builder = symbol_search_builder

    def get_symbols_by_keyword(
        self, keyword: str, limit: int = 10
    ) -> List[TickerSymbolItem]:
        matches = self.symbol_search_builder.search(keyword, limit=limit)
        return [
            TickerSymbolItem(
                symbol=entry.symbol,
                name=entry.name,
            )
            for entry in matches
        ]
