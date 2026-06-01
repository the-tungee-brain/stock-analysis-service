from app.adapters.market.ticker_symbol_adapter import TickerSymbolAdapter
from app.models.ticker_symbol_models import TickerSymbolItem
from typing import List


class TickerSymbolBuilder:
    def __init__(self, ticker_symbol_adapter: TickerSymbolAdapter):
        self.ticker_symbol_adapter = ticker_symbol_adapter

    def get_symbols_by_keyword(
        self, keyword: str, limit: int = 10
    ) -> List[TickerSymbolItem]:
        return self.ticker_symbol_adapter.get_by_keyword(
            keyword=keyword, limit=limit
        )

    def get_by_symbol(self, symbol: str) -> TickerSymbolItem | None:
        return self.ticker_symbol_adapter.get_by_symbol(symbol=symbol)

    def get_by_symbols(self, symbols: list[str]) -> dict[str, TickerSymbolItem]:
        return self.ticker_symbol_adapter.get_by_symbols(symbols=symbols)
